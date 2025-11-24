# retriever.py
import pandas as pd
import numpy as np
import faiss
import yaml
import json
from openai import OpenAI
from collections import Counter, deque, defaultdict
from sklearn.preprocessing import minmax_scale
import os
import time
from db_utils import DBManager

from prompt import PROMPTS


class PathSBERetriever:

    def __init__(self, data_path='./rag_space', config_path='config.yaml'):  # data_path 参数现在可以是 None，或者保留但不使用
        print("Initializing Path-centric SBEA Retriever (Database Version)...")
        # ... (初始化 Config 部分保持不变) ...
        self.data_path = data_path
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.embedding_client = OpenAI(api_key=config['Embedding']['api_key'], base_url=config['Embedding']['base_url'])
        self.llm_client = OpenAI(api_key=config['LLM']['api_key'], base_url=config['LLM']['base_url'])
        # ... (其他参数保持不变) ...
        self.embedding_model = config['Embedding']['model_name']
        self.embedding_dim = config['Embedding']['dimensions']
        self.llm_model = config['LLM']['model_name']
        self.top_p = config['GraphRetrieval']['top_p_per_entity']
        self.bfs_depth = config['GraphRetrieval']['bfs_depth']
        self.TOP_K_ORPHANS_TO_BRIDGE = config['GraphRetrieval'].get('top_k_orphans_to_bridge', 3)
        scoring_config = config.get('Scoring', {})
        self.CHUNK_SCORE_ALPHA = scoring_config.get('chunk_score_alpha', 0.6)
        self.SEED_DENSITY_BONUS = scoring_config.get('seed_density_bonus', 0.5)
        self.TOP_REC_K_FOR_SIMILARITY = scoring_config.get('top_rec_k_for_similarity', 5)
        self.STRONG_CHUNK_RECOMMENDATION_BONUS = scoring_config.get('strong_chunk_recommendation_bonus', 0.25)
        self.WEAK_CHUNK_RECOMMENDATION_BONUS = scoring_config.get('weak_chunk_recommendation_bonus', 0.15)
        self.ENTITY_DEGREE_WEIGHT = scoring_config.get('entity_degree_weight', 0.01)
        self.RELATION_DEGREE_WEIGHT = scoring_config.get('relation_degree_weight', 0.01)
        self.TEXT_CONFIRMATION_BONUS = scoring_config.get('text_confirmation_bonus', 0.5)
        self.ENDORSEMENT_BASE_BONUS = scoring_config.get('endorsement_base_bonus', 0.1)
        self.ENDORSEMENT_DECAY_FACTOR = scoring_config.get('endorsement_decay_factor', 0.85)

        # 初始化 DBManager
        self.db = DBManager(config_path)

        self._load_data()
        self._build_indexes()
        print("Retriever initialized successfully.")

    def _load_data(self):
        print(f"Loading data from Database (Schema: {self.db.schema})...")
        # 替换原来的 read_parquet，改用 db.load_df
        self.chunks_df = self.db.load_df('chunks')
        self.entities_df = self.db.load_df('entities')
        self.relationships_df = self.db.load_df('relationships')

        print(
            f"  - Loaded {len(self.chunks_df)} chunks, {len(self.entities_df)} entities, {len(self.relationships_df)} relations.")

    def _build_indexes(self):
        print("Building in-memory indexes...")
        self.chunk_map = self.chunks_df.set_index('chunk_id').to_dict('index')
        self.entity_map = self.entities_df.set_index('entity_id').to_dict('index')
        self.relation_map = self.relationships_df.set_index('relation_id').to_dict('index')
        print("  - Building FAISS indexes (Chunks, Entities, Relations)...")
        self.chunks_index = self._create_faiss_index(self.chunks_df['embedding'].tolist())
        self.chunk_ids = self.chunks_df['chunk_id'].tolist()
        self.entities_index = self._create_faiss_index(self.entities_df['embedding'].tolist())
        self.entity_ids_faiss = self.entities_df['entity_id'].tolist()
        self.relations_index = self._create_faiss_index(self.relationships_df['embedding'].tolist())
        self.relation_ids_faiss = self.relationships_df['relation_id'].tolist()
        print("  - Building graph adjacency list and edge map...")
        self.adj_list = defaultdict(set)
        self.edge_map = {}
        valid_rels = self.relationships_df.dropna(subset=['source_id', 'target_id', 'degree', 'relation_id'])
        for _, row in valid_rels.iterrows():
            edge_key = tuple(sorted((row['source_id'], row['target_id'])))
            self.edge_map[edge_key] = {"relation_id": row['relation_id'], "degree": row['degree'],
                                       "keywords": row['keywords'], "description": row['description']}
            self.adj_list[row['source_id']].add(row['target_id'])
            self.adj_list[row['target_id']].add(row['source_id'])

    def _create_faiss_index(self, embeddings_list: list):
        zero_vector = np.zeros(self.embedding_dim, dtype=np.float32)
        cleaned_embeddings = [
            emb if isinstance(emb, (list, np.ndarray)) and len(emb) == self.embedding_dim else zero_vector for emb in
            embeddings_list]
        if len(cleaned_embeddings) != len(embeddings_list):
            print(
                f"    ⚠️ Warning: Found and replaced {len(embeddings_list) - len(cleaned_embeddings)} invalid embeddings with zero vectors.")
        embeddings = np.array(cleaned_embeddings).astype('float32')
        index = faiss.IndexFlatL2(self.embedding_dim);
        index.add(embeddings)
        return index

    def _get_canonical_path_key(self, path: list) -> frozenset:
        """为路径生成一个与方向无关的范式键（canonical key）。"""
        # 如果路径只有一个节点，则直接返回该节点ID
        if len(path) < 2:
            return frozenset(path)

        edges = set()
        for i in range(len(path) - 1):
            # 将每条边转换为一个frozenset，使得(A, B)和(B, A)等价
            edge = frozenset([path[i], path[i + 1]])
            edges.add(edge)
        # 最终返回所有边的frozenset，使得路径的遍历顺序不影响结果
        return frozenset(edges)

    # <--- 图谱路径发现算法 ---
    def _graph_pathfinding(self, seed_entity_ids: set) -> list:
        if not seed_entity_ids:
            return []

        completed_paths = []
        # 队列中存储 (当前节点ID, [当前路径上的节点列表])
        queue = deque([(seed_id, [seed_id]) for seed_id in seed_entity_ids])

        while queue:
            current_id, current_path = queue.popleft()

            # 检查可以扩展到的、尚未访问过的新邻居
            neighbors = self.adj_list.get(current_id, set())
            valid_neighbors = [n for n in neighbors if n not in current_path]

            # 终止条件1: 路径达到最大深度限制
            # 注意: 路径长度 = 跳数 + 1, 所以 bfs_depth 为1时，路径长度为2
            if len(current_path) - 1 >= self.bfs_depth:
                if len(current_path) > 1:  # 确保至少有1跳
                    completed_paths.append(current_path)
                continue

            # 终止条件2: 路径无法再扩展 (已到达叶子节点或所有邻居都已在路径中)
            if not valid_neighbors:
                if len(current_path) > 1:  # 确保至少有1跳
                    completed_paths.append(current_path)
                continue

            # 扩展路径：将所有可达的新路径加入队列
            for neighbor_id in valid_neighbors:
                new_path = current_path + [neighbor_id]
                queue.append((neighbor_id, new_path))

        # 对找到的所有完整路径进行去重后返回
        return [list(p) for p in set(tuple(path) for path in completed_paths)]

    def _score_paths_component_based(self, paths: list, query_embedding: np.ndarray, seed_entity_ids: set):
        if not paths: return []
        unique_entity_ids = {eid for path in paths for eid in path}
        unique_relation_ids = set()
        for path in paths:
            for i in range(len(path) - 1):
                edge_key = tuple(sorted((path[i], path[i + 1])))
                if edge_key in self.edge_map:
                    unique_relation_ids.add(self.edge_map[edge_key]['relation_id'])
        entity_sim_map = self._batch_get_similarity(list(unique_entity_ids), self.entity_map, query_embedding,
                                                    'embedding')
        relation_sim_map = self._batch_get_similarity(list(unique_relation_ids), self.relation_map, query_embedding,
                                                      'embedding')
        final_scored_paths = []
        for path in paths:
            # 只有长度为1的路径(孤立实体)也需要评分
            # if len(path) < 2: continue # 旧逻辑
            total_component_score = 0
            for eid in path:
                sim = entity_sim_map.get(eid, 0)
                degree = self.entity_map.get(eid, {}).get('degree', 0)
                total_component_score += sim * (1 + self.ENTITY_DEGREE_WEIGHT * degree)
            # 仅在有关系存在时才计算关系分
            if len(path) > 1:
                for i in range(len(path) - 1):
                    edge_key = tuple(sorted((path[i], path[i + 1])))
                    edge_info = self.edge_map.get(edge_key)
                    if edge_info:
                        rel_id = edge_info['relation_id']
                        sim = relation_sim_map.get(rel_id, 0)
                        degree = edge_info.get('degree', 0)
                        total_component_score += sim * (1 + self.RELATION_DEGREE_WEIGHT * degree)

            # 按长度归一化
            num_components = len(path) + max(0, len(path) - 1)
            avg_quality_score = total_component_score / num_components if num_components > 0 else 0

            # 密度奖励
            num_seeds = len(set(path) & seed_entity_ids)
            density_bonus_factor = 1.0
            if num_seeds > 1 and len(path) > 1:
                path_length = len(path) - 1
                density = num_seeds / path_length
                density_bonus_factor = 1 + self.SEED_DENSITY_BONUS * density

            base_score = avg_quality_score * density_bonus_factor
            final_scored_paths.append({'path': path, 'score': base_score,
                                       'reason': f'AvgQuality({avg_quality_score:.2f}) * DensityBonus({density_bonus_factor:.2f})'})
        return final_scored_paths

    def _batch_get_similarity(self, ids: list, data_map: dict, query_embedding: np.ndarray, emb_key: str) -> dict:
        # ... (此函数不变) ...
        if not ids: return {}
        embeddings = [data_map[id].get(emb_key) for id in ids]
        valid_embeddings, valid_ids = [], []
        for id, emb in zip(ids, embeddings):
            if isinstance(emb, (list, np.ndarray)) and len(emb) == self.embedding_dim:
                valid_embeddings.append(emb);
                valid_ids.append(id)
        if not valid_ids: return {}
        embeddings_np = np.array(valid_embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        query_embedding_normalized = query_embedding.copy();
        faiss.normalize_L2(query_embedding_normalized)
        scores = np.dot(embeddings_np, query_embedding_normalized.T).flatten()
        return {id: float(score) for id, score in zip(valid_ids, scores)}

    def _find_shortest_bridge_path(self, start_node: str, target_nodes: set) -> tuple[list | None, str | None]:
        # ... (此函数不变) ...
        if start_node in target_nodes: return [start_node], start_node
        queue = deque([[start_node]]);
        visited = {start_node}
        while queue:
            path = queue.popleft();
            node = path[-1]
            if node in target_nodes: return path, node
            for neighbor in self.adj_list.get(node, set()):
                if neighbor not in visited: visited.add(neighbor); new_path = list(path); new_path.append(
                    neighbor); queue.append(new_path)
        return None, None

    def search(self, query: str, top_k_chunks: int = 5, top_k_paths: int = 5):
        diagnostics = {}
        total_start_time = time.time()
        print(f"\n{'=' * 20} Starting New Search {'=' * 20}")
        print(f"Query: {query}")

        # STAGE 1
        stage_start_time = time.time()
        query_embedding = self.get_embedding(query).reshape(1, -1)
        extracted_entities, usage = self._extract_entities_from_query(query)
        diagnostics['llm_extraction'] = {'entities': extracted_entities, 'usage': usage}
        seed_entities_dict = self._find_seed_entities(extracted_entities)
        seed_entity_ids = set(seed_entities_dict.keys())
        initial_paths = self._graph_pathfinding(seed_entity_ids)

        if not initial_paths and seed_entity_ids:
            print("  - ⚠️ No multi-hop paths found. Falling back to single-entity results.")
            initial_paths = [[seed_id] for seed_id in seed_entity_ids]

        print(f"  - Graph Channel: Found {len(initial_paths)} initial paths from {len(seed_entities_dict)} seed(s).")
        _, I = self.chunks_index.search(query_embedding, top_k_chunks * 2)
        initial_chunk_ids = {self.chunk_ids[i] for i in I[0]}
        print(f"  - Text Channel: Found {len(initial_chunk_ids)} initial candidate chunks.")
        diagnostics['time_stage1_retrieval'] = f"{time.time() - stage_start_time:.2f}s"

        # STAGE 2
        stage_start_time = time.time()
        scored_paths = self._score_paths_component_based(initial_paths, query_embedding, seed_entity_ids)
        print(f"  - Initial path scoring complete.")

        entities_from_paths = {eid for p_info in scored_paths for eid in p_info['path']}
        initial_chunks_for_entities = [{'id': cid} for cid in initial_chunk_ids]
        entities_from_chunks = {eid for chunk in initial_chunks_for_entities for eid in
                                self.chunk_map[chunk['id']].get('entity_ids', []) if eid}
        for p_info in scored_paths:
            overlap = len(set(p_info['path']).intersection(entities_from_chunks))
            if overlap > 0: p_info['score'] += self.TEXT_CONFIRMATION_BONUS * overlap; p_info[
                'reason'] += f" + TextConfirm({overlap})"
        print(f"  - Applied text confirmation bonus to paths.")
        orphan_entities = entities_from_chunks - entities_from_paths
        endorsing_bridges_map = defaultdict(list)
        bridged_path_objects = []
        if orphan_entities and entities_from_paths:
            print(
                f"  - Found {len(orphan_entities)} orphan entities. Selecting top {self.TOP_K_ORPHANS_TO_BRIDGE} to endorse...")
            orphans_with_degree = [{'id': eid, 'degree': self.entity_map.get(eid, {}).get('degree', 0)} for eid in
                                   orphan_entities]
            sorted_orphans = sorted(orphans_with_degree, key=lambda x: x['degree'], reverse=True)
            orphans_to_process = sorted_orphans[:self.TOP_K_ORPHANS_TO_BRIDGE]
            node_to_initial_path_map = defaultdict(list)
            for p_info in scored_paths:
                for node in p_info['path']: node_to_initial_path_map[node].append(p_info)
            all_found_bridge_paths = []
            for rank, orphan in enumerate(orphans_to_process, 1):
                bridge_path, target_node = self._find_shortest_bridge_path(orphan['id'], entities_from_paths)
                if bridge_path and target_node and len(bridge_path) > 1:
                    all_found_bridge_paths.append(bridge_path)
                    bridge_len = len(bridge_path) - 1
                    bonus_score = self.ENDORSEMENT_BASE_BONUS * (1.0 / rank) * (
                                self.ENDORSEMENT_DECAY_FACTOR ** bridge_len)
                    print(
                        f"    - Bridged orphan '{orphan['id'][:8]}...' (rank {rank}) via {bridge_len}-hop path. Applying bonus: {bonus_score:.3f}")
                    for target_path_info in node_to_initial_path_map[target_node]:
                        target_path_info['score'] *= (1 + bonus_score)
                        target_path_info['reason'] += f" + Endorsed"
                        canonical_key = tuple(sorted(target_path_info['path']))
                        endorsing_bridges_map[canonical_key].append(bridge_path)
            if all_found_bridge_paths:
                scored_bridged_paths = self._score_paths_component_based(all_found_bridge_paths, query_embedding,
                                                                         seed_entity_ids)
                for p_info in scored_bridged_paths: p_info['reason'] = 'Bridged Path'
                bridged_path_objects = scored_bridged_paths
        chunk_recommendations_from_graph = Counter(
            chunk_id for eid in entities_from_paths for chunk_id in self.entity_map[eid].get('source_chunk_ids', []))
        all_candidate_ids, final_chunk_scores_list = self._score_chunks(initial_chunk_ids,
                                                                        chunk_recommendations_from_graph,
                                                                        query_embedding)
        print(f"  - Scored a total of {len(all_candidate_ids)} candidate chunks.")
        merged_paths = {}
        paths_for_merging = scored_paths + bridged_path_objects
        for p_info in paths_for_merging:
            # 使用新的辅助函数生成与方向无关的唯一键
            canonical_key = self._get_canonical_path_key(p_info['path'])

            # 如果这个范式键是第一次出现，直接存入
            if canonical_key not in merged_paths:
                merged_paths[canonical_key] = p_info
            else:
                # 如果出现了重复路径，比较当前路径与已存路径的分数
                # 仅保留分数更高的一条
                if p_info['score'] > merged_paths[canonical_key]['score']:
                    merged_paths[canonical_key] = p_info

        all_scored_paths = list(merged_paths.values())
        print(f"  - Merged paths down to {len(all_scored_paths)} unique paths.")
        diagnostics['time_stage2_fusion'] = f"{time.time() - stage_start_time:.2f}s"
        stage_start_time = time.time()

        final_ranked_paths = sorted(all_scored_paths, key=lambda x: x['score'], reverse=True)[:top_k_paths]

        for p_info in final_ranked_paths:
            canonical_key = tuple(sorted(p_info['path']))
            p_info['endorsing_bridges'] = endorsing_bridges_map.get(canonical_key, [])
        initial_chunks_display = [{'id': cid,
                                   'score': self._batch_get_similarity([cid], self.chunk_map, query_embedding,
                                                                       'embedding').get(cid, 0.0)} for cid in
                                  initial_chunk_ids]
        for chunk in final_chunk_scores_list:
            chunk[
                'reason'] = f"α({self.CHUNK_SCORE_ALPHA})*NormSim({chunk.get('norm_sim', 0):.2f}) + (1-α)*NormRec({chunk.get('norm_rec', 0):.2f})"
        final_ranked_chunks = sorted(final_chunk_scores_list, key=lambda x: x['final_score'], reverse=True)[
                              :top_k_chunks]
        print(f"  - Ranked and selected top {len(final_ranked_paths)} paths and {len(final_ranked_chunks)} chunks.")
        diagnostics['time_stage3_ranking'] = f"{time.time() - stage_start_time:.2f}s"
        results = {"seed_entities": [self.get_item_details(v) for v in seed_entities_dict.values()],
                   "initial_chunks": [self.get_item_details(c) for c in initial_chunks_display],
                   "all_paths": all_scored_paths, "bridged_paths": bridged_path_objects,
                   "candidate_chunks": [self.get_item_details(c) for c in final_chunk_scores_list],
                   "top_paths": [self.get_path_details(p) for p in final_ranked_paths],
                   "top_chunks": [self.get_item_details(c) for c in final_ranked_chunks]}
        diagnostics['time_total_retrieval'] = f"{time.time() - total_start_time:.2f}s"
        print(f"✅ Search complete. Total time: {diagnostics['time_total_retrieval']}.")
        return results, diagnostics

    def _score_chunks(self, initial_chunk_ids, chunk_recommendations_from_graph, query_embedding):
        # ... (此函数不变) ...
        graph_only_recs = {cid: count for cid, count in chunk_recommendations_from_graph.items() if
                           cid not in initial_chunk_ids}
        top_k_recs_to_score = sorted(graph_only_recs.items(), key=lambda item: item[1], reverse=True)[
                              :self.TOP_REC_K_FOR_SIMILARITY]
        top_k_rec_ids = {cid for cid, count in top_k_recs_to_score}
        all_candidate_ids_to_score_sim = list(initial_chunk_ids | top_k_rec_ids)
        all_sim_scores = self._batch_get_similarity(all_candidate_ids_to_score_sim, self.chunk_map, query_embedding,
                                                    'embedding')
        candidate_scores = {}
        for cid in all_candidate_ids_to_score_sim:
            rec_count = chunk_recommendations_from_graph.get(cid, 0)
            rec_bonus = self.STRONG_CHUNK_RECOMMENDATION_BONUS if cid in initial_chunk_ids else self.WEAK_CHUNK_RECOMMENDATION_BONUS
            candidate_scores[cid] = {'sim_score': all_sim_scores.get(cid, 0.0), 'rec_score': rec_count * rec_bonus}
        if not candidate_scores: return [], []
        scoring_df = pd.DataFrame.from_dict(candidate_scores, orient='index')
        if scoring_df['sim_score'].nunique() > 1:
            scoring_df['norm_sim'] = minmax_scale(scoring_df['sim_score'])
        else:
            scoring_df['norm_sim'] = scoring_df['sim_score'].apply(lambda x: 1.0 if x > 0 else 0.0)
        if scoring_df['rec_score'].nunique() > 1:
            scoring_df['norm_rec'] = minmax_scale(scoring_df['rec_score'])
        else:
            scoring_df['norm_rec'] = scoring_df['rec_score'].apply(lambda x: 1.0 if x > 0 else 0.0)
        alpha = self.CHUNK_SCORE_ALPHA
        scoring_df['final_score'] = (alpha * scoring_df['norm_sim']) + ((1 - alpha) * scoring_df['norm_rec'])
        final_chunk_scores_list = scoring_df.reset_index().rename(columns={'index': 'id'}).to_dict('records')
        return list(candidate_scores.keys()), final_chunk_scores_list

    def generate_answer(self, query: str, top_chunks: list, top_paths: list, mode: str):
        print(f"\n[STAGE 5] Generating final answer with mode: {mode}...")
        paths_context, chunks_context = "", ""
        if mode in ["full_context", "paths_only"]:
            paths_context = "无相关知识图谱路径。\n"
            if top_paths:
                context_parts = []
                for i, p in enumerate(top_paths):
                    context_parts.append(f"核心路径 {i + 1}: {p['path_readable']}")
                    described_entities_in_path = set()
                    for segment in p['segments']:
                        source_name, target_name = segment['source'], segment['target']
                        if source_name not in described_entities_in_path:
                            context_parts.append(f"  - 实体: {source_name} (描述: {segment['source_desc']})")
                            described_entities_in_path.add(source_name)
                        context_parts.append(
                            f"  - 关系: 从 '{source_name}' 到 '{target_name}' (描述: {segment['description']})")
                        if target_name not in described_entities_in_path:
                            context_parts.append(f"  - 实体: {target_name} (描述: {segment['target_desc']})")
                            described_entities_in_path.add(target_name)
                    if p.get('endorsing_bridges'):
                        context_parts.append("  - 该路径被以下补全证据所支持:")
                        for bridge_readable in p['endorsing_bridges']:
                            context_parts.append(f"    - 补全路径: {bridge_readable}")
                paths_context = "\n".join(context_parts)
        if mode in ["full_context", "chunks_only"]:
            chunks_context = "无相关文本证据。\n"
            if top_chunks:
                context_parts = [f"证据 {i + 1} (来源文档: {chunk['source_document']}):\n'''\n{chunk['content']}\n'''"
                                 for i, chunk in enumerate(top_chunks)]
                chunks_context = "\n\n".join(context_parts)

        prompt = PROMPTS["final_answer_prompt"].format(query=query, paths_context=paths_context,
                                                       chunks_context=chunks_context)

        try:
            return self.llm_client.chat.completions.create(model=self.llm_model,
                                                           messages=[{"role": "user", "content": prompt}],
                                                           temperature=0.3, stream=True)
        except Exception as e:
            print(f"Error during final answer generation: {e}");
            return iter([f"生成答案时出错: {e}"])

    def get_embedding(self, text: str):
        response = self.embedding_client.embeddings.create(model=self.embedding_model, input=[text])
        return np.array(response.data[0].embedding).astype('float32')

    def _extract_entities_from_query(self, query: str):
        prompt = PROMPTS["query_entity_extraction_prompt"].format(query=query)

        usage = None
        try:
            response = self.llm_client.chat.completions.create(model=self.llm_model,
                                                               messages=[{"role": "user", "content": prompt}],
                                                               temperature=0.0, max_tokens=100)
            entities = json.loads(response.choices[0].message.content)
            usage = response.usage
            return entities, usage
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return [], None

    def _find_seed_entities(self, extracted_entities: list[str]) -> dict:
        # ... (此函数不变) ...
        if not extracted_entities: return {}
        seed_entities = {}
        for entity_name in extracted_entities:
            emb = self.get_embedding(entity_name).reshape(1, -1)
            D, I = self.entities_index.search(emb, self.top_p)
            for i, d in zip(I[0], D[0]):
                eid = self.entity_ids_faiss[i];
                score = 1 - d
                if eid not in seed_entities or score > seed_entities[eid]['score']:
                    seed_entities[eid] = {'id': eid, 'score': score, 'origin': 'initial_entity'}
        return seed_entities

    def get_item_details(self, item):
        # ... (此函数不变) ...
        item_id = item['id'];
        details = item.copy()
        if 'final_score' in details: details['score'] = details['final_score']
        if item_id.startswith('ent-'):
            data = self.entity_map.get(item_id, {})
            details.update(
                {'type': 'entity', 'name': data.get('entity_name', 'Unknown'), 'content': data.get('description', '')})
        else:
            data = self.chunk_map.get(item_id, {})
            doc_name = data.get('source_document_name', 'N/A')
            details.update({'type': 'chunk', 'name': f"Chunk from {doc_name}", 'source_document': doc_name,
                            'content': data.get('text', '')})
        return details

    def get_path_details(self, path_info):
        # ... (此函数不变) ...
        path_ids = path_info['path'];
        path_segments = []
        path_readable_parts = [self.entity_map.get(path_ids[0], {}).get('entity_name', 'Unknown')]
        for i in range(len(path_ids) - 1):
            source_id, target_id = path_ids[i], path_ids[i + 1]
            edge_key = tuple(sorted((source_id, target_id)))
            edge_info = self.edge_map.get(edge_key, {})
            source_name = self.entity_map.get(source_id, {}).get('entity_name', 'Unknown')
            target_name = self.entity_map.get(target_id, {}).get('entity_name', 'Unknown')
            keywords = edge_info.get('keywords', 'N/A')
            path_readable_parts.extend([f" --[{keywords}]--> ", target_name])
            path_segments.append({"source": source_name, "target": target_name, "keywords": keywords,
                                  "description": edge_info.get('description', 'N/A'),
                                  "source_desc": self.entity_map.get(source_id, {}).get('description', ''),
                                  "target_desc": self.entity_map.get(target_id, {}).get('description', '')})
        details = {"path_readable": "".join(path_readable_parts), "segments": path_segments,
                   "score": path_info['score'], "reason": path_info['reason'], "entity_ids": path_ids}
        if path_info.get('endorsing_bridges'):
            details['endorsing_bridges'] = []
            for bridge in path_info['endorsing_bridges']:
                bridge_readable = " -> ".join(
                    [self.entity_map.get(eid, {}).get('entity_name', 'Unknown') for eid in bridge])
                details['endorsing_bridges'].append(bridge_readable)
        return details