# retriever_db.py
import pandas as pd
import numpy as np
import yaml
import json
from openai import OpenAI
from collections import Counter, defaultdict
from sklearn.preprocessing import minmax_scale
import time
from sqlalchemy import text

from db_utils import DBManager
from prompt import PROMPTS


class PathSBERetriever:

    def __init__(self, data_path=None, config_path='config.yaml'):
        print("Initializing Path-centric SBEA Retriever (SQL Native Version)...")
        self.data_path = data_path
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.embedding_client = OpenAI(api_key=config['Embedding']['api_key'], base_url=config['Embedding']['base_url'])
        self.llm_client = OpenAI(api_key=config['LLM']['api_key'], base_url=config['LLM']['base_url'])

        self.embedding_model = config['Embedding']['model_name']
        self.embedding_dim = config['Embedding']['dimensions']
        self.llm_model = config['LLM']['model_name']

        # Graph Retrieval Config
        graph_config = config.get('GraphRetrieval', {})
        self.top_p = graph_config.get('top_p_per_entity', 3)
        self.bfs_depth = graph_config.get('bfs_depth', 3)
        self.TOP_K_ORPHANS_TO_BRIDGE = graph_config.get('top_k_orphans_to_bridge', 3)
        
        # New Hyperparameters for Beam Search
        self.beam_width = graph_config.get('beam_width', 10)       # 束搜索宽度
        self.max_neighbors = graph_config.get('max_neighbors', 50) # 单节点扩展最大邻居数

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

        self.db = DBManager(config_path)
        print("Retriever initialized successfully (Beam Search Mode).")

    def _vector_search_sql(self, table_name, query_embedding, limit, id_col):
        schema = self.db.schema
        sql = text(f"""
            SELECT *, 1 - (embedding <=> :emb) as similarity
            FROM {schema}.{table_name}
            ORDER BY embedding <=> :emb
            LIMIT :limit
        """)

        emb_str = str(query_embedding.tolist())
        engine = self.db.get_engine()

        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"emb": emb_str, "limit": limit})

            if not df.empty and 'embedding' in df.columns:
                df['embedding'] = df['embedding'].apply(
                    lambda x: np.array(json.loads(x)) if isinstance(x, str) else (
                        np.array(x) if x is not None else None)
                )
            return df
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return pd.DataFrame()

    def _graph_pathfinding_beam_search(self, seed_entity_ids: set, query_embedding: np.ndarray):
        """
        基于语义相似度的束搜索 (Beam Search)。
        修复了 SQL ARRAY 语法错误。
        """
        if not seed_entity_ids:
            return [], {}

        schema = self.db.schema
        engine = self.db.get_engine()
        
        visited_memory = {}
        query_norm = np.linalg.norm(query_embedding) + 1e-10
        current_beams = [] 
        
        for seed in seed_entity_ids:
            current_beams.append({'path': [seed], 'score': 1.0})
            visited_memory[seed] = {'score': 1.0, 'path': [seed], 'source_chunk_ids': []}

        print(f"  Start Beam Search: Seeds={len(seed_entity_ids)}, Depth={self.bfs_depth}, Beam={self.beam_width}, MaxNeighbor={self.max_neighbors}")

        final_completed_paths = [] 

        for depth in range(self.bfs_depth):
            if not current_beams:
                break
            
            # 1. 提取当前层 Frontier
            frontier_ids = list(set([b['path'][-1] for b in current_beams]))
            if not frontier_ids:
                break
                
            # 2. 批量查询邻居
            # [FIXED] 使用 Python list 的字符串表示 (['a', 'b']) 来匹配 Postgres 的 ARRAY[] 语法
            # 之前的 tuple 表示 ('a', 'b') 会导致 Syntax Error
            ids_sql = str(list(frontier_ids))
            
            sql_neighbors = text(f"""
                SELECT t.fid as parent_id,
                       CASE WHEN r.source_id = t.fid THEN r.target_id ELSE r.source_id END as neighbor_id,
                       e.embedding,
                       e.source_chunk_ids
                FROM (SELECT unnest(ARRAY{ids_sql}) as fid) t
                JOIN LATERAL (
                    SELECT source_id, target_id 
                    FROM {schema}.relationships 
                    WHERE source_id = t.fid OR target_id = t.fid
                    ORDER BY frequency DESC
                    LIMIT :max_neighbors
                ) r ON true
                JOIN {schema}.entities e ON e.entity_id = (CASE WHEN r.source_id = t.fid THEN r.target_id ELSE r.source_id END)
            """)
            
            neighbors_map = defaultdict(list)
            
            try:
                with engine.connect() as conn:
                    # 注意：ids_sql 是通过 f-string 注入的，max_neighbors 是通过参数绑定的
                    result = conn.execute(sql_neighbors, {"max_neighbors": self.max_neighbors})
                    for row in result:
                        parent_id, neighbor_id, emb_val, chunk_ids_val = row[0], row[1], row[2], row[3]
                        
                        if isinstance(emb_val, str):
                            emb = np.array(json.loads(emb_val))
                        elif emb_val is not None:
                            emb = np.array(emb_val)
                        else:
                            emb = None
                            
                        chunk_ids = []
                        if isinstance(chunk_ids_val, list):
                            chunk_ids = chunk_ids_val
                        elif isinstance(chunk_ids_val, str):
                            try:
                                chunk_ids = json.loads(chunk_ids_val)
                            except:
                                pass
                                
                        neighbors_map[parent_id].append({
                            'id': neighbor_id, 
                            'embedding': emb,
                            'chunk_ids': chunk_ids
                        })
            except Exception as e:
                print(f"⚠️ Beam search SQL error: {e}")
                break
                
            # 3. 扩展路径 & 评分
            candidates = [] 
            
            for beam in current_beams:
                parent = beam['path'][-1]
                path_so_far = beam['path']
                
                if parent not in neighbors_map:
                    continue
                    
                for neighbor in neighbors_map[parent]:
                    n_id = neighbor['id']
                    if n_id in path_so_far: 
                        continue
                        
                    n_emb = neighbor['embedding']
                    if n_emb is None:
                        sim = 0.0
                    else:
                        n_norm = np.linalg.norm(n_emb) + 1e-10
                        sim = float(np.dot(query_embedding, n_emb) / (query_norm * n_norm))
                    
                    new_path = path_so_far + [n_id]
                    
                    if n_id not in visited_memory:
                         visited_memory[n_id] = {
                             'score': sim, 
                             'path': new_path, 
                             'source_chunk_ids': neighbor['chunk_ids']
                         }
                    else:
                        if sim > visited_memory[n_id]['score']:
                            visited_memory[n_id]['score'] = sim
                            visited_memory[n_id]['path'] = new_path
                    
                    candidates.append((new_path, sim))
            
            if not candidates:
                break
                
            # 4. 剪枝
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:self.beam_width]
            
            current_beams = [{'path': c[0], 'score': c[1]} for c in top_candidates]
            final_completed_paths.extend([c[0] for c in top_candidates])
            
        return final_completed_paths, visited_memory

    def _fetch_local_graph_data(self, entity_ids: set):
        if not entity_ids:
            return {}, {}

        schema = self.db.schema
        ids_tuple = tuple(entity_ids)
        ids_sql = str(ids_tuple)
        if len(ids_tuple) == 1: ids_sql = f"('{ids_tuple[0]}')"

        engine = self.db.get_engine()

        entities_sql = f"SELECT * FROM {schema}.entities WHERE entity_id IN {ids_sql}"
        with engine.connect() as conn:
            entities_df = pd.read_sql(entities_sql, conn)

        if not entities_df.empty:
            entities_df['embedding'] = entities_df['embedding'].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) else (np.array(x) if x is not None else None)
            )
            entities_df['source_chunk_ids'] = entities_df['source_chunk_ids'].apply(
                lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
            )

        local_entity_map = entities_df.set_index('entity_id').to_dict('index')

        rels_sql = f"""
            SELECT * FROM {schema}.relationships 
            WHERE source_id IN {ids_sql} AND target_id IN {ids_sql}
        """
        with engine.connect() as conn:
            rels_df = pd.read_sql(rels_sql, conn)

        if not rels_df.empty:
            rels_df['embedding'] = rels_df['embedding'].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) else (np.array(x) if x is not None else None)
            )

        local_edge_map = {}
        for _, row in rels_df.iterrows():
            edge_key = tuple(sorted((row['source_id'], row['target_id'])))
            local_edge_map[edge_key] = row.to_dict()

        return local_entity_map, local_edge_map

    def _get_canonical_path_key(self, path: list) -> frozenset:
        if len(path) < 2: return frozenset(path)
        edges = set()
        for i in range(len(path) - 1):
            edge = frozenset([path[i], path[i + 1]])
            edges.add(edge)
        return frozenset(edges)

    def _filter_redundant_paths(self, paths):
        """
        过滤冗余路径：如果 Path A 的节点集合是 Path B 的子集 (且 Path B 更长)，则删除 Path A。
        """
        if not paths: return []

        keep_indices = set(range(len(paths)))
        path_sets = [set(p['path']) for p in paths]

        for i in range(len(paths)):
            if i not in keep_indices: continue

            for j in range(len(paths)):
                if i == j: continue

                if path_sets[i].issubset(path_sets[j]):
                    if len(path_sets[i]) < len(path_sets[j]):
                        keep_indices.discard(i)
                        break
                    elif len(path_sets[i]) == len(path_sets[j]):
                        if paths[i]['score'] < paths[j]['score']:
                            keep_indices.discard(i)
                            break
                        elif paths[i]['score'] == paths[j]['score'] and i > j:
                            keep_indices.discard(i)
                            break

        filtered_paths = [paths[i] for i in sorted(list(keep_indices))]
        return filtered_paths

    def _score_paths_component_based(self, paths: list, query_embedding: np.ndarray, seed_entity_ids: set,
                                     local_entity_map: dict, local_edge_map: dict):
        if not paths: return []

        unique_entity_ids = {eid for path in paths for eid in path}
        unique_relation_ids = set()

        entity_sim_map = self._batch_get_similarity(list(unique_entity_ids), local_entity_map, query_embedding,
                                                    'embedding')

        for path in paths:
            for i in range(len(path) - 1):
                edge_key = tuple(sorted((path[i], path[i + 1])))
                if edge_key in local_edge_map:
                    rid = local_edge_map[edge_key]['relation_id']
                    unique_relation_ids.add(rid)

        local_relation_map = {
            data['relation_id']: data for data in local_edge_map.values()
        }
        relation_sim_map = self._batch_get_similarity(list(unique_relation_ids), local_relation_map, query_embedding,
                                                      'embedding')

        final_scored_paths = []
        for path in paths:
            total_component_score = 0
            for eid in path:
                sim = entity_sim_map.get(eid, 0)
                deg_val = local_entity_map.get(eid, {}).get('degree')
                degree = deg_val if deg_val is not None else 0
                total_component_score += sim * (1 + self.ENTITY_DEGREE_WEIGHT * degree)

            if len(path) > 1:
                for i in range(len(path) - 1):
                    edge_key = tuple(sorted((path[i], path[i + 1])))
                    edge_info = local_edge_map.get(edge_key)
                    if edge_info:
                        rel_id = edge_info['relation_id']
                        sim = relation_sim_map.get(rel_id, 0)
                        deg_val = edge_info.get('degree')
                        degree = deg_val if deg_val is not None else 0
                        total_component_score += sim * (1 + self.RELATION_DEGREE_WEIGHT * degree)

            num_components = len(path) + max(0, len(path) - 1)
            avg_quality_score = total_component_score / num_components if num_components > 0 else 0

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
        if not ids: return {}

        embeddings = []
        valid_ids = []

        for id in ids:
            item = data_map.get(id)
            if item:
                emb = item.get(emb_key)
                if isinstance(emb, (list, np.ndarray)) and len(emb) == self.embedding_dim:
                    embeddings.append(emb)
                    valid_ids.append(id)

        if not valid_ids: return {}

        embeddings_np = np.array(embeddings).astype('float32')

        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_normalized = embeddings_np / (norms + 1e-10)

        query_norm = np.linalg.norm(query_embedding)
        query_embedding_normalized = query_embedding / (query_norm + 1e-10)

        scores = np.dot(embeddings_normalized, query_embedding_normalized.T).flatten()
        return {id: float(score) for id, score in zip(valid_ids, scores)}

    def search(self, query: str, top_k_chunks: int = 5, top_k_paths: int = 10):
        diagnostics = {}
        total_start_time = time.time()
        print(f"\n{'=' * 20} Starting New Search (Beam Search Mode) {'=' * 20}")
        print(f"Query: {query}")

        # --- STAGE 1: Vector Retrieval & Entity Extraction ---
        stage_start_time = time.time()
        query_embedding = self.get_embedding(query)

        # 1. Extract Entities
        extracted_entities, usage = self._extract_entities_from_query(query)
        diagnostics['llm_extraction'] = {'entities': extracted_entities, 'usage': usage}

        search_targets = extracted_entities
        if not search_targets:
            print(f"  - ⚠️ No entities extracted from query. Fallback: Using full query as a seed entity candidate.")
            search_targets = [query]

        # 2. Find Seed Entities
        seed_entities_dict = {}
        for target in search_targets:
            if target == query:
                target_emb = query_embedding
            else:
                target_emb = self.get_embedding(target)

            df_seeds = self._vector_search_sql('entities', target_emb, limit=self.top_p, id_col='entity_id')
            for _, row in df_seeds.iterrows():
                eid = row['entity_id']
                score = row['similarity']
                if eid not in seed_entities_dict or score > seed_entities_dict[eid]['score']:
                    origin = 'initial_entity' if target != query else 'query_fallback'
                    seed_entities_dict[eid] = {'id': eid, 'score': score, 'origin': origin}

        seed_entity_ids = set(seed_entities_dict.keys())

        # 3. Graph Pathfinding (Beam Search)
        # [MODIFIED] 使用 Beam Search 替代原 SQL 递归
        # visited_memory: {node_id: {'score', 'path', 'source_chunk_ids'}}
        initial_paths, visited_memory = self._graph_pathfinding_beam_search(seed_entity_ids, query_embedding)

        if not initial_paths and seed_entity_ids:
            print("  - ⚠️ No multi-hop paths found. Falling back to single-entity results.")
            initial_paths = [[seed_id] for seed_id in seed_entity_ids]
            # 此时 visited_memory 只有种子节点，可以把它们作为 fallback 放入
            for seed in seed_entity_ids:
                if seed not in visited_memory:
                    visited_memory[seed] = {'score': 1.0, 'path': [seed], 'source_chunk_ids': []}

        print(f"  - Graph Channel: Found {len(initial_paths)} initial paths via Beam Search.")
        print(f"  - Visited Memory: Tracked {len(visited_memory)} unique nodes explored.")

        # 4. Text Channel Retrieval
        df_chunks = self._vector_search_sql('chunks', query_embedding, limit=top_k_chunks * 2, id_col='chunk_id')
        initial_chunk_ids = set(df_chunks['chunk_id'].tolist())

        if not df_chunks.empty:
            df_chunks['entity_ids'] = df_chunks['entity_ids'].apply(
                lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
            )

        local_chunk_map = df_chunks.set_index('chunk_id').to_dict('index')

        print(f"  - Text Channel: Found {len(initial_chunk_ids)} initial candidate chunks via SQL.")
        diagnostics['time_stage1_retrieval'] = f"{time.time() - stage_start_time:.2f}s"

        # --- STAGE 2: Graph Fusion & Scoring ---
        stage_start_time = time.time()

        all_path_node_ids = set()
        for p in initial_paths:
            all_path_node_ids.update(p)

        local_entity_map, local_edge_map = self._fetch_local_graph_data(all_path_node_ids)

        scored_paths = self._score_paths_component_based(initial_paths, query_embedding, seed_entity_ids,
                                                         local_entity_map, local_edge_map)
        print(f"  - Initial path scoring complete.")

        entities_from_paths = {eid for p_info in scored_paths for eid in p_info['path']}

        entities_from_chunks = set()
        for cid in initial_chunk_ids:
            if cid in local_chunk_map:
                entities = local_chunk_map[cid].get('entity_ids', [])
                if entities: entities_from_chunks.update(entities)

        for p_info in scored_paths:
            overlap = len(set(p_info['path']).intersection(entities_from_chunks))
            if overlap > 0:
                p_info['score'] += self.TEXT_CONFIRMATION_BONUS * overlap
                p_info['reason'] += f" + TextConfirm({overlap})"

        # [MODIFIED] Bridging Logic: 检查 Orphans 是否在 visited_memory 中
        orphan_entities = entities_from_chunks - entities_from_paths
        endorsing_bridges_map = defaultdict(list)
        bridged_path_objects = []

        if orphan_entities and visited_memory:
            orphans_to_process = list(orphan_entities)
            # 简单的重要性排序 (基于 degree 需要 fetch data，这里简化为只要在 visited 里就算)
            # 或者我们可以认为 visited_memory 里的 score 已经代表了重要性

            node_to_initial_path_map = defaultdict(list)
            for p_info in scored_paths:
                for node in p_info['path']: node_to_initial_path_map[node].append(p_info)

            found_bridges_count = 0
            
            for orphan in orphans_to_process:
                # 直接查内存，无需 SQL
                if orphan in visited_memory:
                    mem_record = visited_memory[orphan]
                    bridge_path = mem_record['path'] # 这是从 seed 到 orphan 的路径
                    
                    if len(bridge_path) > 1:
                        bridged_path_objects.append({'path': bridge_path, 'score': mem_record['score']}) # 暂存
                        
                        target_node = bridge_path[-1] # 就是 orphan 自己，或者路径末端
                        
                        bridge_len = len(bridge_path) - 1
                        # 计算加分
                        # 注意：这里我们反向给“原始路径”加分不太容易，因为 Bridge Path 本身就是一条新发现的路径
                        # 如果 Bridge Path 连接到了某个正在被考虑的路径上的节点，可以加分
                        # 这里沿用原逻辑：如果 orphan 被桥接了，那么它所在的 Text Chunk 变得更重要，
                        # 同时这个 Bridge Path 本身也可以成为一条新的推理路径。
                        
                        # 简化处理：我们将 Bridge Path 视为一条独立发掘的高价值路径
                        # 并且，如果这个 bridge path 连接到了已有的路径网络，可以视为 Endorsement
                        # 这里只保留 path object，稍后统一评分
                        
                        found_bridges_count += 1
                        if found_bridges_count >= self.TOP_K_ORPHANS_TO_BRIDGE * 2: # 稍微放宽限制
                            break

            if bridged_path_objects:
                print(f"    - Bridged {len(bridged_path_objects)} orphans via Visited Memory check.")
                new_bridge_nodes = set()
                for p_obj in bridged_path_objects: new_bridge_nodes.update(p_obj['path'])
                
                # Fetch details needed for scoring
                bridge_ent_map, bridge_edge_map = self._fetch_local_graph_data(new_bridge_nodes)
                local_entity_map.update(bridge_ent_map)
                local_edge_map.update(bridge_edge_map)

                # 提取 path list 进行评分
                raw_bridge_paths = [p['path'] for p in bridged_path_objects]
                scored_bridged_paths = self._score_paths_component_based(
                    raw_bridge_paths, query_embedding, seed_entity_ids, local_entity_map, local_edge_map
                )
                for p_info in scored_bridged_paths: p_info['reason'] = 'Bridged Path (Visited Check)'
                bridged_path_objects = scored_bridged_paths

        diagnostics['time_stage2_fusion'] = f"{time.time() - stage_start_time:.2f}s"
        stage_start_time = time.time()

        # --- STAGE 3: Ranking & Formatting ---
        
        # [MODIFIED] Graph Enhancing Chunk: 统计全局集合 visited_memory 推荐 Chunk
        # 只要是 visited_memory 里的节点，都视为“图谱相关”，用来推荐 Chunk
        chunk_recommendations_from_graph = Counter()
        
        for eid, record in visited_memory.items():
            # record['source_chunk_ids'] 是在 Beam Search 时顺便取出的，无需额外查询
            chunks = record.get('source_chunk_ids', [])
            if chunks:
                # 可选：可以利用 record['score'] 进行加权，而不只是计数
                # 这里保持计数逻辑，但范围扩大到了 global visited
                chunk_recommendations_from_graph.update(chunks)

        graph_only_recs = {cid: count for cid, count in chunk_recommendations_from_graph.items() if
                           cid not in initial_chunk_ids}
        top_k_recs = sorted(graph_only_recs.items(), key=lambda item: item[1], reverse=True)[
                     :self.TOP_REC_K_FOR_SIMILARITY]
        extra_chunk_ids = {cid for cid, count in top_k_recs}

        if extra_chunk_ids:
            extra_ids_tuple = tuple(extra_chunk_ids)
            sql_extra = str(extra_ids_tuple)
            if len(extra_ids_tuple) == 1: sql_extra = f"('{extra_ids_tuple[0]}')"
            schema = self.db.schema
            with self.db.get_engine().connect() as conn:
                df_extra = pd.read_sql(f"SELECT * FROM {schema}.chunks WHERE chunk_id IN {sql_extra}", conn)
            if not df_extra.empty:
                df_extra['embedding'] = df_extra['embedding'].apply(
                    lambda x: np.array(json.loads(x)) if isinstance(x, str) else (
                        np.array(x) if x is not None else None)
                )
                extra_map = df_extra.set_index('chunk_id').to_dict('index')
                local_chunk_map.update(extra_map)

        all_candidate_ids, final_chunk_scores_list = self._score_chunks(
            initial_chunk_ids, chunk_recommendations_from_graph, query_embedding, local_chunk_map
        )

        for chunk in final_chunk_scores_list:
            chunk['reason'] = f"α({self.CHUNK_SCORE_ALPHA})*NormSim({chunk.get('norm_sim', 0):.2f}) + (1-α)*NormRec({chunk.get('norm_rec', 0):.2f})"

        merged_paths = {}
        paths_for_merging = scored_paths + bridged_path_objects
        for p_info in paths_for_merging:
            canonical_key = self._get_canonical_path_key(p_info['path'])
            if canonical_key not in merged_paths:
                merged_paths[canonical_key] = p_info
            else:
                if p_info['score'] > merged_paths[canonical_key]['score']:
                    merged_paths[canonical_key] = p_info

        all_scored_paths = list(merged_paths.values())

        # 路径去重
        filtered_scored_paths = self._filter_redundant_paths(all_scored_paths)

        # 排序截断
        final_ranked_paths = sorted(filtered_scored_paths, key=lambda x: x['score'], reverse=True)[:top_k_paths]

        for p_info in final_ranked_paths:
            canonical_key = tuple(sorted(p_info['path']))
            p_info['endorsing_bridges'] = endorsing_bridges_map.get(canonical_key, [])

        results = {
            "top_paths": [self.get_path_details(p, local_entity_map, local_edge_map) for p in final_ranked_paths],
            "top_chunks": [self.get_item_details(c, local_chunk_map, local_entity_map) for c in
                           sorted(final_chunk_scores_list, key=lambda x: x['final_score'], reverse=True)[:top_k_chunks]]
        }

        diagnostics['time_stage3_ranking'] = f"{time.time() - stage_start_time:.2f}s"
        diagnostics['time_total_retrieval'] = f"{time.time() - total_start_time:.2f}s"
        print(f"✅ Search complete. Total time: {diagnostics['time_total_retrieval']}.")

        full_results = results.copy()
        full_results['all_paths'] = all_scored_paths
        full_results['candidate_chunks'] = final_chunk_scores_list
        full_results['bridged_paths'] = bridged_path_objects
        full_results['seed_entities'] = [self.get_item_details({'id': eid}, entity_map=local_entity_map) for eid in
                                         seed_entity_ids]
        full_results['initial_chunks'] = [self.get_item_details({'id': cid, 'score': 0}, chunk_map=local_chunk_map) for
                                          cid in initial_chunk_ids]

        return full_results, diagnostics

    def _score_chunks(self, initial_chunk_ids, chunk_recommendations_from_graph, query_embedding, chunk_map):
        graph_only_recs = {cid: count for cid, count in chunk_recommendations_from_graph.items() if
                           cid not in initial_chunk_ids}
        top_k_recs_to_score = sorted(graph_only_recs.items(), key=lambda item: item[1], reverse=True)[
                              :self.TOP_REC_K_FOR_SIMILARITY]
        top_k_rec_ids = {cid for cid, count in top_k_recs_to_score}
        all_candidate_ids_to_score_sim = list(initial_chunk_ids | top_k_rec_ids)

        all_sim_scores = self._batch_get_similarity(all_candidate_ids_to_score_sim, chunk_map, query_embedding,
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

    def get_item_details(self, item, chunk_map=None, entity_map=None):
        item_id = item['id']
        details = item.copy()
        if 'final_score' in details: details['score'] = details['final_score']

        if item_id.startswith('ent-'):
            data = entity_map.get(item_id, {}) if entity_map else {}
            details.update(
                {'type': 'entity', 'name': data.get('entity_name', 'Unknown'), 'content': data.get('description', '')})
        else:
            data = chunk_map.get(item_id, {}) if chunk_map else {}
            doc_name = data.get('source_document_name', 'N/A')
            details.update({'type': 'chunk', 'name': f"Chunk from {doc_name}", 'source_document': doc_name,
                            'content': data.get('text', '')})
        return details

    def get_path_details(self, path_info, entity_map, edge_map):
        path_ids = path_info['path']
        path_segments = []
        path_readable_parts = [entity_map.get(path_ids[0], {}).get('entity_name', 'Unknown')]

        for i in range(len(path_ids) - 1):
            source_id, target_id = path_ids[i], path_ids[i + 1]
            edge_key = tuple(sorted((source_id, target_id)))
            edge_info = edge_map.get(edge_key, {})

            source_name = entity_map.get(source_id, {}).get('entity_name', 'Unknown')
            target_name = entity_map.get(target_id, {}).get('entity_name', 'Unknown')
            keywords = edge_info.get('keywords', 'N/A')

            path_readable_parts.extend([f" --[{keywords}]--> ", target_name])
            path_segments.append({"source": source_name, "target": target_name, "keywords": keywords,
                                  "description": edge_info.get('description', 'N/A'),
                                  "source_desc": entity_map.get(source_id, {}).get('description', ''),
                                  "target_desc": entity_map.get(target_id, {}).get('description', '')})

        details = {"path_readable": "".join(path_readable_parts), "segments": path_segments,
                   "score": path_info['score'], "reason": path_info['reason'], "entity_ids": path_ids}

        if path_info.get('endorsing_bridges'):
            details['endorsing_bridges'] = []
            for bridge in path_info['endorsing_bridges']:
                bridge_readable = " -> ".join(
                    [entity_map.get(eid, {}).get('entity_name', 'Unknown') for eid in bridge])
                details['endorsing_bridges'].append(bridge_readable)
        return details

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
