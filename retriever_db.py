# retriever.py
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

        self.db = DBManager(config_path)
        print("Retriever initialized successfully (SQL Mode).")

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
                    lambda x: np.array(json.loads(x)) if isinstance(x, str) else (np.array(x) if x is not None else None)
                )
            return df
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return pd.DataFrame()

    def _graph_pathfinding_sql(self, seed_entity_ids: set) -> list:
        if not seed_entity_ids:
            return []

        schema = self.db.schema
        seeds_tuple = tuple(seed_entity_ids)
        if len(seeds_tuple) == 1:
            seeds_tuple = f"('{seeds_tuple[0]}')" 
        else:
            seeds_tuple = str(seeds_tuple)

        sql = text(f"""
            WITH RECURSIVE graph_path(current_id, path_ids, depth) AS (
                SELECT entity_id, ARRAY[entity_id], 0
                FROM {schema}.entities 
                WHERE entity_id IN {seeds_tuple}
                
                UNION ALL
                
                SELECT
                    CASE WHEN r.source_id = gp.current_id THEN r.target_id ELSE r.source_id END,
                    gp.path_ids || (CASE WHEN r.source_id = gp.current_id THEN r.target_id ELSE r.source_id END),
                    gp.depth + 1
                FROM graph_path gp
                JOIN {schema}.relationships r
                  ON (r.source_id = gp.current_id OR r.target_id = gp.current_id)
                WHERE gp.depth < :max_depth
                  AND NOT (CASE WHEN r.source_id = gp.current_id THEN r.target_id ELSE r.source_id END = ANY(gp.path_ids))
            )
            SELECT path_ids FROM graph_path WHERE depth >= 1; 
        """)

        engine = self.db.get_engine()
        paths = []
        try:
            with engine.connect() as conn:
                result = conn.execute(sql, {"max_depth": self.bfs_depth})
                rows = result.fetchall()
                unique_paths = set()
                for row in rows:
                    path = tuple(row[0]) 
                    unique_paths.add(path)
                paths = [list(p) for p in unique_paths]
        except Exception as e:
            print(f"⚠️ Error in SQL Pathfinding: {e}")
        
        return paths

    def _find_bridge_path_sql(self, start_node: str, target_nodes: set) -> list:
        if start_node in target_nodes:
            return [start_node]
            
        schema = self.db.schema
        targets_tuple = tuple(target_nodes)
        if not targets_tuple: return []
        
        targets_sql = str(targets_tuple)
        if len(targets_tuple) == 1: targets_sql = f"('{targets_tuple[0]}')"

        sql = text(f"""
            WITH RECURSIVE bridge_walk(current_id, path_ids, depth) AS (
                SELECT entity_id, ARRAY[entity_id], 0
                FROM {schema}.entities WHERE entity_id = :start_node
                
                UNION ALL
                
                SELECT
                    CASE WHEN r.source_id = bw.current_id THEN r.target_id ELSE r.source_id END,
                    bw.path_ids || (CASE WHEN r.source_id = bw.current_id THEN r.target_id ELSE r.source_id END),
                    bw.depth + 1
                FROM bridge_walk bw
                JOIN {schema}.relationships r
                  ON (r.source_id = bw.current_id OR r.target_id = bw.current_id)
                WHERE bw.depth < :max_depth
                  AND NOT (CASE WHEN r.source_id = bw.current_id THEN r.target_id ELSE r.source_id END = ANY(bw.path_ids))
            )
            SELECT path_ids FROM bridge_walk 
            WHERE current_id IN {targets_sql}
            ORDER BY depth ASC
            LIMIT 1;
        """)

        engine = self.db.get_engine()
        with engine.connect() as conn:
            result = conn.execute(sql, {"start_node": start_node, "max_depth": 3})
            row = result.fetchone()
            if row:
                return list(row[0])
        return None

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

    # <--- 新增: 路径去重逻辑 --->
    def _filter_redundant_paths(self, paths):
        """
        过滤冗余路径：
        如果 Path A 的节点集合是 Path B 的子集 (且 Path B 更长)，则认为 Path A 是 redundant。
        保留信息量更大的长路径。
        """
        if not paths: return []
        
        keep_indices = set(range(len(paths)))
        # 预计算节点集合
        path_sets = [set(p['path']) for p in paths]
        
        for i in range(len(paths)):
            if i not in keep_indices: continue
            
            for j in range(len(paths)):
                if i == j: continue
                
                # 如果 i 是 j 的子集
                if path_sets[i].issubset(path_sets[j]):
                    # Case 1: 真子集 (j 包含 i 且 j 比 i 长) -> 删除 i
                    if len(path_sets[i]) < len(path_sets[j]):
                        keep_indices.discard(i)
                        break 
                    
                    # Case 2: 集合完全相同 -> 保留分数高的
                    elif len(path_sets[i]) == len(path_sets[j]):
                        if paths[i]['score'] < paths[j]['score']:
                            keep_indices.discard(i)
                            break
                        elif paths[i]['score'] == paths[j]['score'] and i > j:
                            keep_indices.discard(i)
                            break
                            
        filtered_paths = [paths[i] for i in sorted(list(keep_indices))]
        if len(paths) > len(filtered_paths):
            print(f"  - Filtered {len(paths) - len(filtered_paths)} redundant sub-paths.")
            
        return filtered_paths

    def _score_paths_component_based(self, paths: list, query_embedding: np.ndarray, seed_entity_ids: set, local_entity_map: dict, local_edge_map: dict):
        if not paths: return []
        
        unique_entity_ids = {eid for path in paths for eid in path}
        unique_relation_ids = set()
        
        entity_sim_map = self._batch_get_similarity(list(unique_entity_ids), local_entity_map, query_embedding, 'embedding')
        
        for path in paths:
             for i in range(len(path) - 1):
                edge_key = tuple(sorted((path[i], path[i + 1])))
                if edge_key in local_edge_map:
                    rid = local_edge_map[edge_key]['relation_id']
                    unique_relation_ids.add(rid)
        
        local_relation_map = {
            data['relation_id']: data for data in local_edge_map.values()
        }
        relation_sim_map = self._batch_get_similarity(list(unique_relation_ids), local_relation_map, query_embedding, 'embedding')

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


    def search(self, query: str, top_k_chunks: int = 5, top_k_paths: int = 5):
        diagnostics = {}
        total_start_time = time.time()
        print(f"\n{'=' * 20} Starting New Search (SQL Mode) {'=' * 20}")
        print(f"Query: {query}")

        # --- STAGE 1: Vector Retrieval ---
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
        
        # 3. Graph Pathfinding
        initial_paths = self._graph_pathfinding_sql(seed_entity_ids)

        if not initial_paths and seed_entity_ids:
            print("  - ⚠️ No multi-hop paths found. Falling back to single-entity results.")
            initial_paths = [[seed_id] for seed_id in seed_entity_ids]

        print(f"  - Graph Channel: Found {len(initial_paths)} initial paths via SQL.")

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
        
        scored_paths = self._score_paths_component_based(initial_paths, query_embedding, seed_entity_ids, local_entity_map, local_edge_map)
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
        
        orphan_entities = entities_from_chunks - entities_from_paths
        endorsing_bridges_map = defaultdict(list)
        bridged_path_objects = []
        
        if orphan_entities and entities_from_paths:
            orphan_entity_map, _ = self._fetch_local_graph_data(orphan_entities)
            
            orphans_with_degree = [{'id': eid, 'degree': orphan_entity_map.get(eid, {}).get('degree', 0)} for eid in orphan_entities]
            sorted_orphans = sorted(orphans_with_degree, key=lambda x: x['degree'], reverse=True)
            orphans_to_process = sorted_orphans[:self.TOP_K_ORPHANS_TO_BRIDGE]
            
            node_to_initial_path_map = defaultdict(list)
            for p_info in scored_paths:
                for node in p_info['path']: node_to_initial_path_map[node].append(p_info)
            
            all_found_bridge_paths = []
            
            for rank, orphan in enumerate(orphans_to_process, 1):
                bridge_path = self._find_bridge_path_sql(orphan['id'], entities_from_paths)
                
                if bridge_path and len(bridge_path) > 1:
                    target_node = bridge_path[-1] 
                    all_found_bridge_paths.append(bridge_path)
                    
                    bridge_len = len(bridge_path) - 1
                    bonus_score = self.ENDORSEMENT_BASE_BONUS * (1.0 / rank) * (self.ENDORSEMENT_DECAY_FACTOR ** bridge_len)
                    
                    print(f"    - Bridged orphan '{orphan['id'][:8]}...' via SQL ({bridge_len}-hop). Bonus: {bonus_score:.3f}")
                    
                    if target_node in node_to_initial_path_map:
                        for target_path_info in node_to_initial_path_map[target_node]:
                            target_path_info['score'] *= (1 + bonus_score)
                            target_path_info['reason'] += f" + Endorsed"
                            canonical_key = tuple(sorted(target_path_info['path']))
                            endorsing_bridges_map[canonical_key].append(bridge_path)
            
            if all_found_bridge_paths:
                new_bridge_nodes = set()
                for p in all_found_bridge_paths: new_bridge_nodes.update(p)
                bridge_ent_map, bridge_edge_map = self._fetch_local_graph_data(new_bridge_nodes)
                local_entity_map.update(bridge_ent_map)
                local_edge_map.update(bridge_edge_map)
                
                scored_bridged_paths = self._score_paths_component_based(
                    all_found_bridge_paths, query_embedding, seed_entity_ids, local_entity_map, local_edge_map
                )
                for p_info in scored_bridged_paths: p_info['reason'] = 'Bridged Path'
                bridged_path_objects = scored_bridged_paths

        diagnostics['time_stage2_fusion'] = f"{time.time() - stage_start_time:.2f}s"
        stage_start_time = time.time() 

        # --- STAGE 3: Ranking & Formatting ---
        chunk_recommendations_from_graph = Counter()
        for eid in entities_from_paths:
            if eid in local_entity_map:
                chunks = local_entity_map[eid].get('source_chunk_ids', [])
                if chunks: chunk_recommendations_from_graph.update(chunks)
        
        graph_only_recs = {cid: count for cid, count in chunk_recommendations_from_graph.items() if cid not in initial_chunk_ids}
        top_k_recs = sorted(graph_only_recs.items(), key=lambda item: item[1], reverse=True)[:self.TOP_REC_K_FOR_SIMILARITY]
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
                     lambda x: np.array(json.loads(x)) if isinstance(x, str) else (np.array(x) if x is not None else None)
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
        
        # <--- 【修改】在此处插入冗余路径过滤 --->
        # 过滤掉子路径，保留更长的全集路径
        filtered_scored_paths = self._filter_redundant_paths(all_scored_paths)
        
        # 然后再排序和截断
        final_ranked_paths = sorted(filtered_scored_paths, key=lambda x: x['score'], reverse=True)[:top_k_paths]
        # <--- 【修改结束】 --->
        
        for p_info in final_ranked_paths:
            canonical_key = tuple(sorted(p_info['path']))
            p_info['endorsing_bridges'] = endorsing_bridges_map.get(canonical_key, [])

        results = {
            "top_paths": [self.get_path_details(p, local_entity_map, local_edge_map) for p in final_ranked_paths],
            "top_chunks": [self.get_item_details(c, local_chunk_map, local_entity_map) for c in sorted(final_chunk_scores_list, key=lambda x: x['final_score'], reverse=True)[:top_k_chunks]]
        }
        
        diagnostics['time_stage3_ranking'] = f"{time.time() - stage_start_time:.2f}s"
        diagnostics['time_total_retrieval'] = f"{time.time() - total_start_time:.2f}s"
        print(f"✅ Search complete. Total time: {diagnostics['time_total_retrieval']}.")
        
        full_results = results.copy()
        full_results['all_paths'] = all_scored_paths
        full_results['candidate_chunks'] = final_chunk_scores_list
        full_results['bridged_paths'] = bridged_path_objects
        full_results['seed_entities'] = [self.get_item_details({'id': eid}, entity_map=local_entity_map) for eid in seed_entity_ids]
        full_results['initial_chunks'] = [self.get_item_details({'id': cid, 'score': 0}, chunk_map=local_chunk_map) for cid in initial_chunk_ids]
        
        return full_results, diagnostics

    def _score_chunks(self, initial_chunk_ids, chunk_recommendations_from_graph, query_embedding, chunk_map):
        graph_only_recs = {cid: count for cid, count in chunk_recommendations_from_graph.items() if
                           cid not in initial_chunk_ids}
        top_k_recs_to_score = sorted(graph_only_recs.items(), key=lambda item: item[1], reverse=True)[
                              :self.TOP_REC_K_FOR_SIMILARITY]
        top_k_rec_ids = {cid for cid, count in top_k_recs_to_score}
        all_candidate_ids_to_score_sim = list(initial_chunk_ids | top_k_rec_ids)
        
        all_sim_scores = self._batch_get_similarity(all_candidate_ids_to_score_sim, chunk_map, query_embedding, 'embedding')
        
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
