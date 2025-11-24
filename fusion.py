# fusion.py

import pandas as pd
from pathlib import Path
from typing import List, Dict
from embedding import generate_entity_embeddings, generate_relation_embeddings
from collections import Counter, defaultdict
import numpy as np
from openai import OpenAI
import json
from sqlalchemy import text

from db_utils import DBManager
from extraction import _get_unique_id
from prompt import PROMPTS


def _summarize_descriptions(
    name: str,
    desc_type: str,
    desc_list: List[str],
    llm_config: Dict,
    token_usage: Dict
) -> str:
    """调用 LLM 将过长的描述列表合并为一个摘要"""
    if not llm_config or not llm_config.get('api_key'):
        return " | ".join(desc_list)[:2000]

    client = OpenAI(api_key=llm_config['api_key'], base_url=llm_config['base_url'])
    
    desc_json = "\n".join([json.dumps({"desc": d}, ensure_ascii=False) for d in desc_list])
    
    prompt = PROMPTS["summarize_entity_descriptions"].format(
        description_type="Entity" if desc_type == "entity" else "Relation",
        description_name=name,
        description_list=desc_json,
        summary_length=300,
        language="Chinese"
    )

    try:
        response = client.chat.completions.create(
            model=llm_config['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        if response.usage:
            token_usage["extraction"] += response.usage.total_tokens
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Summarization failed for {name}: {e}")
        return " | ".join(desc_list)[:1000]


def _merge_and_summarize_group(
    df_group, 
    group_name: str, 
    desc_col: str, 
    desc_type: str, 
    llm_config: Dict, 
    token_usage: Dict,
    threshold: int = 3
):
    """处理分组数据的描述合并逻辑"""
    all_descs = []
    for x in df_group[desc_col]:
        if isinstance(x, str):
            all_descs.append(x)
        elif isinstance(x, list):
            all_descs.extend(x)
    
    unique_descs = sorted(list(set([d.strip() for d in all_descs if d and d.strip()])))
    
    if not unique_descs:
        return ""

    if len(unique_descs) > threshold:
        return _summarize_descriptions(group_name, desc_type, unique_descs, llm_config, token_usage)
    else:
        return " ".join(unique_descs)


def fuse_and_update_knowledge_base(
        all_new_entities: List[Dict],
        all_new_relations: List[Dict],
        all_new_chunks_df: pd.DataFrame,
        rag_space_path: Path,
        embedding_config: Dict,
        llm_config: Dict,
        token_usage: Dict
):
    """
    执行全局知识融合 (增量模式)。
    仅查询和更新受影响的数据，不再加载全量数据库。
    """
    print("\n🔗 Starting incremental knowledge fusion...")
    
    db = DBManager()
    engine = db.get_engine()
    schema = db.schema

    # 用于后续更新 Degree 的实体 ID 集合
    touched_entity_ids = set()

    # ==========================================
    # 1. 融合实体 (Entities)
    # ==========================================
    print(f"   - Processing {len(all_new_entities)} new entities...")
    new_entities_df = pd.DataFrame(all_new_entities)
    final_entities_df = pd.DataFrame()

    if not new_entities_df.empty:
        new_entities_df['entity_name'] = new_entities_df['entity_name'].str.strip()
        
        # 1.1 按需查询：只查数据库中已存在的同名实体
        unique_names = new_entities_df['entity_name'].unique().tolist()
        existing_entities_df = pd.DataFrame()
        
        if unique_names:
            # 处理 SQL 转义
            safe_names = [n.replace("'", "''") for n in unique_names]
            # 分批查询以防止 SQL 过长 (虽然 text-list 一般不大)
            if len(safe_names) > 0:
                names_str = "', '".join(safe_names)
                sql = text(f"SELECT * FROM {schema}.entities WHERE entity_name IN ('{names_str}')")
                with engine.connect() as conn:
                    existing_entities_df = pd.read_sql(sql, conn)

        # 1.2 预处理旧数据格式
        if not existing_entities_df.empty:
             # JSON 解析
             existing_entities_df['source_chunk_ids'] = existing_entities_df['source_chunk_ids'].apply(
                 lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
             )
             # Vector 解析
             existing_entities_df['embedding'] = existing_entities_df['embedding'].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) else (np.array(x) if x is not None else None)
             )

        # 1.3 内存合并
        combined_entities_df = pd.concat([existing_entities_df, new_entities_df], ignore_index=True)
        combined_entities_df['source_chunk_ids'] = combined_entities_df['source_chunk_ids'].apply(
             lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
        )

        # 1.4 分组聚合
        final_entities_rows = []
        grouped = combined_entities_df.groupby('entity_name')
        
        for name, group in grouped:
            # 优先沿用旧 ID
            first_row = group.iloc[0]
            eid = first_row['entity_id']
            existing_record = group[group['entity_id'].astype(str).str.startswith('ent-')]
            if not existing_record.empty:
                 eid = existing_record.iloc[0]['entity_id']
            elif not str(eid).startswith('ent-'):
                 eid = _get_unique_id(name, prefix="ent-")
            
            touched_entity_ids.add(eid)

            etype = Counter(group['entity_type']).most_common(1)[0][0]
            freq = group['frequency'].sum()
            chunks = list(set(sum(group['source_chunk_ids'], [])))
            
            # 描述融合 & Embedding 检查
            new_description = _merge_and_summarize_group(
                group, name, 'description', 'entity', llm_config, token_usage
            )
            
            old_embedding = None
            old_desc = None
            if not existing_entities_df.empty:
                old_rec = existing_entities_df[existing_entities_df['entity_name'] == name]
                if not old_rec.empty:
                    old_embedding = old_rec.iloc[0]['embedding']
                    old_desc = old_rec.iloc[0]['description']
            
            final_embedding = old_embedding
            if new_description != old_desc:
                final_embedding = None 
            if old_embedding is None:
                final_embedding = None
                
            final_entities_rows.append({
                'entity_id': eid,
                'entity_name': name,
                'entity_type': etype,
                'description': new_description,
                'source_chunk_ids': chunks,
                'frequency': freq,
                'embedding': final_embedding
            })
            
        final_entities_df = pd.DataFrame(final_entities_rows)

    # ==========================================
    # 2. 融合关系 (Relations)
    # ==========================================
    print(f"   - Processing {len(all_new_relations)} new relations...")
    new_relations_df = pd.DataFrame(all_new_relations)
    final_relations_df = pd.DataFrame()

    if not new_relations_df.empty:
        new_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)
        new_relations_df['source_name'] = new_relations_df['source_name'].str.strip()
        new_relations_df['target_name'] = new_relations_df['target_name'].str.strip()
        new_relations_df['key'] = new_relations_df.apply(lambda row: tuple(sorted((str(row['source_name']), str(row['target_name'])))), axis=1)
        
        # 2.1 按需查询：查找涉及这些实体的旧关系
        # 只要 source 或 target 在本次涉及的实体名单中，就有可能发生合并
        unique_keys = new_relations_df['key'].unique().tolist()
        involved_names = set()
        for k in unique_keys:
            involved_names.add(k[0])
            involved_names.add(k[1])
            
        existing_relations_df = pd.DataFrame()
        if involved_names:
            safe_names = [n.replace("'", "''") for n in involved_names]
            names_str = "', '".join(safe_names)
            
            # 查询 source 和 target 都在 involved_names 中的关系 (这是无向图合并的最小闭包)
            sql = text(f"""
                SELECT * FROM {schema}.relationships 
                WHERE source_name IN ('{names_str}') AND target_name IN ('{names_str}')
            """)
            with engine.connect() as conn:
                existing_relations_df = pd.read_sql(sql, conn)

        # 2.2 预处理旧关系
        if not existing_relations_df.empty:
             existing_relations_df['source_chunk_ids'] = existing_relations_df['source_chunk_ids'].apply(
                 lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
             )
             existing_relations_df['embedding'] = existing_relations_df['embedding'].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) else (np.array(x) if x is not None else None)
             )
             # 生成 key 用于合并
             existing_relations_df['key'] = existing_relations_df.apply(
                 lambda row: tuple(sorted((str(row['source_name']), str(row['target_name'])))), axis=1
             )
             # 过滤：只保留 key 在 unique_keys 中的 (精确匹配)
             existing_keys_set = set(unique_keys)
             existing_relations_df = existing_relations_df[existing_relations_df['key'].isin(existing_keys_set)]

        # 2.3 内存合并
        combined_relations_df = pd.concat([existing_relations_df, new_relations_df], ignore_index=True)
        combined_relations_df['source_chunk_ids'] = combined_relations_df['source_chunk_ids'].apply(
             lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
        )

        final_relations_rows = []
        grouped = combined_relations_df.groupby('key')

        for key_tuple, group in grouped:
            first_row = group.iloc[0]
            rid = first_row['relation_id']
            existing_record = group[group['relation_id'].astype(str).str.startswith('rel-')]
            if not existing_record.empty:
                rid = existing_record.iloc[0]['relation_id']
            elif not str(rid).startswith('rel-'):
                rid = _get_unique_id(f"{key_tuple[0]}-{key_tuple[1]}", prefix="rel-")
            
            src_name = key_tuple[0]
            tgt_name = key_tuple[1]
            freq = group['frequency'].sum()
            
            all_kws = []
            for k in group['keywords']:
                if k: all_kws.extend(str(k).split(','))
            keywords = ", ".join(sorted(list(set([k.strip() for k in all_kws if k.strip()]))))
            
            chunks = list(set(sum(group['source_chunk_ids'], [])))
            
            rel_name_str = f"({src_name}, {tgt_name})"
            new_description = _merge_and_summarize_group(
                group, rel_name_str, 'description', 'relation', llm_config, token_usage
            )

            old_embedding = None
            old_desc = None
            if not existing_relations_df.empty:
                match = existing_relations_df[existing_relations_df['key'] == key_tuple]
                if not match.empty:
                    old_embedding = match.iloc[0]['embedding']
                    old_desc = match.iloc[0]['description']
            
            final_embedding = old_embedding
            if new_description != old_desc:
                final_embedding = None
            if old_embedding is None:
                final_embedding = None

            final_relations_rows.append({
                'relation_id': rid,
                'source_name': src_name,
                'target_name': tgt_name,
                'keywords': keywords,
                'description': new_description,
                'source_chunk_ids': chunks,
                'frequency': freq,
                'embedding': final_embedding
            })

        final_relations_df = pd.DataFrame(final_relations_rows)

    # ==========================================
    # 3. 实体补全 (Placeholders) & ID 映射
    # ==========================================
    if not final_relations_df.empty:
        # 确保 source_id / target_id 存在
        # 先用 final_entities_df 映射
        name_to_id = dict(zip(final_entities_df['entity_name'], final_entities_df['entity_id'])) if not final_entities_df.empty else {}
        
        # 找出还缺 ID 的实体名
        needed_names = set(final_relations_df['source_name']) | set(final_relations_df['target_name'])
        missing_names = [n for n in needed_names if n not in name_to_id]
        
        # 再次查询 DB，看这些缺失的是否早已存在但本次没更新
        if missing_names:
            safe_miss = [n.replace("'", "''") for n in missing_names]
            miss_str = "', '".join(safe_miss)
            sql = text(f"SELECT entity_name, entity_id FROM {schema}.entities WHERE entity_name IN ('{miss_str}')")
            with engine.connect() as conn:
                res = conn.execute(sql).fetchall()
                for row in res:
                    name_to_id[row[0]] = row[1]
                    touched_entity_ids.add(row[1]) # 记录这些相关实体用于 Degree 更新
        
        # 剩下的才是真缺失，创建 Placeholder
        real_missing = [n for n in missing_names if n not in name_to_id]
        if real_missing:
            new_placeholders = []
            for name in real_missing:
                eid = _get_unique_id(name, prefix="ent-")
                name_to_id[name] = eid
                touched_entity_ids.add(eid)
                new_placeholders.append({
                    "entity_id": eid,
                    "entity_name": name,
                    "entity_type": "UNKNOWN",
                    "description": "",
                    "source_chunk_ids": [],
                    "frequency": 0,
                    "degree": 0,
                    "embedding": None
                })
            
            if new_placeholders:
                final_entities_df = pd.concat([final_entities_df, pd.DataFrame(new_placeholders)], ignore_index=True)

        # 应用映射
        final_relations_df['source_id'] = final_relations_df['source_name'].map(name_to_id)
        final_relations_df['target_id'] = final_relations_df['target_name'].map(name_to_id)

    # ==========================================
    # 4. 增量 Embedding 计算
    # ==========================================
    if embedding_config.get('api_key'):
        if not final_entities_df.empty:
            def is_invalid_embedding(x):
                if x is None: return True
                if isinstance(x, (list, np.ndarray)) and len(x) == 0: return True
                return False
            
            mask = final_entities_df['embedding'].apply(is_invalid_embedding)
            entities_to_embed_df = final_entities_df[mask]
            
            if not entities_to_embed_df.empty:
                print(f"   - Generating embeddings for {len(entities_to_embed_df)} new/updated entities...")
                entities_list = entities_to_embed_df.to_dict('records')
                entities_with_embeddings, tokens_used = generate_entity_embeddings(entities_list, embedding_config)
                token_usage["embedding_entities"] += tokens_used

                temp_emb_map = {e['entity_id']: e['embedding'] for e in entities_with_embeddings}
                final_entities_df['embedding'] = final_entities_df.apply(
                    lambda row: temp_emb_map.get(row['entity_id'], row['embedding']), axis=1
                )

        if not final_relations_df.empty:
            mask = final_relations_df['embedding'].apply(is_invalid_embedding)
            relations_to_embed_df = final_relations_df[mask]
            
            if not relations_to_embed_df.empty:
                print(f"   - Generating embeddings for {len(relations_to_embed_df)} new/updated relations...")
                relations_list = relations_to_embed_df.to_dict('records')
                relations_with_embeddings, tokens_used = generate_relation_embeddings(relations_list, embedding_config)
                token_usage["embedding_relations"] += tokens_used

                temp_emb_map = {r['relation_id']: r['embedding'] for r in relations_with_embeddings}
                final_relations_df['embedding'] = final_relations_df.apply(
                    lambda row: temp_emb_map.get(row['relation_id'], row['embedding']), axis=1
                )

    # ==========================================
    # 5. 保存到数据库 (Upsert)
    # ==========================================
    # 数据清理
    for df in [final_entities_df, final_relations_df]:
        if not df.empty:
            # Degree 和 Frequency 在 Python 端暂时填0或累加值，稍后 SQL 统一更新 Degree
            if 'degree' in df.columns: df['degree'] = df['degree'].fillna(0).astype(int)
            if 'frequency' in df.columns: df['frequency'] = df['frequency'].fillna(0).astype(int)

    if not final_relations_df.empty:
        final_relation_cols = ['relation_id', 'source_id', 'source_name', 'target_id', 'target_name', 'keywords',
                               'description', 'source_chunk_ids', 'frequency', 'degree', 'embedding']
        for col in final_relation_cols:
            if col not in final_relations_df.columns: final_relations_df[col] = None
        final_relations_df = final_relations_df[final_relation_cols]

    print(f"   - Saving {len(final_entities_df)} entities and {len(final_relations_df)} relations to DB...")
    db.save_df(final_entities_df, 'entities', pk_col='entity_id')
    db.save_df(final_relations_df, 'relationships', pk_col='relation_id')

    # ==========================================
    # 6. SQL Degree 更新 (高效更新)
    # ==========================================
    if touched_entity_ids:
        print(f"   - Updating degrees for {len(touched_entity_ids)} entities via SQL...")
        ids_tuple = tuple(touched_entity_ids)
        # 如果 ID 只有一个，tuple 转换后会有尾逗号问题，手动处理 string
        ids_sql_str = str(ids_tuple) if len(ids_tuple) > 1 else f"('{list(touched_entity_ids)[0]}')"
        
        sql_degree = text(f"""
            UPDATE {schema}.entities 
            SET degree = (
                SELECT COUNT(*) 
                FROM {schema}.relationships 
                WHERE source_id = {schema}.entities.entity_id 
                   OR target_id = {schema}.entities.entity_id
            )
            WHERE entity_id IN {ids_sql_str}
        """)
        try:
            with engine.connect() as conn:
                conn.execute(sql_degree)
                conn.commit()
        except Exception as e:
            print(f"⚠️ Degree update failed: {e}")

    # ==========================================
    # 7. 保存 Chunks 并建立映射
    # ==========================================
    print(f"   - Saving chunks mapping...")
    
    # all_new_chunks_df 只有当前 batch 的 chunk
    target_chunks_df = all_new_chunks_df.copy()
    target_chunks_df['entity_ids'] = [[] for _ in range(len(target_chunks_df))]
    target_chunks_df['relation_ids'] = [[] for _ in range(len(target_chunks_df))]
    
    # 为了快速反查，建立 chunk_id -> index 映射
    chunk_index_map = {cid: idx for idx, cid in enumerate(target_chunks_df['chunk_id'])}

    # 遍历本次涉及的所有实体/关系，如果它们属于当前 Batch 的 chunk，则添加进去
    # 注意：final_entities_df 包含了本次更新的所有实体，它们的 source_chunk_ids 是全量的
    
    for _, entity in final_entities_df.iterrows():
        sources = entity.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_index_map:
                    target_chunks_df.at[chunk_index_map[chunk_id], 'entity_ids'].append(entity['entity_id'])

    for _, relation in final_relations_df.iterrows():
        sources = relation.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_index_map:
                    target_chunks_df.at[chunk_index_map[chunk_id], 'relation_ids'].append(relation['relation_id'])

    db.save_df(target_chunks_df, 'chunks', pk_col='chunk_id')
    
    print("✅ Knowledge fusion complete (Incremental).")
