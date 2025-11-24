# fusion.py

import pandas as pd
from pathlib import Path
from typing import List, Dict
from embedding import generate_entity_embeddings, generate_relation_embeddings
from collections import Counter, defaultdict
import numpy as np
from openai import OpenAI
import json

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

    # 构造 Input 内容
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
        # print(f"   ⚡ Summarizing {len(unique_descs)} descriptions for {desc_type} '{group_name}'...")
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
    执行全局知识融合 (Database Mode)，包含 LLM 摘要优化。
    """
    print("\n🔗 Starting global knowledge fusion (Smart Deduplication)...")

    db = DBManager()

    # --- 1. 加载全局数据 ---
    global_entities_df = db.load_df('entities')
    if global_entities_df.empty:
        global_entities_df = pd.DataFrame(
            columns=['entity_id', 'entity_name', 'entity_type', 'description', 'source_chunk_ids', 'degree',
                     'frequency', 'embedding'])

    global_relations_df = db.load_df('relationships')
    if global_relations_df.empty:
        global_relations_df = pd.DataFrame(
            columns=['relation_id', 'source_id', 'source_name', 'target_id', 'target_name', 'keywords', 'description',
                     'source_chunk_ids', 'frequency', 'degree', 'embedding'])

    global_chunks_df = db.load_df('chunks')
    if global_chunks_df.empty:
        global_chunks_df = pd.DataFrame(
            columns=['chunk_id', 'text', 'token_count', 'embedding', 'source_document_name', 'entity_ids',
                     'relation_ids'])

    # --- 2. 融合实体 ---
    print(f"   - Fusing {len(all_new_entities)} new entities...")
    new_entities_df = pd.DataFrame(all_new_entities)
    if not new_entities_df.empty:
        new_entities_df['entity_name'] = new_entities_df['entity_name'].str.strip()

    combined_entities_df = pd.concat([global_entities_df, new_entities_df], ignore_index=True)

    if not combined_entities_df.empty:
        # 确保 source_chunk_ids 是列表
        combined_entities_df['source_chunk_ids'] = combined_entities_df['source_chunk_ids'].apply(
            lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else [])
        )

        final_entities_rows = []
        grouped = combined_entities_df.groupby('entity_name')

        for name, group in grouped:
            # 基础字段聚合
            first_row = group.iloc[0]
            eid = first_row['entity_id']

            # 优先保留已有的 ID (通常是 ent- 开头)
            existing_record = group[group['entity_id'].astype(str).str.startswith('ent-')]
            if not existing_record.empty:
                eid = existing_record.iloc[0]['entity_id']
            elif not str(eid).startswith('ent-'):
                eid = _get_unique_id(name, prefix="ent-")

            etype = Counter(group['entity_type']).most_common(1)[0][0]
            freq = group['frequency'].sum()
            chunks = list(set(sum(group['source_chunk_ids'], [])))

            # 描述智能融合
            new_description = _merge_and_summarize_group(
                group, name, 'description', 'entity', llm_config, token_usage
            )

            # Embedding 处理
            old_embedding = None
            old_desc = None

            if not global_entities_df.empty:
                old_rec = global_entities_df[global_entities_df['entity_name'] == name]
                if not old_rec.empty:
                    old_embedding = old_rec.iloc[0]['embedding']
                    old_desc = old_rec.iloc[0]['description']

            final_embedding = old_embedding
            # 如果描述发生变化，或者之前就没有 embedding，则置为 None 以触发重算
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
    else:
        final_entities_df = pd.DataFrame(columns=global_entities_df.columns)

    # --- 3. 融合关系 ---
    print(f"   - Fusing {len(all_new_relations)} new relations...")
    new_relations_df = pd.DataFrame(all_new_relations)
    if not new_relations_df.empty:
        new_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)
        new_relations_df['source_name'] = new_relations_df['source_name'].str.strip()
        new_relations_df['target_name'] = new_relations_df['target_name'].str.strip()

    if not new_relations_df.empty or not global_relations_df.empty:
        # 统一列名
        if 'source' in global_relations_df.columns:
            global_relations_df.rename(columns={'source': 'source_name', 'target': 'target_name'}, inplace=True)

        # 生成无向键
        for df in [global_relations_df, new_relations_df]:
            if not df.empty and 'source_name' in df.columns:
                df['source_name'] = df['source_name'].str.strip()
                df['target_name'] = df['target_name'].str.strip()
                df['key'] = df.apply(lambda row: tuple(sorted((str(row['source_name']), str(row['target_name'])))),
                                     axis=1)

        combined_relations_df = pd.concat([global_relations_df, new_relations_df], ignore_index=True)

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

            # 频率求和 (权重相加)
            freq = group['frequency'].sum()

            # 关键词合并
            all_kws = []
            for k in group['keywords']:
                if k: all_kws.extend(str(k).split(','))
            keywords = ", ".join(sorted(list(set([k.strip() for k in all_kws if k.strip()]))))

            chunks = list(set(sum(group['source_chunk_ids'], [])))

            # 描述合并
            rel_name_str = f"({src_name}, {tgt_name})"
            new_description = _merge_and_summarize_group(
                group, rel_name_str, 'description', 'relation', llm_config, token_usage
            )

            # Embedding 处理
            old_embedding = None
            old_desc = None

            if not global_relations_df.empty:
                match = global_relations_df[global_relations_df['key'] == key_tuple]
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
    else:
        final_relations_df = pd.DataFrame(columns=global_relations_df.columns)

    # --- 4. 实体补全 (Placeholders) ---
    if not final_relations_df.empty:
        relation_entity_names = set(final_relations_df['source_name']).union(set(final_relations_df['target_name']))
        existing_entity_names = set(final_entities_df['entity_name'])
        missing_names = relation_entity_names - existing_entity_names
        if missing_names:
            new_placeholder_entities = []
            for name in missing_names:
                new_placeholder_entities.append({
                    "entity_id": _get_unique_id(name, prefix="ent-"),
                    "entity_name": name,
                    "entity_type": "UNKNOWN",
                    "description": "",
                    "source_chunk_ids": [],
                    "frequency": 0,
                    "degree": 0,
                    "embedding": None
                })
            missing_entities_df = pd.DataFrame(new_placeholder_entities)
            final_entities_df = pd.concat([final_entities_df, missing_entities_df], ignore_index=True)

    # --- 5. ID 映射 ---
    if not final_relations_df.empty and not final_entities_df.empty:
        name_to_id_map = pd.Series(final_entities_df.entity_id.values, index=final_entities_df.entity_name).to_dict()
        final_relations_df['source_id'] = final_relations_df['source_name'].map(name_to_id_map)
        final_relations_df['target_id'] = final_relations_df['target_name'].map(name_to_id_map)

    # --- 6 & 7. Degree 计算 ---
    if not final_entities_df.empty and not final_relations_df.empty:
        degree_counter = defaultdict(int)
        clean_relations = final_relations_df.dropna(subset=['source_id', 'target_id'])
        for _, row in clean_relations.iterrows():
            degree_counter[row['source_id']] += 1
            degree_counter[row['target_id']] += 1
        final_entities_df['degree'] = final_entities_df['entity_id'].map(degree_counter).fillna(0).astype(int)

    if not final_relations_df.empty:
        final_relations_df['degree'] = final_relations_df['frequency']

    # --- 8. 增量向量化 ---
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
                # embedding.py 中的函数会自动组合 name + description
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

    # --- 9. 保存到数据库 ---
    # 填充 NaN 并确保列对齐
    for df in [final_entities_df, final_relations_df]:
        if not df.empty:
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

    # --- 10. Chunks 保存 ---
    print(f"   - Saving chunks mapping...")
    if 'source_document_name' not in global_chunks_df.columns:
        global_chunks_df['source_document_name'] = None
    if 'embedding' not in global_chunks_df.columns:
        global_chunks_df['embedding'] = None

    global_chunks_df = pd.concat([global_chunks_df, all_new_chunks_df]).drop_duplicates(subset=['chunk_id'],
                                                                                        keep='last').reset_index(
        drop=True)

    global_chunks_df['entity_ids'] = [[] for _ in range(len(global_chunks_df))]
    global_chunks_df['relation_ids'] = [[] for _ in range(len(global_chunks_df))]

    chunk_index_map = {cid: idx for idx, cid in enumerate(global_chunks_df['chunk_id'])}

    for _, entity in final_entities_df.iterrows():
        sources = entity.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_index_map:
                    global_chunks_df.at[chunk_index_map[chunk_id], 'entity_ids'].append(entity['entity_id'])

    for _, relation in final_relations_df.iterrows():
        sources = relation.get('source_chunk_ids', [])
        if isinstance(sources, list):
            for chunk_id in sources:
                if chunk_id in chunk_index_map:
                    global_chunks_df.at[chunk_index_map[chunk_id], 'relation_ids'].append(relation['relation_id'])

    db.save_df(global_chunks_df, 'chunks', pk_col='chunk_id')

    print("✅ Global knowledge fusion finished (Smart Deduplication & Summarization).")