import argparse
import yaml
import pandas as pd
import json
from sqlalchemy import text
from db_utils import DBManager
import sys


def list_knowledge_bases(config_path='config.yaml'):
    """列出当前数据库中的所有知识库 (Schemas)"""
    print(f"🔍 正在读取数据库配置: {config_path}...")
    try:
        # 临时初始化以获取连接引擎
        db = DBManager(config_path)
        engine = db.get_engine()

        # 查询所有非系统 Schema
        sql = text("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name NOT IN ('information_schema', 'public') 
              AND schema_name NOT LIKE 'pg_%'
        """)

        with engine.connect() as conn:
            result = conn.execute(sql)
            schemas = [row[0] for row in result]

        print("\n📚 当前存在的知识库 (Rag Spaces):")
        print("=" * 40)
        if schemas:
            for s in schemas:
                print(f"  • {s}")
        else:
            print("  (暂无自定义知识库)")
        print("=" * 40)

    except Exception as e:
        print(f"❌ 获取知识库列表失败: {e}")


def _batch_fetch_chunks(engine, schema, chunk_ids):
    """批量获取 Chunk 详情"""
    if not chunk_ids:
        return {}

    # 去重并转 tuple
    ids = tuple(set(chunk_ids))
    if not ids: return {}

    ids_sql = str(ids)
    if len(ids) == 1: ids_sql = f"('{ids[0]}')"

    sql = text(f"SELECT chunk_id, source_document_name, text FROM {schema}.chunks WHERE chunk_id IN {ids_sql}")

    chunk_map = {}
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
            for _, row in df.iterrows():
                chunk_map[row['chunk_id']] = {
                    'doc': row['source_document_name'],
                    'preview': row['text'][:50].replace('\n', ' ') + "..."
                }
    except Exception as e:
        print(f"  ⚠️ Chunk 详情获取失败: {e}")

    return chunk_map


def _batch_fetch_entity_names(engine, schema, entity_ids):
    """批量获取实体名称"""
    if not entity_ids:
        return {}

    ids = tuple(set(entity_ids))
    if not ids: return {}

    ids_sql = str(ids)
    if len(ids) == 1: ids_sql = f"('{ids[0]}')"

    sql = text(f"SELECT entity_id, entity_name FROM {schema}.entities WHERE entity_id IN {ids_sql}")

    name_map = {}
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)
            for _, row in df.iterrows():
                name_map[row['entity_id']] = row['entity_name']
    except Exception as e:
        pass
    return name_map


def search_knowledge_base(query_str, scope, config_path='config.yaml'):
    try:
        db = DBManager(config_path)
        engine = db.get_engine()
        schema = db.schema
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    print(f"🔍 搜索目标: '{query_str}' | 知识库: {db.rag_space} ({schema})")
    print("=" * 80)

    search_pattern = f"%{query_str}%"

    # --- 1. 搜索实体 ---
    if 'entities' in scope:
        print(f"\n[🧩 Entities / 实体]")
        sql = text(f"""
            SELECT * FROM {schema}.entities 
            WHERE entity_name ILIKE :pattern OR description ILIKE :pattern
            LIMIT 20
        """)

        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": search_pattern})

        if not df.empty:
            # 预处理 chunk ids
            all_chunk_ids = []
            for x in df['source_chunk_ids']:
                if isinstance(x, list):
                    all_chunk_ids.extend(x)
                elif isinstance(x, str):
                    all_chunk_ids.extend(json.loads(x))

            chunk_map = _batch_fetch_chunks(engine, schema, all_chunk_ids)

            for _, row in df.iterrows():
                print(f"  📍 {row['entity_name']} (Type: {row['entity_type']})")
                print(f"     ID: {row['entity_id']}")
                print(f"     📊 权重(Freq): {row['frequency']} | 连接度(Degree): {row['degree']}")

                desc = row['description']
                if len(desc) > 100: desc = desc[:100] + "..."
                print(f"     📝 描述: {desc}")

                # 显示来源
                src_ids = row['source_chunk_ids']
                if isinstance(src_ids, str): src_ids = json.loads(src_ids)

                if src_ids:
                    print(f"     📄 来源 ({len(src_ids)} Chunks):")
                    # 按文档聚合显示
                    docs = {}
                    for cid in src_ids:
                        info = chunk_map.get(cid, {'doc': 'Unknown', 'preview': '?'})
                        dname = info['doc']
                        if dname not in docs: docs[dname] = []
                        docs[dname].append(cid)

                    for dname, cids in docs.items():
                        print(f"       - 文档: {dname}")
                        print(f"         Chunks: {', '.join(cids)}")
                print("    " + "-" * 60)
            print(f"  ✅ 找到 {len(df)} 个相关实体。")
        else:
            print("  (未找到相关实体)")

    # --- 2. 搜索关系 ---
    if 'relations' in scope:
        print(f"\n[🔗 Relationships / 关系]")
        sql = text(f"""
            SELECT * FROM {schema}.relationships 
            WHERE source_name ILIKE :pattern 
               OR target_name ILIKE :pattern 
               OR keywords ILIKE :pattern
            LIMIT 20
        """)

        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": search_pattern})

        if not df.empty:
            all_chunk_ids = []
            for x in df['source_chunk_ids']:
                if isinstance(x, list):
                    all_chunk_ids.extend(x)
                elif isinstance(x, str):
                    all_chunk_ids.extend(json.loads(x))
            chunk_map = _batch_fetch_chunks(engine, schema, all_chunk_ids)

            for _, row in df.iterrows():
                print(f"  🔗 {row['source_name']} -> {row['target_name']}")
                print(f"     ID: {row['relation_id']}")
                print(f"     🏷️  关键词: {row['keywords']}")
                print(f"     📊 权重(Freq): {row['frequency']} | 连接度(Degree): {row['degree']}")
                print(f"     📝 描述: {row['description'][:100]}..." if len(
                    str(row['description'])) > 100 else f"     📝 描述: {row['description']}")

                src_ids = row['source_chunk_ids']
                if isinstance(src_ids, str): src_ids = json.loads(src_ids)

                if src_ids:
                    print(f"     📄 来源 ({len(src_ids)} Chunks):")
                    docs = {}
                    for cid in src_ids:
                        info = chunk_map.get(cid, {'doc': 'Unknown'})
                        dname = info['doc']
                        if dname not in docs: docs[dname] = []
                        docs[dname].append(cid)
                    for dname, cids in docs.items():
                        print(f"       - {dname}: {', '.join(cids)}")
                print("    " + "-" * 60)
            print(f"  ✅ 找到 {len(df)} 个相关关系。")
        else:
            print("  (未找到相关关系)")

    # --- 3. 搜索文本块 ---
    if 'chunks' in scope:
        print(f"\n[📄 Chunks / 文本块]")
        sql = text(f"""
            SELECT * FROM {schema}.chunks 
            WHERE text ILIKE :pattern
            LIMIT 10
        """)

        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": search_pattern})

        if not df.empty:
            # 收集所有实体ID进行反查
            all_ent_ids = []
            for x in df['entity_ids']:
                if isinstance(x, list):
                    all_ent_ids.extend(x)
                elif isinstance(x, str):
                    all_ent_ids.extend(json.loads(x))

            ent_name_map = _batch_fetch_entity_names(engine, schema, all_ent_ids)

            for _, row in df.iterrows():
                print(f"  📄 ID: {row['chunk_id']}")
                print(f"     来源文档: {row['source_document_name']}")

                # 包含实体
                e_ids = row['entity_ids']
                if isinstance(e_ids, str): e_ids = json.loads(e_ids)
                if e_ids:
                    e_names = [ent_name_map.get(eid, eid) for eid in e_ids]
                    # 限制显示数量
                    display_names = e_names[:10]
                    suffix = f"... (+{len(e_names) - 10} more)" if len(e_names) > 10 else ""
                    print(f"     🧩 包含实体 ({len(e_ids)}): {', '.join(display_names)} {suffix}")

                # 高亮内容
                content = row['text']
                idx = content.lower().find(query_str.lower())
                start = max(0, idx - 60)
                end = min(len(content), idx + len(query_str) + 100)
                preview = content[start:end].replace('\n', ' ')
                print(f"     🔍 上下文: \"...{preview}...\"")
                print("    " + "-" * 60)
            print(f"  ✅ 找到 {len(df)} 个文本块。")
        else:
            print("  (未找到相关文本块)")

    print("\n🏁 搜索完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGS_RAG 知识库高级搜索工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令: list, search')

    # 命令 1: list
    parser_list = subparsers.add_parser('list', help='列出所有知识库')
    parser_list.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    # 命令 2: search
    parser_search = subparsers.add_parser('search', help='在指定知识库中搜索')
    parser_search.add_argument("query", type=str, help="搜索关键词")
    parser_search.add_argument("--scope", type=str, default="all", help="搜索范围 (entities, relations, chunks)")
    parser_search.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    args = parser.parse_args()

    if args.command == 'list':
        list_knowledge_bases(args.config)
    elif args.command == 'search':
        scope_list = ['entities', 'relations', 'chunks'] if args.scope == 'all' else [s.strip() for s in
                                                                                      args.scope.split(',')]
        search_knowledge_base(args.query, scope_list, args.config)
    else:
        parser.print_help()