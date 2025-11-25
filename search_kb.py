import argparse
import yaml
import pandas as pd
import json
from sqlalchemy import text
from db_utils import DBManager
import sys
from psycopg2.extras import execute_values


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


def delete_document(doc_name, config_path='config.yaml'):
    """删除指定文档及其关联的所有数据"""
    print(f"🔍 正在初始化数据库连接...")
    try:
        db = DBManager(config_path)
        # 使用 psycopg2 原生连接以获得更好的事务控制和 execute_values 支持
        conn = db.get_conn()
        conn.autocommit = False  # 开启事务
        schema = db.schema
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    print(f"🗑️  准备删除文档: '{doc_name}' (知识库: {schema})")
    print("=" * 60)

    try:
        with conn.cursor() as cur:
            # 1. 查找属于该文档的所有 Chunk IDs
            print(f"1️⃣  正在查找相关 Chunks...")
            cur.execute(f"SELECT chunk_id FROM {schema}.chunks WHERE source_document_name = %s", (doc_name,))
            rows = cur.fetchall()

            if not rows:
                print(f"⚠️  未找到名为 '{doc_name}' 的文档数据。请检查文件名（包含扩展名）。")
                return

            deleted_chunk_ids = set(row[0] for row in rows)
            deleted_chunk_ids_list = list(deleted_chunk_ids)  # 用于 SQL 参数
            print(f"   ✅ 找到 {len(deleted_chunk_ids)} 个 Chunks。")

            # 定义处理实体和关系的通用函数
            def process_table(table_name, id_col, name_col):
                print(f"\n2️⃣  正在检查受影响的 {table_name}...")

                # 使用 JSONB 操作符 ?| 查找包含任意待删除 Chunk ID 的行
                # 这比 Python 循环过滤全表要高效得多
                query = f"""
                    SELECT {id_col}, {name_col}, source_chunk_ids 
                    FROM {schema}.{table_name} 
                    WHERE source_chunk_ids ?| %s
                """
                cur.execute(query, (deleted_chunk_ids_list,))
                candidates = cur.fetchall()

                if not candidates:
                    print(f"   ℹ️  没有 {table_name} 受到影响。")
                    return

                to_update = []  # [(id, new_json_str), ...]
                to_delete = []  # [id, ...]

                for row in candidates:
                    row_id, row_name, src_chunks = row

                    # 过滤 chunk list
                    if isinstance(src_chunks, str): src_chunks = json.loads(src_chunks)
                    if not isinstance(src_chunks, list): src_chunks = []

                    new_chunks = [c for c in src_chunks if c not in deleted_chunk_ids]

                    if not new_chunks:
                        # 如果来源列表空了，说明该实体/关系仅来源于被删除的文档 -> 删除
                        to_delete.append(row_id)
                    elif len(new_chunks) != len(src_chunks):
                        # 否则 -> 更新
                        to_update.append((row_id, json.dumps(new_chunks)))

                # 执行更新
                if to_update:
                    print(f"   📝 更新 {len(to_update)} 个 {table_name} (移除引用源)...")
                    update_sql = f"""
                        UPDATE {schema}.{table_name} AS t
                        SET source_chunk_ids = v.new_ids::jsonb,
                            -- 可选：如果 frequency 是基于 chunk 计数的，这里可能需要递减，
                            -- 但由于 freq 逻辑较复杂，暂只处理引用 ID，保证图谱连通性正确。
                            frequency = GREATEST(1, cardinality(ARRAY(SELECT jsonb_array_elements_text(v.new_ids::jsonb))))
                        FROM (VALUES %s) AS v(id, new_ids)
                        WHERE t.{id_col} = v.id
                    """
                    execute_values(cur, update_sql, to_update)

                # 执行删除
                if to_delete:
                    print(f"   🗑️  删除 {len(to_delete)} 个 {table_name} (引用源归零)...")
                    cur.execute(f"DELETE FROM {schema}.{table_name} WHERE {id_col} = ANY(%s)", (to_delete,))

            # 2. 处理实体
            process_table('entities', 'entity_id', 'entity_name')

            # 3. 处理关系
            process_table('relationships', 'relation_id', 'relation_id')  # relation_id 既是ID也是占位名

            # 4. 删除 Chunks
            print(f"\n4️⃣  正在物理删除 Chunks...")
            cur.execute(f"DELETE FROM {schema}.chunks WHERE chunk_id = ANY(%s)", (deleted_chunk_ids_list,))

            conn.commit()
            print(f"\n✅ 删除操作成功完成！文档 '{doc_name}' 已从知识库移除。")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ 发生错误，已回滚所有操作: {e}")
    finally:
        conn.close()


def _batch_fetch_chunks(engine, schema, chunk_ids):
    """批量获取 Chunk 详情"""
    if not chunk_ids: return {}
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
                chunk_map[row['chunk_id']] = {'doc': row['source_document_name'],
                                              'preview': row['text'][:50].replace('\n', ' ') + "..."}
    except Exception:
        pass
    return chunk_map


def _batch_fetch_entity_names(engine, schema, entity_ids):
    """批量获取实体名称"""
    if not entity_ids: return {}
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
    except Exception:
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

    if 'entities' in scope:
        print(f"\n[🧩 Entities / 实体]")
        sql = text(
            f"SELECT * FROM {schema}.entities WHERE entity_name ILIKE :pattern OR description ILIKE :pattern LIMIT 20")
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
                print(f"  📍 {row['entity_name']} (Type: {row['entity_type']})")
                print(f"     ID: {row['entity_id']}")
                print(f"     📊 权重(Freq): {row['frequency']} | 连接度(Degree): {row['degree']}")
                desc = row['description']
                if len(desc) > 100: desc = desc[:100] + "..."
                print(f"     📝 描述: {desc}")
                src_ids = row['source_chunk_ids']
                if isinstance(src_ids, str): src_ids = json.loads(src_ids)
                if src_ids:
                    print(f"     📄 来源 ({len(src_ids)} Chunks):")
                    docs = {}
                    for cid in src_ids:
                        info = chunk_map.get(cid, {'doc': 'Unknown', 'preview': '?'})
                        dname = info['doc']
                        if dname not in docs: docs[dname] = []
                        docs[dname].append(cid)
                    for dname, cids in docs.items():
                        print(f"       - 文档: {dname}\n         Chunks: {', '.join(cids)}")
                print("    " + "-" * 60)
            print(f"  ✅ 找到 {len(df)} 个相关实体。")
        else:
            print("  (未找到相关实体)")

    if 'relations' in scope:
        print(f"\n[🔗 Relationships / 关系]")
        sql = text(
            f"SELECT * FROM {schema}.relationships WHERE source_name ILIKE :pattern OR target_name ILIKE :pattern OR keywords ILIKE :pattern LIMIT 20")
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

    if 'chunks' in scope:
        print(f"\n[📄 Chunks / 文本块]")
        sql = text(f"SELECT * FROM {schema}.chunks WHERE text ILIKE :pattern LIMIT 10")
        with engine.connect() as conn:
            df = pd.read_sql(sql, conn, params={"pattern": search_pattern})
        if not df.empty:
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
                e_ids = row['entity_ids']
                if isinstance(e_ids, str): e_ids = json.loads(e_ids)
                if e_ids:
                    e_names = [ent_name_map.get(eid, eid) for eid in e_ids]
                    display_names = e_names[:10]
                    suffix = f"... (+{len(e_names) - 10} more)" if len(e_names) > 10 else ""
                    print(f"     🧩 包含实体 ({len(e_ids)}): {', '.join(display_names)} {suffix}")
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
    parser = argparse.ArgumentParser(description="TGS_RAG 知识库管理与搜索工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # Command: list
    parser_list = subparsers.add_parser('list', help='列出所有知识库')
    parser_list.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    # Command: search
    parser_search = subparsers.add_parser('search', help='在指定知识库中搜索')
    parser_search.add_argument("query", type=str, help="搜索关键词")
    parser_search.add_argument("--scope", type=str, default="all", help="搜索范围 (entities, relations, chunks)")
    parser_search.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    # Command: delete_doc
    parser_del = subparsers.add_parser('delete_doc', help='删除指定文档及其所有关联数据')
    parser_del.add_argument("doc_name", type=str, help="要删除的文档全名 (例如: '71 (film).md')")
    parser_del.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    args = parser.parse_args()

    if args.command == 'list':
        list_knowledge_bases(args.config)
    elif args.command == 'search':
        scope_list = ['entities', 'relations', 'chunks'] if args.scope == 'all' else [s.strip() for s in
                                                                                      args.scope.split(',')]
        search_knowledge_base(args.query, scope_list, args.config)
    elif args.command == 'delete_doc':
        delete_document(args.doc_name, args.config)
    else:
        parser.print_help()