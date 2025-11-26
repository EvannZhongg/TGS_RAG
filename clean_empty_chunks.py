import argparse
import json
import sys
from sqlalchemy import text
from db_utils import DBManager


def check_and_clean_empty_chunks(config_path='config.yaml'):
    print(f"🔍 正在读取数据库配置: {config_path}...")
    try:
        db = DBManager(config_path)
        engine = db.get_engine()
        schema = db.schema
        print(f"✅ 连接到知识库: {db.rag_space} (Schema: {schema})")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    print("\n🔍 正在扫描没有关联实体和关系的'空文本块'...")

    # 查询条件：entity_ids 为空/NULL/[] 且 relation_ids 为空/NULL/[]
    # 注意：JSONB 字段可能是 NULL，也可能是 '[]' 字符串
    sql_find = text(f"""
        SELECT chunk_id, source_document_name, text, entity_ids, relation_ids
        FROM {schema}.chunks
        WHERE (entity_ids IS NULL OR jsonb_array_length(entity_ids) = 0)
          AND (relation_ids IS NULL OR jsonb_array_length(relation_ids) = 0)
    """)

    empty_chunks = []

    try:
        with engine.connect() as conn:
            result = conn.execute(sql_find)
            rows = result.fetchall()

            for row in rows:
                empty_chunks.append({
                    'id': row[0],
                    'doc': row[1],
                    'text': row[2],
                })
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return

    if not empty_chunks:
        print("🎉 恭喜！当前知识库中没有'空文本块'。所有 Chunk 都包含了实体或关系。")
        return

    # --- 打印报告 ---
    print(f"\n⚠️  发现 {len(empty_chunks)} 个未包含任何知识的文本块：")
    print("=" * 80)

    # 按文档分组统计
    doc_stats = {}
    for c in empty_chunks:
        doc = c['doc']
        doc_stats[doc] = doc_stats.get(doc, 0) + 1

    print(f"📊 文档分布统计:")
    for doc, count in doc_stats.items():
        print(f"   - {doc}: {count} 个空块")

    print("\n📄 详细列表 (前 10 个示例):")
    for i, c in enumerate(empty_chunks[:10]):
        preview = c['text'][:80].replace('\n', ' ') + "..."
        print(f"   {i + 1}. [{c['id']}] ({c['doc']})")
        print(f"      \"{preview}\"")

    if len(empty_chunks) > 10:
        print(f"      ... 以及其他 {len(empty_chunks) - 10} 个")
    print("=" * 80)

    # --- 用户确认 ---
    print(f"\n❓ 是否要删除这 {len(empty_chunks)} 个空文本块？")
    print("   注意：这仅会删除 chunk 表中的记录，不会影响已提取的实体和关系。")
    confirm = input("   请输入 'yes' 确认删除，输入其他任意键取消: ").strip().lower()

    if confirm == 'yes':
        print("\n🗑️  正在执行删除...")
        chunk_ids_to_delete = tuple([c['id'] for c in empty_chunks])

        # 处理 SQL 参数格式
        ids_sql = str(chunk_ids_to_delete)
        if len(chunk_ids_to_delete) == 1:
            ids_sql = f"('{chunk_ids_to_delete[0]}')"

        sql_delete = text(f"DELETE FROM {schema}.chunks WHERE chunk_id IN {ids_sql}")

        try:
            with engine.connect() as conn:
                result = conn.execute(sql_delete)
                conn.commit()
                print(f"✅ 成功删除 {result.rowcount} 行数据。")
        except Exception as e:
            print(f"❌ 删除失败: {e}")
    else:
        print("🚫 操作已取消。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGS_RAG 清理空文本块工具")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    check_and_clean_empty_chunks(args.config)