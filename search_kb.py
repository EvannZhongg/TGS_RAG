import argparse
import yaml
import pandas as pd
from sqlalchemy import text
from db_utils import DBManager
import sys


def search_knowledge_base(query_str, scope, config_path='config.yaml'):
    """
    在知识库中搜索指定字符串
    :param query_str: 搜索关键词
    :param scope: 搜索范围列表 ['entities', 'relations', 'chunks']
    :param config_path: 配置文件路径
    """
    print(f"🔍 正在初始化数据库连接，读取配置: {config_path}...")

    try:
        # 初始化 DBManager (它会自动读取 config.yaml 中的 rag_space 确定 schema)
        db = DBManager(config_path)
        engine = db.get_engine()
        schema = db.schema
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return

    print(f"🔍 正在知识库 '{db.rag_space}' (Schema: {schema}) 中搜索: '{query_str}'")
    print(f"🎯 搜索范围: {', '.join(scope)}")
    print("=" * 60)

    search_pattern = f"%{query_str}%"

    # --- 1. 搜索实体 ---
    if 'entities' in scope:
        print(f"\n[🧩 Entities / 实体]")
        sql = text(f"""
            SELECT entity_id, entity_name, entity_type, description 
            FROM {schema}.entities 
            WHERE entity_name ILIKE :pattern OR description ILIKE :pattern
            LIMIT 50
        """)

        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"pattern": search_pattern})

            if not df.empty:
                for _, row in df.iterrows():
                    print(f"  • ID: {row['entity_id']}")
                    print(f"    名称: {row['entity_name']} ({row['entity_type']})")
                    print(f"    描述: {row['description'][:150]}..." if len(
                        str(row['description'])) > 150 else f"    描述: {row['description']}")
                    print("    " + "-" * 40)
                print(f"  ✅ 找到 {len(df)} 个相关实体。")
            else:
                print("  (未找到相关实体)")
        except Exception as e:
            print(f"  ⚠️ 查询出错: {e}")

    # --- 2. 搜索关系 ---
    if 'relations' in scope:
        print(f"\n[🔗 Relationships / 关系]")
        sql = text(f"""
            SELECT relation_id, source_name, target_name, keywords, description 
            FROM {schema}.relationships 
            WHERE source_name ILIKE :pattern 
               OR target_name ILIKE :pattern 
               OR keywords ILIKE :pattern
               OR description ILIKE :pattern
            LIMIT 50
        """)

        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"pattern": search_pattern})

            if not df.empty:
                for _, row in df.iterrows():
                    print(f"  • ID: {row['relation_id']}")
                    print(f"    路径: {row['source_name']} --[{row['keywords']}]--> {row['target_name']}")
                    print(f"    描述: {row['description'][:150]}..." if len(
                        str(row['description'])) > 150 else f"    描述: {row['description']}")
                    print("    " + "-" * 40)
                print(f"  ✅ 找到 {len(df)} 个相关关系。")
            else:
                print("  (未找到相关关系)")
        except Exception as e:
            print(f"  ⚠️ 查询出错: {e}")

    # --- 3. 搜索文本块 ---
    if 'chunks' in scope:
        print(f"\n[📄 Chunks / 文本块]")
        sql = text(f"""
            SELECT chunk_id, source_document_name, text 
            FROM {schema}.chunks 
            WHERE text ILIKE :pattern
            LIMIT 20
        """)

        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"pattern": search_pattern})

            if not df.empty:
                for _, row in df.iterrows():
                    print(f"  • ID: {row['chunk_id']}")
                    print(f"    来源: {row['source_document_name']}")

                    # 高亮显示上下文
                    content = row['text']
                    idx = content.lower().find(query_str.lower())
                    start = max(0, idx - 50)
                    end = min(len(content), idx + len(query_str) + 100)
                    preview = content[start:end].replace('\n', ' ')

                    print(f"    内容摘要: ...{preview}...")
                    print("    " + "-" * 40)
                print(f"  ✅ 找到 {len(df)} 个包含关键词的文本块。")
            else:
                print("  (未找到相关文本块)")
        except Exception as e:
            print(f"  ⚠️ 查询出错: {e}")

    print("\n" + "=" * 60)
    print("🏁 搜索完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGS_RAG 知识库搜索工具")

    # 必须参数：搜索词
    parser.add_argument("query", type=str, help="要搜索的关键词字符串")

    # 可选参数：范围配置
    parser.add_argument("--scope", type=str, default="all",
                        help="搜索范围，可选值: all, entities, relations, chunks (可用逗号分隔，如: entities,chunks)")

    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")

    args = parser.parse_args()

    # 解析 Scope
    if args.scope.lower() == "all":
        search_scope = ['entities', 'relations', 'chunks']
    else:
        search_scope = [s.strip() for s in args.scope.split(',')]

    search_knowledge_base(args.query, search_scope, args.config)