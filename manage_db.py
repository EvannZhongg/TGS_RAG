import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import yaml
import argparse
import sys


def get_db_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('Database', {})
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        sys.exit(1)


def init_database(config_path='config.yaml'):
    """
    连接到系统默认的 'postgres' 数据库，检查并创建 'TGS_RAG' 数据库。
    """
    db_config = get_db_config(config_path)
    target_db_name = db_config.get('NAME', 'TGS_RAG')

    print(f"🔄 正在检查数据库 '{target_db_name}' 是否存在...")

    # 连接到默认的 postgres 数据库进行管理操作
    try:
        conn = psycopg2.connect(
            dbname='postgres',  # 连接到默认库
            user=db_config.get('USER', 'postgres'),
            password=db_config.get('PASSWORD'),
            host=db_config.get('HOST', 'localhost'),
            port=db_config.get('PORT', '5432')
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # 创建数据库必须在自动提交模式下
        cur = conn.cursor()

        # 检查数据库是否存在
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{target_db_name}'")
        exists = cur.fetchone()

        if not exists:
            print(f"🛠️  数据库 '{target_db_name}' 不存在，正在创建...")
            cur.execute(f'CREATE DATABASE "{target_db_name}"')
            print(f"✅ 数据库 '{target_db_name}' 创建成功！")
        else:
            print(f"✅ 数据库 '{target_db_name}' 已存在，无需创建。")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ 初始化数据库失败: {e}")
        print("💡 提示：请检查 config.yaml 中的 HOST/PORT/PASSWORD 是否正确，特别是 Docker 端口映射 (5433 -> 5432)。")


def delete_knowledge_base(kb_name, config_path='config.yaml'):
    """
    删除指定的知识库（即 Drop Schema）。
    """
    db_config = get_db_config(config_path)
    dbname = db_config.get('NAME', 'TGS_RAG')

    # 转换 rag_space 名称为 schema 名称 (逻辑需与 db_utils 保持一致)
    schema_name = kb_name.lower().replace('-', '_')

    print(f"⚠️  警告：你即将删除知识库 '{kb_name}' (Schema: {schema_name})。")
    print(f"    此操作将永久删除数据库 '{dbname}' 中该知识库下的所有实体、关系和文本块数据。")
    confirm = input("❓ 确认删除吗？(输入 'yes' 确认): ")

    if confirm.lower() != 'yes':
        print("🚫 操作已取消。")
        return

    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=db_config.get('USER', 'postgres'),
            password=db_config.get('PASSWORD'),
            host=db_config.get('HOST', 'localhost'),
            port=db_config.get('PORT', '5432')
        )
        conn.autocommit = True
        cur = conn.cursor()

        print(f"🔄 正在删除 Schema '{schema_name}'...")
        # CASCADE 会级联删除该 Schema 下的所有表 (entities, relationships, chunks)
        cur.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE;")

        print(f"✅ 知识库 '{kb_name}' (Schema: {schema_name}) 已成功删除。")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ 删除知识库失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGS_RAG 数据库管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # Init 命令
    parser_init = subparsers.add_parser('init', help='初始化创建数据库 (如果不存在)')

    # Delete 命令
    parser_delete = subparsers.add_parser('delete', help='删除指定的知识库')
    parser_delete.add_argument('name', type=str, help='要删除的 rag_space 名称 (如 my_electronics_kb)')

    args = parser.parse_args()

    if args.command == 'init':
        init_database()
    elif args.command == 'delete':
        delete_knowledge_base(args.name)
    else:
        parser.print_help()