import psycopg2
import psycopg2.extras
from psycopg2.extensions import register_adapter, AsIs
import pandas as pd
import numpy as np
import json
import yaml
from typing import List, Dict, Any
from sqlalchemy import create_engine
import urllib.parse


# 注册 numpy 数组适配器以适配 pgvector
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


def addapt_numpy_array(numpy_array):
    return AsIs(str(numpy_array.tolist()))


register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.ndarray, addapt_numpy_array)


class DBManager:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        db_config = config.get('Database', {})

        self.dbname = db_config.get('NAME', 'TGS_RAG')
        self.user = db_config.get('USER', 'postgres')
        self.password = db_config.get('PASSWORD', '15905190993zjr')
        self.host = db_config.get('HOST', 'localhost')
        self.port = db_config.get('PORT', '5433')
        self.rag_space = config.get('General', {}).get('rag_space', 'public')

        self.schema = self.rag_space.lower().replace('-', '_')

        safe_password = urllib.parse.quote_plus(self.password)
        self.db_url = f"postgresql+psycopg2://{self.user}:{safe_password}@{self.host}:{self.port}/{self.dbname}"

        self._init_db()

    def get_conn(self):
        return psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )

    def get_engine(self):
        return create_engine(self.db_url)

    def _init_db(self):
        conn = self.get_conn()
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema};")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT,
                token_count INT,
                embedding vector(1024),
                source_document_name TEXT,
                entity_ids JSONB,
                relation_ids JSONB
            );
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.entities (
                entity_id TEXT PRIMARY KEY,
                entity_name TEXT,
                entity_type TEXT,
                description TEXT,
                source_chunk_ids JSONB,
                degree INT,
                frequency INT,
                embedding vector(1024)
            );
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.relationships (
                relation_id TEXT PRIMARY KEY,
                source_id TEXT,
                source_name TEXT,
                target_id TEXT,
                target_name TEXT,
                keywords TEXT,
                description TEXT,
                source_chunk_ids JSONB,
                frequency INT,
                degree INT,
                embedding vector(1024)
            );
        """)

        cur.close()
        conn.close()
        print(f"✅ Database schema '{self.schema}' initialized.")

    def load_df(self, table_name: str) -> pd.DataFrame:
        engine = self.get_engine()
        query = f"SELECT * FROM {self.schema}.{table_name}"

        try:
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)

            vector_cols = ['embedding']
            json_cols = ['source_chunk_ids', 'entity_ids', 'relation_ids']

            for col in df.columns:
                if col in vector_cols:
                    df[col] = df[col].apply(lambda x: np.array(json.loads(x)) if isinstance(x, str) else (
                        np.array(x) if x is not None else None))
                elif col in json_cols:
                    df[col] = df[col].apply(
                        lambda x: x if isinstance(x, list) else (json.loads(x) if isinstance(x, str) else []))

            return df
        except Exception as e:
            print(f"⚠️ Error loading table {table_name}: {e}")
            return pd.DataFrame()
        finally:
            engine.dispose()

    def save_df(self, df: pd.DataFrame, table_name: str, pk_col: str):
        if df.empty:
            return

        conn = self.get_conn()
        cur = conn.cursor()

        df_to_save = df.copy()

        for col in df_to_save.columns:
            # 1. 类型标准化处理
            first_valid_idx = df_to_save[col].first_valid_index()
            sample = df_to_save[col].loc[first_valid_idx] if first_valid_idx is not None else None

            if sample is None:
                continue

            if isinstance(sample, np.ndarray):
                # 将 numpy 数组转为 list
                df_to_save[col] = df_to_save[col].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
            elif isinstance(sample, (list, dict)):
                # 将 list/dict 转为 JSON 字符串
                df_to_save[col] = df_to_save[col].apply(json.dumps)

        columns = list(df_to_save.columns)

        # 2. 【核心修复】安全的空值检查辅助函数
        def safe_sql_val(x):
            # 如果是容器类型（此时可能是转换后的 list 或字符串），直接返回，不进行 pd.isna 检查
            if isinstance(x, (list, tuple, dict, np.ndarray)):
                return x
            # 对于标量，使用 pd.isna 检查
            if pd.isna(x):
                return None
            return x

        # 使用辅助函数处理每一行数据
        values = [tuple(safe_sql_val(x) for x in row) for row in df_to_save.to_numpy()]

        cols_str = ', '.join(columns)
        update_sets = [f"{col} = EXCLUDED.{col}" for col in columns if col != pk_col]
        update_str = ', '.join(update_sets)

        sql = f"""
            INSERT INTO {self.schema}.{table_name} ({cols_str})
            VALUES %s
            ON CONFLICT ({pk_col}) DO UPDATE
            SET {update_str};
        """

        try:
            psycopg2.extras.execute_values(cur, sql, values, page_size=100)
            conn.commit()
            print(f"💾 Saved {len(df)} rows to {self.schema}.{table_name}")
        except Exception as e:
            conn.rollback()
            print(f"❌ Error saving to {table_name}: {e}")
            raise e
        finally:
            cur.close()
            conn.close()