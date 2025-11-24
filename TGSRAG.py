# TGSRAG.py

import sys
from pathlib import Path
import hashlib
import json
import yaml
import pandas as pd
import asyncio
import math

from pdf2md import process_pdf
from chunks import chunk_dispatcher
from embedding import generate_chunk_embeddings
from extraction import extract_entities_and_relations
from fusion import fuse_and_update_knowledge_base
from db_utils import DBManager


def main():
    print("🚀 Starting TGSRAG processing pipeline (Stream Processing Mode)...")

    # 初始化总token计数器
    total_token_usage = {
        "embedding_chunks": 0,
        "extraction": 0,
        "embedding_entities": 0,
        "embedding_relations": 0
    }

    input_dir = Path("D:/Personal_Project/TSG_RAG/test")  # 请根据你的实际路径修改
    # 临时目录只用于存放中间调试文件，核心数据进数据库
    output_dir_base = Path("D:/Personal_Project/TSG_RAG/test_temp")
    config_path = Path("config.yaml")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

    general_config = config.get('General', {})
    chunking_config = config.get('Chunking', {})
    embedding_config = config.get('Embedding', {})
    llm_config = config.get('LLM', {})
    extraction_config = config.get('Extraction', {})

    rag_space = general_config.get('rag_space')
    rag_space_path = Path(rag_space)

    # 获取 Batch Size 配置
    # embedding 的 batch size 通常较大（如 10-20），LLM 并发通常较小（如 4-5）
    # 我们取 embedding 的 max_batch_size 作为处理单元，或者自定义一个流水线 batch
    pipeline_batch_size = embedding_config.get('max_batch_size', 10)
    print(f"⚙️ Pipeline Batch Size set to: {pipeline_batch_size}")

    # 初始化 DBManager 检查已处理文档
    db_manager = DBManager()
    existing_chunks_df = db_manager.load_df('chunks')
    processed_docs = set(existing_chunks_df['source_document_name'].unique()) if not existing_chunks_df.empty else set()

    input_dir.mkdir(exist_ok=True)
    output_dir_base.mkdir(exist_ok=True)

    for doc_path in input_dir.iterdir():
        if not doc_path.is_file():
            continue

        print(f"\n{'=' * 50}\n-> Found document: {doc_path.name}")

        if doc_path.name in processed_docs:
            print(f"✅ Document '{doc_path.name}' has been processed before (found in DB). Skipping.")
            continue

        doc_hash_name = hashlib.md5(doc_path.read_bytes()).hexdigest()
        unique_output_dir = output_dir_base / doc_hash_name
        unique_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 读取与转换
        text_to_chunk = ""
        if doc_path.suffix.lower() == '.pdf':
            md_path = process_pdf(doc_path, unique_output_dir)
            if md_path and md_path.is_file():
                text_to_chunk = md_path.read_text(encoding='utf-8')
        elif doc_path.suffix.lower() in ['.txt', '.md']:
            text_to_chunk = doc_path.read_text(encoding='utf-8')
        else:
            print(f"ℹ️  Skipping: Unsupported file type")
            continue

        if text_to_chunk:
            # 2. 分块 (Chunking)
            print("🧠 Chunking text...")
            all_chunks = chunk_dispatcher(text_to_chunk, doc_hash_name, chunking_config)
            source_filename = doc_path.name
            for chunk in all_chunks:
                chunk['source_document_name'] = source_filename

            total_chunks = len(all_chunks)
            print(f"📦 Total chunks generated: {total_chunks}")

            # 3. 分批处理流水线 (Embedding -> Extraction -> Fusion/Save)
            # 使用 range 和切片进行批处理
            num_batches = math.ceil(total_chunks / pipeline_batch_size)

            for i in range(0, total_chunks, pipeline_batch_size):
                current_batch_idx = i // pipeline_batch_size + 1
                batch_chunks = all_chunks[i: i + pipeline_batch_size]

                print(
                    f"\n🔄 Processing Batch {current_batch_idx}/{num_batches} (Chunks {i + 1}-{min(i + pipeline_batch_size, total_chunks)})...")

                # 3.1 Batch Embedding (Chunks)
                # 即时调用 Embedding
                if embedding_config.get('api_key'):
                    batch_chunks_embedded, tokens = generate_chunk_embeddings(batch_chunks, embedding_config)
                    total_token_usage["embedding_chunks"] += tokens
                    # 过滤掉 embedding 失败的
                    batch_chunks_ready = [c for c in batch_chunks_embedded if c.get('embedding') is not None]
                else:
                    batch_chunks_ready = batch_chunks

                if not batch_chunks_ready:
                    print("⚠️ Batch skipped due to embedding failure.")
                    continue

                # 3.2 Batch Extraction (LLM)
                if llm_config.get('api_key'):
                    print(f"✨ Extracting entities from batch...")
                    batch_entities, batch_relations, tokens = asyncio.run(extract_entities_and_relations(
                        batch_chunks_ready, llm_config, extraction_config
                    ))
                    total_token_usage["extraction"] += tokens

                    # 3.3 Batch Fusion & Save (这是关键，每批处理完立即存库)
                    # fuse_and_update_knowledge_base 内部会处理：
                    # 1. 实体/关系去重
                    # 2. 调用 generate_entity_embeddings / generate_relation_embeddings 补充缺失的向量
                    # 3. 保存到数据库
                    if batch_entities or batch_relations or batch_chunks_ready:
                        doc_chunks_df = pd.DataFrame(batch_chunks_ready)
                        fuse_and_update_knowledge_base(
                            batch_entities,
                            batch_relations,
                            doc_chunks_df,
                            rag_space_path,
                            embedding_config,
                            llm_config,
                            total_token_usage
                        )
                    else:
                        print("ℹ️  Batch yielded no new knowledge.")
                else:
                    # 如果没有 LLM Key，至少保存 Chunk 到数据库
                    print("💾 Saving chunks only (No Extraction)...")
                    doc_chunks_df = pd.DataFrame(batch_chunks_ready)
                    fuse_and_update_knowledge_base(
                        [], [], doc_chunks_df, rag_space_path, embedding_config, total_token_usage
                    )

    # --- 最终报告 ---
    print(f"\n{'=' * 50}\n📊 Total Token Usage Report 📊")
    grand_total = 0
    for key, value in total_token_usage.items():
        print(f"   - {key.replace('_', ' ').title()}: {value:,} tokens")
        grand_total += value
    print(f"   ------------------------------------")
    print(f"   - Grand Total: {grand_total:,} tokens")
    print(f"{'=' * 50}")

    print(f"\n🏁 Processing pipeline finished. 🏁")


if __name__ == "__main__":
    main()