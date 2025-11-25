# TGSRAG.py

import sys
from pathlib import Path
import hashlib
import json
import yaml
import pandas as pd
import asyncio
import math
import traceback

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

    # <--- 新增：失败文档记录列表 --->
    failed_documents = []

    input_dir = Path("F:/Code/TSG_RAG/hotpot_context")  # 请根据你的实际路径修改
    output_dir_base = Path("F:/Code/TSG_RAG/hotpot_temp")
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

    pipeline_batch_size = embedding_config.get('max_batch_size', 10)
    print(f"⚙️ Pipeline Batch Size set to: {pipeline_batch_size}")

    # 初始化 DBManager 检查已处理文档
    try:
        db_manager = DBManager()
        existing_chunks_df = db_manager.load_df('chunks')
        processed_docs = set(
            existing_chunks_df['source_document_name'].unique()) if not existing_chunks_df.empty else set()
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    input_dir.mkdir(exist_ok=True)
    output_dir_base.mkdir(exist_ok=True)

    for doc_path in input_dir.iterdir():
        if not doc_path.is_file():
            continue

        print(f"\n{'=' * 50}\n-> Found document: {doc_path.name}")

        if doc_path.name in processed_docs:
            print(f"✅ Document '{doc_path.name}' has been processed before (found in DB). Skipping.")
            continue

        # <--- 修改：增加 try-except 捕获单个文档处理的异常 --->
        try:
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

            if not text_to_chunk:
                print(f"⚠️  Warning: Document '{doc_path.name}' extracted empty text.")
                failed_documents.append({"name": doc_path.name, "reason": "Empty text extracted"})
                continue

            # 2. 分块 (Chunking)
            print("🧠 Chunking text...")
            all_chunks = chunk_dispatcher(text_to_chunk, doc_hash_name, chunking_config)
            source_filename = doc_path.name
            for chunk in all_chunks:
                chunk['source_document_name'] = source_filename

            total_chunks = len(all_chunks)
            print(f"📦 Total chunks generated: {total_chunks}")

            if total_chunks == 0:
                print(f"⚠️  Warning: No chunks generated for '{doc_path.name}'.")
                failed_documents.append({"name": doc_path.name, "reason": "No chunks generated"})
                continue

            # 3. 分批处理流水线
            num_batches = math.ceil(total_chunks / pipeline_batch_size)
            doc_failed_batches = 0  # 记录该文档失败的批次数

            for i in range(0, total_chunks, pipeline_batch_size):
                current_batch_idx = i // pipeline_batch_size + 1
                batch_chunks = all_chunks[i: i + pipeline_batch_size]

                print(
                    f"\n🔄 Processing Batch {current_batch_idx}/{num_batches} (Chunks {i + 1}-{min(i + pipeline_batch_size, total_chunks)})...")

                # 3.1 Batch Embedding
                if embedding_config.get('api_key'):
                    batch_chunks_embedded, tokens = generate_chunk_embeddings(batch_chunks, embedding_config)
                    total_token_usage["embedding_chunks"] += tokens
                    batch_chunks_ready = [c for c in batch_chunks_embedded if c.get('embedding') is not None]
                else:
                    batch_chunks_ready = batch_chunks

                if not batch_chunks_ready:
                    print("⚠️ Batch skipped due to embedding failure.")
                    doc_failed_batches += 1
                    continue

                # 3.2 Batch Extraction
                if llm_config.get('api_key'):
                    print(f"✨ Extracting entities from batch...")
                    # 在这里也可以加 try-except 如果你想让批次失败不影响文档后续批次
                    try:
                        batch_entities, batch_relations, tokens = asyncio.run(extract_entities_and_relations(
                            batch_chunks_ready, llm_config, extraction_config
                        ))
                        total_token_usage["extraction"] += tokens

                        # 3.3 Batch Fusion & Save
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
                    except Exception as batch_e:
                        print(f"❌ Batch {current_batch_idx} failed: {batch_e}")
                        traceback.print_exc()
                        doc_failed_batches += 1
                else:
                    print("💾 Saving chunks only (No Extraction)...")
                    doc_chunks_df = pd.DataFrame(batch_chunks_ready)
                    fuse_and_update_knowledge_base(
                        [], [], doc_chunks_df, rag_space_path, embedding_config, llm_config, total_token_usage
                    )

            # 如果有批次失败，记录为部分失败
            if doc_failed_batches > 0:
                failed_documents.append({
                    "name": doc_path.name,
                    "reason": f"Partial failure: {doc_failed_batches}/{num_batches} batches failed"
                })

        except Exception as e:
            print(f"❌ Error processing document '{doc_path.name}': {e}")
            traceback.print_exc()  # 打印堆栈以便调试
            failed_documents.append({"name": doc_path.name, "reason": str(e)})
            continue  # 继续处理下一个文档

    # --- 最终报告 ---
    print(f"\n{'=' * 50}\n📊 Total Token Usage Report 📊")
    grand_total = 0
    for key, value in total_token_usage.items():
        print(f"   - {key.replace('_', ' ').title()}: {value:,} tokens")
        grand_total += value
    print(f"   ------------------------------------")
    print(f"   - Grand Total: {grand_total:,} tokens")
    print(f"{'=' * 50}")

    # <--- 新增：打印失败文档报告 --->
    if failed_documents:
        print(f"\n{'!' * 50}")
        print(f"⚠️  WARNING: {len(failed_documents)} documents encountered errors during processing.")
        print(f"{'!' * 50}")
        print("Failed Documents List:")
        for idx, fail in enumerate(failed_documents, 1):
            print(f"  {idx}. {fail['name']}")
            print(f"     Reason: {fail['reason']}")
        print(f"{'=' * 50}")
    else:
        print("\n✅ All documents processed successfully without errors.")

    print(f"\n🏁 Processing pipeline finished. 🏁")


if __name__ == "__main__":
    main()