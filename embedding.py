from typing import List, Dict, Tuple
from openai import OpenAI
import time


# <--- MODIFIED: 返回值增加一个整数用于记录token ---
def _generate_embeddings_for_texts(texts: List[str], config: Dict, item_type: str = "items") -> Tuple[
    List[list[float] | None], int]:
    """
    一个通用的、核心的向量生成函数，接收一个字符串列表。
    现在返回一个元组：(向量列表, 总token消耗)。
    """
    api_key = config.get('api_key')
    base_url = config.get('base_url')
    model_name = config.get('model_name')
    dimensions = config.get('dimensions')
    max_batch_size = config.get('max_batch_size', 25)

    # --- NEW: Get retry configurations ---
    max_retries = config.get('max_retries', 0)
    retry_delay = config.get('retry_delay_seconds', 0)
    # -----------------------------------

    total_tokens_used = 0  # <--- NEW: 初始化token计数器

    if not all([api_key, base_url, model_name, dimensions]):
        print("⚠️  Warning: Embedding config is incomplete. Skipping embedding generation.")
        return [None] * len(texts), total_tokens_used

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}. Skipping embedding generation.")
        return [None] * len(texts), total_tokens_used

    cleaned_texts = []
    original_indices = []
    for i, text in enumerate(texts):
        if text and not text.isspace():
            cleaned_texts.append(text)
            original_indices.append(i)
        else:
            print(f"   - Skipping empty or whitespace-only chunk at original index {i}.")

    if not cleaned_texts:
        print("✅ No valid texts to embed in this batch.")
        return [None] * len(texts), total_tokens_used

    all_embeddings = []
    print(f"🧬 Generating embeddings for {len(cleaned_texts)} valid {item_type} using '{model_name}'...")

    for i in range(0, len(cleaned_texts), max_batch_size):
        batch_texts = cleaned_texts[i:i + max_batch_size]

        # --- NEW: Retry loop for each batch ---
        for attempt in range(max_retries + 1):
            try:
                print(
                    f"   - Processing batch {i // max_batch_size + 1}/{-(-len(cleaned_texts) // max_batch_size)}... (Attempt {attempt + 1})")
                response = client.embeddings.create(
                    model=model_name,
                    input=batch_texts,
                    dimensions=dimensions
                )

                # <--- NEW: 累加从API返回的token消耗 ---
                if response.usage:
                    total_tokens_used += response.usage.total_tokens

                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                time.sleep(0.1)  # Original delay
                break  # Success, break the retry loop

            except Exception as e:
                if attempt < max_retries:
                    print(
                        f"⚠️  Retry {attempt + 1}/{max_retries} failed for batch {i // max_batch_size + 1}: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(
                        f"❌ Error during API call for batch {i // max_batch_size + 1} after {max_retries} retries: {e}. This batch will be skipped.")
                    # On final failure, fill with None and break the retry loop
                    all_embeddings.extend([None] * len(batch_texts))
                    break
        # -----------------------------------

    final_results = [None] * len(texts)
    for i, embedding in enumerate(all_embeddings):
        original_idx = original_indices[i]
        final_results[original_idx] = embedding

    return final_results, total_tokens_used  # <--- MODIFIED: 返回token总数


# --- 修改所有调用处的封装函数以处理新的返回值 ---

def generate_chunk_embeddings(chunks: List[Dict], config: Dict) -> Tuple[List[Dict], int]:
    """为文本块列表生成向量嵌入，并返回token消耗。"""
    texts_to_embed = [chunk['text'] for chunk in chunks]
    embeddings, tokens_used = _generate_embeddings_for_texts(texts_to_embed, config, item_type="chunks")

    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding

    print("✅ Chunk embedding generation complete.")
    return chunks, tokens_used


def generate_entity_embeddings(entities: List[Dict], config: Dict) -> Tuple[List[Dict], int]:
    """为实体列表生成向量嵌入，并返回token消耗。"""
    texts_to_embed = [f"{e['entity_name']}\n{e['description']}" for e in entities]
    embeddings, tokens_used = _generate_embeddings_for_texts(texts_to_embed, config, item_type="entities")

    for entity, embedding in zip(entities, embeddings):
        entity['embedding'] = embedding

    print("✅ Entity embedding generation complete.")
    return entities, tokens_used


def generate_relation_embeddings(relations: List[Dict], config: Dict) -> Tuple[List[Dict], int]:
    """为关系列表生成向量嵌入，并返回token消耗。"""
    texts_to_embed = [
        f"{r['keywords']}\t{r['source_name']}\n{r['target_name']}\n{r['description']}"
        for r in relations
    ]
    embeddings, tokens_used = _generate_embeddings_for_texts(texts_to_embed, config, item_type="relations")

    for relation, embedding in zip(relations, embeddings):
        relation['embedding'] = embedding

    print("✅ Relation embedding generation complete.")
    return relations, tokens_used