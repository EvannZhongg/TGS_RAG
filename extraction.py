# extraction.py

import asyncio
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
import re
import hashlib
from collections import Counter

from prompt import PROMPTS


# ... _get_unique_id, _parse_llm_output, _process_single_chunk_async 函数保持不变 ...
def _get_unique_id(text: str, prefix: str = "") -> str:
    return f"{prefix}{hashlib.md5(text.encode('utf-8')).hexdigest()}"


def _parse_llm_output(
        raw_text: str,
        chunk_id: str
) -> Tuple[List[Dict], List[Dict]]:
    entities = []
    relations = []

    lines = [
        line.strip() for line in raw_text.split(PROMPTS["DEFAULT_COMPLETION_DELIMITER"])[0].split('\n')
        if line.strip()
    ]

    for line in lines:
        parts = line.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])

        if parts[0].lower() == 'entity' and len(parts) == 4:
            entity_name = parts[1]
            entities.append({
                "entity_id": _get_unique_id(entity_name, prefix="ent-"),
                "entity_name": entity_name,
                "entity_type": parts[2],
                "description": parts[3],
                "source_chunk_id": chunk_id
            })
        elif parts[0].lower() == 'relation' and len(parts) == 5:
            source, target = sorted((parts[1], parts[2]))
            relations.append({
                "relation_id": _get_unique_id(f"{source}-{target}", prefix="rel-"),
                "source": source,
                "target": target,
                "keywords": parts[3],
                "description": parts[4],
                "source_chunk_id": chunk_id
            })
    return entities, relations


async def _process_single_chunk_async(
        chunk: Dict,
        client: AsyncOpenAI,
        llm_config: Dict,
        extraction_config: Dict,
        semaphore: asyncio.Semaphore
) -> Tuple[Dict, int]:
    async with semaphore:
        print(f"   -> Processing chunk {chunk['chunk_id']}...")
        total_tokens_used = 0

        # --- NEW: Get retry configurations ---
        max_retries = llm_config.get('max_retries', 0)
        retry_delay = llm_config.get('retry_delay_seconds', 0)
        # -----------------------------------

        shared_prompt_context = {
            "entity_types": ", ".join(extraction_config.get("entity_types", ["other"])),
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "language": "Chinese"
        }
        system_prompt = PROMPTS["entity_extraction_system_prompt"].format(**shared_prompt_context,
                                                                          input_text=chunk['text'])
        user_prompt = PROMPTS["entity_extraction_user_prompt"].format(**shared_prompt_context)

        history = []
        first_pass_result = ""
        # --- NEW: Retry loop for first pass ---
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=llm_config['model_name'],
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    temperature=0.1
                )
                if response.usage:
                    total_tokens_used += response.usage.total_tokens
                first_pass_result = response.choices[0].message.content
                history.append({"role": "user", "content": user_prompt})
                history.append({"role": "assistant", "content": first_pass_result})
                break  # Success, exit the loop
            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  Retry {attempt + 1}/{max_retries} on first pass for chunk {chunk['chunk_id']}: {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"❌ ERROR on first pass for chunk {chunk['chunk_id']} after {max_retries} retries: {e}")
                    first_pass_result = ""  # Ensure it's empty on final failure
        # -----------------------------------

        entities, relations = _parse_llm_output(first_pass_result, chunk['chunk_id'])

        if extraction_config.get("entity_extract_max_gleaning", 0) > 0 and history:
            glean_user_prompt = PROMPTS["entity_continue_extraction_user_prompt"].format(**shared_prompt_context)
            history.append({"role": "user", "content": glean_user_prompt})

            # --- NEW: Retry loop for second pass ---
            for attempt in range(max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=llm_config['model_name'],
                        messages=[{"role": "system", "content": system_prompt}, *history],
                        temperature=0.1
                    )
                    if response.usage:
                        total_tokens_used += response.usage.total_tokens
                    second_pass_result = response.choices[0].message.content
                    gleaned_entities, gleaned_relations = _parse_llm_output(second_pass_result, chunk['chunk_id'])

                    entities.extend(gleaned_entities)
                    relations.extend(gleaned_relations)
                    break  # Success, exit the loop
                except Exception as e:
                    if attempt < max_retries:
                        print(
                            f"⚠️  Retry {attempt + 1}/{max_retries} on second pass for chunk {chunk['chunk_id']}: {e}")
                        await asyncio.sleep(retry_delay)
                    else:
                        print(f"❌ ERROR on second pass for chunk {chunk['chunk_id']} after {max_retries} retries: {e}")
            # -----------------------------------

        return {"entities": entities, "relations": relations}, total_tokens_used


# <--- MODIFIED FUNCTION ---
def _merge_results_from_chunks(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    对单个文档内所有 chunk 的抽取结果进行合并与去重，并计算出现频率(frequency)。
    """
    all_entities_grouped = {}
    all_relations_grouped = {}
    for result in results:
        for entity in result['entities']:
            name = entity['entity_name']
            if name not in all_entities_grouped:
                all_entities_grouped[name] = []
            all_entities_grouped[name].append(entity)
        for relation in result['relations']:
            key = (relation['source'], relation['target'])
            if key not in all_relations_grouped:
                all_relations_grouped[key] = []
            all_relations_grouped[key].append(relation)

    merged_entities = []
    for name, entities_list in all_entities_grouped.items():
        main_entity = max(entities_list, key=lambda x: len(x['description']))
        types = [e['entity_type'] for e in entities_list]
        main_entity['entity_type'] = Counter(types).most_common(1)[0][0]
        main_entity['source_chunk_ids'] = list(set([e['source_chunk_id'] for e in entities_list]))
        main_entity['frequency'] = len(entities_list)  # <--- MODIFIED: 重命名为 frequency
        main_entity.pop('degree', None)  # 移除可能存在的旧degree字段
        del main_entity['source_chunk_id']
        merged_entities.append(main_entity)

    merged_relations = []
    for key, relations_list in all_relations_grouped.items():
        main_relation = max(relations_list, key=lambda x: len(x['description']))
        main_relation['description'] = " | ".join(set(r['description'] for r in relations_list))
        main_relation['keywords'] = ", ".join(set(r['keywords'] for r in relations_list))
        main_relation['source_chunk_ids'] = list(set(r['source_chunk_id'] for r in relations_list))
        main_relation['frequency'] = len(relations_list)  # <--- MODIFIED: 重命名为 frequency
        main_relation.pop('degree', None)  # 移除可能存在的旧degree字段
        merged_relations.append(main_relation)

    return merged_entities, merged_relations


async def extract_entities_and_relations(
        chunks: List[Dict],
        llm_config: Dict,
        extraction_config: Dict
) -> Tuple[List[Dict], List[Dict], int]:
    """
    从所有文本块中并发地抽取实体和关系，合并结果，并返回总token消耗。
    """
    client = AsyncOpenAI(
        api_key=llm_config['api_key'],
        base_url=llm_config['base_url']
    )

    semaphore = asyncio.Semaphore(llm_config.get('max_async', 4))

    tasks = [
        _process_single_chunk_async(chunk, client, llm_config, extraction_config, semaphore)
        for chunk in chunks
    ]

    results_with_tokens = await asyncio.gather(*tasks)

    chunk_results = [result for result, tokens in results_with_tokens]
    total_tokens_used = sum([tokens for result, tokens in results_with_tokens])

    doc_entities, doc_relations = _merge_results_from_chunks(chunk_results)

    print(
        f"✅ Document extraction complete. Found {len(doc_entities)} unique entities and {len(doc_relations)} unique relations.")

    return doc_entities, doc_relations, total_tokens_used