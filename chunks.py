import tiktoken
import re
from typing import List, Dict

# 加载编码器
ENCODING = tiktoken.get_encoding("cl100k_base")


def _get_token_count(text: str) -> int:
    """辅助函数：计算文本的 token 数量"""
    return len(ENCODING.encode(text))


def _preprocess_text(text: str) -> str:
    """
    文本预处理器：在分块前清洗掉 PDF 转 Markdown 插入的图片占位符和 Markdown 图片链接。
    """
    # 1. 移除 PDF 转 Markdown 生成的图片占位符，例如 <!-- image -->
    cleaned_text = re.sub(r'<!--\s*image\s*-->', '', text)

    # 2. 移除所有 Markdown 图片链接，例如 ![alt](url)
    cleaned_text = re.sub(r'\n?!\[.*?\]\(.*?\)\n?', '', cleaned_text)

    return cleaned_text.strip()


def chunk_text_fixed_size(
        text: str,
        file_hash: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
) -> List[Dict]:
    """固定大小的分块策略 (已修复乱码问题)"""
    if not text:
        return []

    # 编码为 token list
    tokens = ENCODING.encode(text)
    chunks = []
    step = chunk_size - chunk_overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i: i + chunk_size]

        # 1. 解码 (使用 errors='replace' 会生成 ，这是默认行为)
        chunk_text = ENCODING.decode(chunk_tokens, errors='replace')

        # 2. 【核心修复】去除首尾的替换字符  (U+FFFD)
        # 这样可以移除因 token 切割导致的无效字符片段
        chunk_text = chunk_text.strip('\ufffd')

        # 3. 额外的清理 (可选)：去除因为 strip 留下的首尾空白，视需求而定
        # chunk_text = chunk_text.strip()

        if not chunk_text:
            continue

        chunk_id = f"{file_hash}_{len(chunks)}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_tokens)
        })

    print(f"✅ 'fixed' 策略分块完成，共计 {len(chunks)} 个块。")
    return chunks


def chunk_text_semantic(
        text: str,
        file_hash: str,
        target_size: int = 1000,
        overlap_target: int = 150,
        pre_context_limit: int = 50,
        hard_limit: int = 500
) -> List[Dict]:
    """
    高级语义分块策略 V6：最终版，集成了“贪心且不超标”的常规重叠逻辑。
    """
    if not text:
        return []

    # 1. 结构化拆分 (逻辑不变)
    separators_regex = r'(\n##+\s.*)'
    blocks = re.split(separators_regex, text)
    structured_blocks = []
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        if not block: i += 1; continue
        if re.match(separators_regex, block) and i + 1 < len(blocks):
            structured_blocks.append(f"{block}\n\n{blocks[i + 1].strip()}")
            i += 2
        else:
            structured_blocks.append(block)
            i += 1
    if not structured_blocks:
        return [{"chunk_id": f"{file_hash}_0", "text": text, "token_count": _get_token_count(text)}] if text else []

    # 2. 构建基础块 (逻辑不变)
    base_chunks_of_blocks = []
    current_base_chunk = []
    current_tokens = 0
    for block in structured_blocks:
        block_tokens = _get_token_count(block)
        if block_tokens > target_size:
            if current_base_chunk:
                base_chunks_of_blocks.append(current_base_chunk)
            base_chunks_of_blocks.append([block])
            current_base_chunk = []
            current_tokens = 0
            continue
        if current_tokens + block_tokens > target_size and current_base_chunk:
            base_chunks_of_blocks.append(current_base_chunk)
            current_base_chunk = [block]
            current_tokens = block_tokens
        else:
            current_base_chunk.append(block)
            current_tokens += block_tokens
    if current_base_chunk:
        base_chunks_of_blocks.append(current_base_chunk)

    # 3. 为基础块构建最终分块，并应用最终版的重叠逻辑
    final_chunks = []
    for i, current_blocks in enumerate(base_chunks_of_blocks):
        final_chunk_content_blocks = list(current_blocks)

        if i > 0:
            previous_blocks = base_chunks_of_blocks[i - 1]
            last_unit_of_prev = previous_blocks[-1]
            last_unit_tokens = _get_token_count(last_unit_of_prev)

            overlap_blocks_to_prepend = []

            # 规则 1: 硬性上限
            if last_unit_tokens > hard_limit:
                pass
                # 规则 2: 大单元特殊规则
            elif last_unit_tokens > overlap_target:
                overlap_blocks_to_prepend.append(last_unit_of_prev)
                if len(previous_blocks) > 1:
                    pre_context_unit = previous_blocks[-2]
                    if _get_token_count(pre_context_unit) <= pre_context_limit:
                        overlap_blocks_to_prepend.insert(0, pre_context_unit)
            # --- 核心改动：新的“贪心且不超标”常规重叠逻辑 ---
            else:
                current_overlap_tokens = 0
                for prev_block in reversed(previous_blocks):
                    prev_block_tokens = _get_token_count(prev_block)
                    # 检查下一个块加进来是否会“超标”
                    if current_overlap_tokens + prev_block_tokens > overlap_target:
                        break  # 如果会超标，则立即停止，不添加这个块

                    overlap_blocks_to_prepend.insert(0, prev_block)
                    current_overlap_tokens += prev_block_tokens

            final_chunk_content_blocks = overlap_blocks_to_prepend + final_chunk_content_blocks

        final_text = "\n\n".join(final_chunk_content_blocks)
        chunk_id = f"{file_hash}_{len(final_chunks)}"
        final_chunks.append({"chunk_id": chunk_id, "text": final_text, "token_count": _get_token_count(final_text)})

    print(f"✅ 'semantic' 策略分块完成，共计 {len(final_chunks)} 个块。")
    return final_chunks


def chunk_dispatcher(text: str, file_hash: str, config: Dict) -> List[Dict]:
    """分块调度器 (现在会传递所有语义配置参数)"""
    print("🧹 Preprocessing text to remove image placeholders and links...")
    cleaned_text = _preprocess_text(text)

    strategy = config.get('strategy', 'fixed')

    if strategy == 'semantic':
        return chunk_text_semantic(
            cleaned_text,
            file_hash,
            target_size=config.get('semantic_target', 1000),
            overlap_target=config.get('semantic_overlap', 150),
            pre_context_limit=config.get('semantic_pre_context_limit', 50),
            hard_limit=config.get('semantic_hard_limit', 500)
        )
    elif strategy == 'fixed':
        return chunk_text_fixed_size(
            cleaned_text, file_hash,
            chunk_size=config.get('fixed_size', 1000),
            chunk_overlap=config.get('fixed_overlap', 150)
        )
    else:
        print(f"⚠️  Warning: Unknown chunking strategy '{strategy}'. Falling back to 'fixed'.")
        return chunk_text_fixed_size(cleaned_text, file_hash)