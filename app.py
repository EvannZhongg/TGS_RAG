# app.py
import streamlit as st
import pandas as pd
import json
import time
import yaml
from sqlalchemy import text

# 引入新的检索器
from retriever_db import PathSBERetriever
from streamlit_agraph import agraph, Node, Edge, Config

# --- 页面配置 ---
st.set_page_config(page_title="可视化路径增强RAG (Path-SBEA)", layout="wide")


# --- 缓存和加载 ---
@st.cache_resource
def load_retriever():
    # 新版初始化不需要加载大量数据，非常快
    return PathSBERetriever()


# 加载配置
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    VIZ_CONFIG = config.get('Visualization', {})
except:
    VIZ_CONFIG = {}

MAX_PATHS_TO_RENDER_DEFAULT = VIZ_CONFIG.get('max_paths_to_render', 100)

retriever = load_retriever()


# --- 新增：数据按需抓取辅助函数 ---
def fetch_realtime_data(retriever_instance, entity_ids=None, chunk_ids=None):
    """
    由于检索器不再将所有数据加载到内存，可视化前需要根据ID去数据库抓取详情。
    """
    local_entity_map = {}
    local_chunk_map = {}

    schema = retriever_instance.db.schema
    engine = retriever_instance.db.get_engine()

    # 1. 抓取实体详情
    if entity_ids:
        # 复用 retriever 内部的抓取逻辑
        local_entity_map, _ = retriever_instance._fetch_local_graph_data(entity_ids)

    # 2. 抓取文本块详情
    if chunk_ids:
        ids_tuple = tuple(chunk_ids)
        if ids_tuple:
            ids_sql = str(ids_tuple)
            if len(ids_tuple) == 1: ids_sql = f"('{ids_tuple[0]}')"

            sql = f"SELECT chunk_id, text, source_document_name FROM {schema}.chunks WHERE chunk_id IN {ids_sql}"
            try:
                with engine.connect() as conn:
                    df = pd.read_sql(sql, conn)
                for _, row in df.iterrows():
                    local_chunk_map[row['chunk_id']] = {
                        'text': row['text'],
                        'source_document_name': row['source_document_name']
                    }
            except Exception as e:
                st.error(f"Error fetching chunks: {e}")

    return local_entity_map, local_chunk_map


# --- 可视化辅助函数 (微调) ---
def build_graph_viz(paths, entity_map, seed_ids, bridged_path_ids, highlight_path_ids=None, key=None):
    """
    构建并渲染图谱。
    修复说明：通过修改 config 对象来实现唯一性，解决 streamlit-agraph 不支持 key 参数的问题。
    """
    nodes, edges = [], []
    added_nodes, added_edges = set(), set()

    bridged_nodes = {eid for p in bridged_path_ids for eid in p}
    highlight_nodes = {eid for p in highlight_path_ids for eid in p} if highlight_path_ids else set()

    for path in paths:
        for i, entity_id in enumerate(path):
            if entity_id not in added_nodes:
                entity_info = entity_map.get(entity_id, {})
                entity_name = entity_info.get('entity_name', entity_id)

                color = "#6495ED"
                size = 15

                if entity_id in seed_ids:
                    color = "#3CB371"
                    size = 20
                elif entity_id in highlight_nodes:
                    color = "#FF4500"
                    size = 20
                elif entity_id in bridged_nodes:
                    color = "#FFA500"

                added_nodes.add(entity_id)
                nodes.append(Node(id=entity_id, label=entity_name, color=color, size=size))

            if len(path) > 1 and i > 0:
                edge = tuple(sorted((path[i - 1], entity_id)))
                if edge not in added_edges:
                    edges.append(Edge(source=edge[0], target=edge[1]))
                    added_edges.add(edge)

    config = Config(width='100%', height=600, directed=False, physics=True, hierarchical=False,
                    solver='forceAtlas2Based',
                    forceAtlas2Based={"gravitationalConstant": -50, "centralGravity": 0.005,
                                      "springLength": 100, "springConstant": 0.18})

    # <--- 核心修复：将 key 注入到 config 中 --->
    # 这样 Streamlit 会检测到 config 对象发生了变化，从而为图表生成唯一的 Element ID
    # 而 agraph 函数本身不需要接收 key 参数，避免了 TypeError
    if key is not None:
        config.__dict__['hack_unique_key'] = key

    # 移除 key=key，只传 config
    return agraph(nodes=nodes, edges=edges, config=config)


# --- 主界面 ---
st.title("💡 可视化路径增强协同检索 (Path-SBEA) - DB版")
st.sidebar.title("🔍 控制面板")

# Sidebar controls
query = st.sidebar.text_area("1. 请输入您的问题:", value="RoHS指令和峰值正向电流有什么关系?", height=100)
top_k_chunks = st.sidebar.slider("2. Top K Chunks", 1, 10, 5)
top_k_paths = st.sidebar.slider("3. Top K Paths", 1, 10, 5)
max_paths_to_render = st.sidebar.slider("5. 可视化最大路径数", 10, 500, MAX_PATHS_TO_RENDER_DEFAULT)
answering_mode = st.sidebar.selectbox(
    "4. 最终答案生成模式",
    options=["full_context", "chunks_only", "paths_only"],
    index=0,
    format_func=lambda x: {"full_context": "完整上下文", "chunks_only": "仅文本", "paths_only": "仅图谱"}[x],
    help="选择将哪种类型的上下文信息提交给LLM以生成最终答案。"
)

if st.sidebar.button("🚀 执行检索与回答"):
    if not query:
        st.warning("请输入问题！")
    else:
        for key in ['results', 'diagnostics', 'final_answer_stream', 'viz_data_cache']:
            if key in st.session_state:
                del st.session_state[key]

        with st.spinner("执行中... (1/2) 检索相关信息..."):
            # 执行搜索
            results, diagnostics = retriever.search(query, top_k_chunks, top_k_paths)
            st.session_state.results = results
            st.session_state.diagnostics = diagnostics

            # --- 预加载可视化数据 ---
            # 收集所有需要展示的 Entity ID 和 Chunk ID
            all_viz_entity_ids = set()

            # 1. 种子实体
            seed_entities = results.get('seed_entities', [])
            all_viz_entity_ids.update([s['id'] for s in seed_entities])

            # 2. 所有路径上的节点 (包括被桥接的)
            all_paths = results.get('all_paths', [])
            for p in all_paths:
                all_viz_entity_ids.update(p['path'])

            # 3. 候选 Chunk IDs
            candidate_chunks = results.get('candidate_chunks', [])
            all_viz_chunk_ids = {c['id'] for c in candidate_chunks}

            # 批量抓取详情并缓存
            ent_map, chunk_map = fetch_realtime_data(retriever, all_viz_entity_ids, all_viz_chunk_ids)
            st.session_state.viz_data_cache = {
                'entity_map': ent_map,
                'chunk_map': chunk_map
            }

        with st.spinner("执行中... (2/2) 生成最终答案..."):
            answer_gen_start_time = time.time()
            answer_stream = retriever.generate_answer(
                query,
                st.session_state.results['top_chunks'],
                st.session_state.results['top_paths'],
                answering_mode
            )
            st.session_state.final_answer_stream = answer_stream
            answer_gen_time = time.time() - answer_gen_start_time
            st.session_state.diagnostics['time_answer_generation'] = f"{answer_gen_time:.2f}s"

# --- 侧边栏监控信息展示 ---
if 'diagnostics' in st.session_state:
    with st.sidebar.expander("🤖 LLM 初始实体抽取", expanded=True):
        extracted = st.session_state.diagnostics.get('llm_extraction', {})
        st.json(extracted.get('entities', ["无"]))
        usage = extracted.get('usage')
        if usage:
            st.write(f"**Tokens:** {usage.total_tokens}")

# --- 主页面结果展示 ---
if 'results' in st.session_state and 'viz_data_cache' in st.session_state:
    results = st.session_state.results
    viz_cache = st.session_state.viz_data_cache
    entity_map_cache = viz_cache['entity_map']
    chunk_map_cache = viz_cache['chunk_map']

    # 最终答案区
    if 'final_answer_stream' in st.session_state:
        st.markdown("---")
        st.subheader("🤖 最终生成答案")
        st.write_stream(st.session_state.final_answer_stream)

    st.markdown("---")
    st.header("🔍 检索过程详解")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 最终检索结果", "🕸️ 图谱可视化演进", "📚 文本块筛选过程", "⏱️ 性能监控"])

    # Tab 1: 最终结果
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏆 Top K 推理路径")
            for i, path in enumerate(results['top_paths']):
                with st.expander(f"路径 #{i + 1} | Score: {path['score']:.3f}", expanded=False):
                    st.success(f"`{path['path_readable']}`")
                    st.caption(f"生成原因: {path['reason']}")
                    # 这里的 segments 已经在 retriever.get_path_details 中填充好了，无需额外处理
                    for segment in path.get('segments', []):
                        st.markdown(f"**- {segment['source']}**")
                        st.caption(f"  {segment['source_desc']}")
                        st.markdown(f"  └ `关系`: {segment['keywords']} - *{segment['description']}*")
                        st.markdown(f"**- {segment['target']}**")
                        st.caption(f"  {segment['target_desc']}")
                    if path.get('endorsing_bridges'):
                        st.markdown("---")
                        st.markdown("**由以下证据补全支持:**")
                        for bridge_readable in path['endorsing_bridges']:
                            st.info(f"`{bridge_readable}`")
        with col2:
            st.subheader("📚 Top K 文本证据")
            for i, chunk in enumerate(results['top_chunks']):
                with st.expander(f"文本 #{i + 1} | Score: {chunk['score']:.3f}", expanded=False):
                    st.info(f"**{chunk['name']}**")
                    st.caption(chunk['content'])
                    st.caption(f"评分构成: {chunk['reason']}")

    # Tab 2: 图谱可视化
    with tab2:
        st.subheader("图谱构建与筛选的三个阶段")
        st.markdown("""
        - **<font color='#3CB371'>绿色节点</font>**: 起点实体 (LLM抽取+向量检索)。
        - **<font color='#6495ED'>蓝色节点</font>**: 通过BFS从起点扩展的实体。
        - **<font color='#FFA500'>橙色节点</font>**: 通过“桥接”文本证据中的孤立实体而补全的路径节点。
        - **<font color='#FF4500'>红色节点</font>**: 最终被选入Top-K推理路径的核心实体。
        """, unsafe_allow_html=True)

        seed_entities = results.get('seed_entities', [])
        all_paths = results.get('all_paths', [])  # 这里的 item 只有 path (id list), score 等
        bridged_paths = results.get('bridged_paths', [])
        top_paths_info = results.get('top_paths', [])

        seed_ids = {s['id'] for s in seed_entities}
        bridged_path_ids = [p['path'] for p in bridged_paths]
        top_paths_ids = [p['entity_ids'] for p in top_paths_info]  # get_path_details 返回了 entity_ids

        with st.expander("1. 起点实体图谱", expanded=True):
            if seed_ids:
                build_graph_viz(
                    paths=[[sid] for sid in seed_ids],
                    entity_map=entity_map_cache,
                    seed_ids=seed_ids,
                    bridged_path_ids=[],
                    key="viz_seed_graph"  # <--- 新增唯一 Key
                )
            else:
                st.info("未能找到起点实体。")

        with st.expander("2. 扩展与桥接全图", expanded=False):
            if all_paths:
                bridged_path_ids_set = {tuple(p.get('path', [])) for p in bridged_paths}
                paths_to_render = [p['path'] for p in all_paths if tuple(p['path']) in bridged_path_ids_set]

                # 补充非桥接路径直到上限
                remaining_slots = max_paths_to_render - len(paths_to_render)
                if remaining_slots > 0:
                    non_bridged = [p['path'] for p in all_paths if tuple(p['path']) not in bridged_path_ids_set]
                    paths_to_render.extend(non_bridged[:remaining_slots])

                if len(all_paths) > max_paths_to_render:
                    st.info(f"为保持流畅，仅显示 {len(paths_to_render)}/{len(all_paths)} 条路径。")

                build_graph_viz(
                    paths=paths_to_render,
                    entity_map=entity_map_cache,
                    seed_ids=seed_ids,
                    bridged_path_ids=bridged_path_ids,
                    key="viz_full_graph"  # <--- 新增唯一 Key
                )
            else:
                st.info("未能通过BFS扩展图谱。")

        with st.expander("3. 最终选定路径图", expanded=False):
            if top_paths_ids:
                build_graph_viz(
                    paths=top_paths_ids,
                    entity_map=entity_map_cache,
                    seed_ids=seed_ids,
                    bridged_path_ids=bridged_path_ids,
                    highlight_path_ids=top_paths_ids,
                    key="viz_final_graph"  # <--- 新增唯一 Key
                )
            else:
                st.info("最终未能筛选出任何路径。")

    # Tab 3: 文本块过程
    with tab3:
        st.subheader("文本块从初始检索到最终排序的全过程")

        # 1. 获取数据
        initial_chunks = results.get('initial_chunks', [])  # Retriever 已经填充好了内容
        candidate_chunks = results.get('candidate_chunks', [])

        # 2. 准备候选池数据 (需要用 cache 填充内容)
        display_candidates = []
        for c in candidate_chunks:
            cid = c['id']
            details = chunk_map_cache.get(cid, {})
            display_candidates.append({
                **c,
                'name': f"Chunk from {details.get('source_document_name', 'Unknown')}",
                'content': details.get('text', '内容加载失败'),
            })

        # <--- 【新增】展示初始检索 (Top-K 向量检索结果) --->
        with st.expander(f"1. 初始检索文本块 (Vector Search Top-2K: {len(initial_chunks)}个)", expanded=False):
            # 注意：initial_chunks 已经在 retriever 中通过 get_item_details 填充了 content
            for chunk in initial_chunks:
                st.info(
                    f"**{chunk['name']}**\n\n"
                    f"{chunk['content'][:200]}..."  # 预览
                )

        # <--- 【修改】序号顺延 --->
        with st.expander(f"2. 候选池文本块 (Initial + Graph Recs: {len(display_candidates)}个)", expanded=False):
            sorted_candidates = sorted(display_candidates, key=lambda x: x['final_score'], reverse=True)
            for chunk in sorted_candidates:
                st.warning(
                    f"**{chunk['name']}** (Final Score: {chunk['final_score']:.3f})\n\n"
                    f"*评分构成: {chunk['reason']}*\n\n"
                    f"{chunk['content'][:200]}..."
                )

        with st.expander(f"3. 最终选定文本块 (Top {len(results['top_chunks'])} Chunks)", expanded=False):
            for chunk in results['top_chunks']:
                st.success(
                    f"**{chunk['name']}** (Final Score: {chunk['score']:.3f})\n\n{chunk['content']}"
                )

    # Tab 4: 性能
    with tab4:
        st.subheader("⏱️ 各阶段性能指标")
        diagnostics = st.session_state.diagnostics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 耗时")
            st.metric(label="总检索耗时", value=diagnostics.get('time_total_retrieval', 'N/A'))
            st.text(f"  - 阶段1 (初始检索): {diagnostics.get('time_stage1_retrieval', 'N/A')}")
            st.text(f"  - 阶段2 (融合评分): {diagnostics.get('time_stage2_fusion', 'N/A')}")
            st.text(f"  - 阶段3 (排序): {diagnostics.get('time_stage3_ranking', 'N/A')}")
            st.metric(label="最终答案生成耗时", value=diagnostics.get('time_answer_generation', 'N/A'))
        with col2:
            st.markdown("#### Token 消耗")
            usage = diagnostics.get('llm_extraction', {}).get('usage')
            if usage:
                st.metric(label="实体抽取总 Tokens", value=usage.total_tokens)