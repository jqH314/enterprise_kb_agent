"""
Streamlit Web 界面
==================
启动：streamlit run app.py

功能：
  - 侧边栏上传文档 / 一键建索引 / 查看库内状态
  - 主对话区展示答案 + Agent 思考轨迹（调了哪些工具、拿到什么）
"""
from __future__ import annotations

import os
import shutil

import streamlit as st

from config import config
from tools import get_rag

st.set_page_config(page_title="企业知识库问答 Agent", page_icon="📚", layout="wide")


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def build_index_ui() -> None:
    st.sidebar.header("📥 知识库管理")

    uploaded = st.sidebar.file_uploader(
        "上传文档",
        type=["pdf", "docx", "txt", "md", "xlsx", "csv"],
        accept_multiple_files=True,
    )
    if uploaded and st.sidebar.button("保存并入库"):
        os.makedirs(config.docs_dir, exist_ok=True)
        for f in uploaded:
            with open(os.path.join(config.docs_dir, f.name), "wb") as out:
                out.write(f.getbuffer())
        with st.sidebar.status("正在建索引（首次会下载模型）..."):
            try:
                rag = get_rag()
                n = rag.build_index()
                st.sidebar.success(f"入库完成，新增 {n} 块，库内共 {rag.count()} 块")
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"建索引失败: {e}")

    if st.sidebar.button("重建索引（扫描 data/documents）"):
        with st.sidebar.status("重建索引中..."):
            try:
                rag = get_rag()
                rag.vs.clear()
                n = rag.build_index()
                st.sidebar.success(f"重建完成，共 {n} 块")
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"重建失败: {e}")

    try:
        rag = get_rag()
        st.sidebar.metric("库内文本块数", rag.count())
    except Exception:  # noqa: BLE001
        st.sidebar.info("尚未加载知识库")


def chat_ui() -> None:
    st.title("📚 企业知识库问答 Agent")
    st.caption("RAG + Agent 双范式 · 混合检索 / Rerank 精排 / 工具调用 / 沙箱执行")

    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("请输入你的问题，例如：公司的差旅报销标准是什么？"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not config.llm_api_key:
                st.error("未配置 LLM_API_KEY，请在 .env 中填写。")
                return
            from core.agent import KBAgent
            agent = KBAgent()
            result = agent.run(question)

            # 折叠展示思考轨迹
            with st.expander("🔧 Agent 思考轨迹"):
                for s in result["steps"]:
                    if s["type"] == "tool_call":
                        st.markdown(f"**🛠 调用工具 `{s['tool']}`**")
                        st.code(str(s["args"]), language="json")
                        st.markdown("返回：")
                        st.text(s["observation"][:1500])
                    elif s["type"] in ("final", "forced_final") and s.get("content"):
                        st.markdown("**✅ 生成最终答案**")

            st.markdown(result["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"]}
            )


def main() -> None:
    config.ensure_dirs()
    init_state()
    build_index_ui()
    chat_ui()


if __name__ == "__main__":
    main()
