"""
工具服务单例
============
把 RAG 管线、沙箱这些重对象做成懒加载单例，供 @tool 函数调用，
避免每次工具调用都重新加载模型。
"""
from __future__ import annotations

from functools import lru_cache

from config import config
from core.sandbox import Sandbox


@lru_cache(maxsize=1)
def get_rag():
    from rag.pipeline import RAGPipeline
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_sandbox() -> Sandbox:
    return Sandbox(workspace=config.workspace_dir, timeout=config.sandbox_timeout)


def register_all_tools() -> None:
    """导入所有工具模块，触发 @tool 注册。"""
    from tools import kb_search, code_exec, file_tools  # noqa: F401
