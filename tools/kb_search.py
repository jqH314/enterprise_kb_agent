"""
知识库检索工具
==============
Agent 判断需要查企业知识库时调用。
"""
from __future__ import annotations

from core.tool import tool
from tools import get_rag


@tool
def search_knowledge_base(query: str) -> str:
    """在企业知识库中检索与问题相关的文档片段。
    当用户问题涉及公司制度、产品资料、业务规则等内部资料时，必须先用此工具检索，
    不要凭记忆编造答案。

    :param query: 要检索的问题或关键词，用中文完整表述
    """
    rag = get_rag()
    if rag.count() == 0:
        return "知识库为空，请先运行索引构建（python main.py --build）或在网页上传文档。"
    return rag.search(query)
