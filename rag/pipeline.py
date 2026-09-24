"""
RAG 流水线封装
==============
对外暴露：
  - build_index(): 从 docs_dir 加载文档 → 切分 → 向量化入库
  - search(): 检索并格式化成带引用的文本（给 Agent 工具用）
"""
from __future__ import annotations

from config import config
from .loader import load_dir, load_file
from .splitter import split_documents
from .vectorstore import VectorStore
from .retriever import Retriever


class RAGPipeline:
    def __init__(self) -> None:
        self.vs = VectorStore()
        self.retriever = Retriever(self.vs)

    def build_index(self, dir_or_file: str | None = None,
                    chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> int:
        """建索引。返回入库 chunk 数。"""
        chunk_size = chunk_size or config.chunk_size
        chunk_overlap = chunk_overlap or config.chunk_overlap

        target = dir_or_file or config.docs_dir
        import os
        if os.path.isdir(target):
            docs = load_dir(target)
        else:
            docs = load_file(target)

        chunks = split_documents(docs, chunk_size=chunk_size,
                                 chunk_overlap=chunk_overlap)
        n = self.vs.add_documents(chunks)
        # 建库后清掉 BM25 缓存，强制重建
        self.retriever._build_bm25() if self.retriever._bm25 is None else None
        return n

    def search(self, query: str, top_k: int | None = None) -> str:
        """检索并格式化成带 [来源] 引用的文本。"""
        hits = self.retriever.retrieve(query, rerank_k=top_k)
        if not hits:
            return "知识库中未检索到相关内容。"

        lines = []
        for i, h in enumerate(hits, 1):
            src = h["metadata"].get("source", "未知")
            page = h["metadata"].get("page")
            tag = f"{src}#p{page}" if page else src
            lines.append(f"[片段{i} | 来源: {tag}]\n{h['content']}")
        return "\n\n".join(lines)

    def count(self) -> int:
        return self.vs.count()
