"""
文本切分器
==========
递归字符切分：优先按段落/换行/中文句号切，保证语义完整，
再合并到 chunk_size，相邻块保留 overlap。
自实现，不依赖 LangChain，面试可直接讲原理。
"""
from __future__ import annotations

from .loader import Document


_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", " ", ""]


def _split_by_separator(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    parts = text.split(sep)
    # 把分隔符粘回去（除了最后一段）
    return [p + sep for p in parts[:-1]] + [parts[-1]] if parts else []


def _merge_chunks(splits: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把切碎的片段合并成不超过 chunk_size 的块，相邻块保留 overlap。"""
    chunks: list[str] = []
    cur = ""
    for s in splits:
        if len(cur) + len(s) <= chunk_size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            # 开启下一块时，带上上一块结尾的 overlap 文字
            if overlap and cur:
                cur = cur[-overlap:] + s
            else:
                cur = s
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    if not separators:
        # 没分隔符了，硬切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, *rest = separators
    pieces = _split_by_separator(text, sep)
    result: list[str] = []
    for p in pieces:
        if len(p) <= chunk_size:
            result.append(p)
        else:
            result.extend(_recursive_split(p, chunk_size, rest))
    return result


def split_documents(documents: list[Document],
                    chunk_size: int = 500,
                    chunk_overlap: int = 80) -> list[Document]:
    """把长文档切成带 overlap 的小块，保留 metadata 用于溯源。"""
    chunks: list[Document] = []
    for doc in documents:
        pieces = _recursive_split(doc.content, chunk_size, _SEPARATORS)
        merged = _merge_chunks(pieces, chunk_size, chunk_overlap)
        for i, text in enumerate(merged):
            meta = dict(doc.metadata)
            meta["chunk_id"] = i
            chunks.append(Document(content=text, metadata=meta))
    return chunks
