"""
向量存储（纯 NumPy 内存实现）
============================
演示项目数据量小，不需要专业向量数据库。这里直接用 numpy 存向量矩阵，
自己算 cosine 相似度检索。

优点：
  - 没有 sqlite / client 锁 / 线程安全问题，Streamlit 多 rerun 场景稳
  - 代码完全透明，面试可讲"向量归一化 + 矩阵乘法算相似度"
  - 零外部数据库进程

持久化：向量存 .npy，元数据存 .json。
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from config import config
from .embedding import get_embedding
from .loader import Document


_VEC_FILE = "vectors.npy"
_META_FILE = "metadatas.json"


def _doc_id(content: str, source: str, chunk_id: int) -> str:
    raw = f"{source}::{chunk_id}::{content[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class VectorStore:
    def __init__(self) -> None:
        self.embedder = get_embedding()
        self.dir = Path(config.chroma_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self._vecs: np.ndarray | None = None      # (N, dim) 归一化后的向量
        self._metas: list[dict] = []               # 每条 {id, content, metadata}
        self._load()

    def _load(self) -> None:
        vf = self.dir / _VEC_FILE
        mf = self.dir / _META_FILE
        if vf.exists() and mf.exists():
            self._vecs = np.load(vf)
            with open(mf, "r", encoding="utf-8") as f:
                self._metas = json.load(f)

    def _save(self) -> None:
        if self._vecs is not None:
            np.save(self.dir / _VEC_FILE, self._vecs)
        with open(self.dir / _META_FILE, "w", encoding="utf-8") as f:
            json.dump(self._metas, f, ensure_ascii=False)

    def add_documents(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        texts = [c.content for c in chunks]
        embeddings = np.array(self.embedder.embed_documents(texts), dtype=np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        new_items = []
        for i, c in enumerate(chunks):
            new_items.append({
                "id": _doc_id(c.content, c.metadata.get("source", ""),
                              c.metadata.get("chunk_id", i)),
                "content": c.content,
                "metadata": c.metadata,
                "vec": embeddings[i],
            })

        # 以 id 为键：已存在则覆盖，不存在则追加
        old_by_id = {m["id"]: idx for idx, m in enumerate(self._metas)}
        keep_vecs = []
        keep_metas = []
        new_vecs = []
        for idx, m in enumerate(self._metas):
            # 检查是否被新数据覆盖
            hit = next((it for it in new_items if it["id"] == m["id"]), None)
            if hit is not None:
                new_vecs.append(hit["vec"])
                keep_metas.append({"id": hit["id"], "content": hit["content"],
                                   "metadata": hit["metadata"]})
            else:
                keep_vecs.append(self._vecs[idx])
                keep_metas.append(m)

        # 追加全新的条目
        seen_ids = {m["id"] for m in keep_metas}
        for it in new_items:
            if it["id"] not in seen_ids:
                new_vecs.append(it["vec"])
                keep_metas.append({"id": it["id"], "content": it["content"],
                                   "metadata": it["metadata"]})

        self._vecs = np.vstack(new_vecs) if new_vecs else None
        self._metas = keep_metas
        self._save()
        return len(chunks)

    def query(self, query_text: str, top_k: int) -> list[dict]:
        if self._vecs is None or len(self._metas) == 0:
            return []
        q = np.array(self.embedder.embed_query(query_text), dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)

        scores = self._vecs @ q                     # (N,) 已归一化，点积=cosine
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            m = self._metas[idx]
            results.append({
                "content": m["content"],
                "metadata": m["metadata"],
                "score": round(float(1 - scores[idx]), 4),
            })
        return results

    def all_documents(self) -> list[dict]:
        """返回全部 {content, metadata}，供 BM25 建索引用。"""
        return [{"content": m["content"], "metadata": m["metadata"]}
                for m in self._metas]

    def count(self) -> int:
        return len(self._metas)

    def clear(self) -> None:
        self._vecs = None
        self._metas = []
        for f in (self.dir / _VEC_FILE, self.dir / _META_FILE):
            if f.exists():
                f.unlink()
