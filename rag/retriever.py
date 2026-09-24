"""
混合检索 + Rerank 重排序
========================
这是项目的核心检索亮点：
  1. 向量检索（语义相似）—— 来自 Chroma
  2. BM25 关键词检索（精确匹配）—— 本地建索引
  3. RRF（Reciprocal Rank Fusion）融合两路结果
  4. bge-reranker 精排，取最终 top_k

向量擅长"意思相近"，BM25 擅长"专有名词/编号/关键词精确命中"，
融合 + 精排后召回率和准确率都比单路向量检索高。
"""
from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from config import config
from .vectorstore import VectorStore


def _tokenize(text: str) -> list[str]:
    """简单中文分词：中文字单字 + 英文/数字整词。够用且零依赖。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fa5]", text)
    return tokens


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Rerank] 加载 {config.rerank_model} (device={device}) ...")
    return CrossEncoder(config.rerank_model, device=device)


class Retriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store
        self._bm25 = None
        self._corpus: list[str] = []      # 与 bm25 索引对齐的原文
        self._corpus_meta: list[dict] = []

    def _build_bm25(self) -> None:
        """把库里所有 chunk 捞出来建 BM25 索引（懒加载一次）。"""
        all_data = self.vs.all_documents()
        self._corpus = [d["content"] for d in all_data]
        self._corpus_meta = [d["metadata"] for d in all_data]
        tokenized = [_tokenize(doc) for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            self._build_bm25()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            if scores[idx] > 0:
                results.append({
                    "content": self._corpus[idx],
                    "metadata": self._corpus_meta[idx],
                    "bm25_score": float(scores[idx]),
                })
        return results

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        return self.vs.query(query, top_k)

    @staticmethod
    def _rrf_fuse(vec_results: list[dict], bm25_results: list[dict],
                  k: int = 60) -> list[dict]:
        """RRF 融合：按排名打分，同一文档出现在两路会被加权。"""
        scores: dict[str, float] = {}
        merged: dict[str, dict] = {}

        for rank, r in enumerate(vec_results):
            key = r["content"][:128]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            merged.setdefault(key, r)

        for rank, r in enumerate(bm25_results):
            key = r["content"][:128]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            merged.setdefault(key, r)

        fused = sorted(merged.values(), key=lambda r: -scores[r["content"][:128]])
        return fused

    def retrieve(self, query: str,
                 retrieve_k: int | None = None,
                 rerank_k: int | None = None) -> list[dict]:
        """完整检索链路：向量 + BM25 → RRF 融合 → Rerank 精排。"""
        retrieve_k = retrieve_k or config.retrieve_top_k
        rerank_k = rerank_k or config.rerank_top_k

        vec_hits = self._vector_search(query, retrieve_k)
        bm25_hits = self._bm25_search(query, retrieve_k)
        candidates = self._rrf_fuse(vec_hits, bm25_hits)

        if not candidates:
            return []

        if config.use_rerank:
            try:
                reranker = _get_reranker()
                pairs = [[query, c["content"]] for c in candidates]
                rerank_scores = reranker.predict(pairs)
                for c, s in zip(candidates, rerank_scores):
                    c["rerank_score"] = float(s)
                candidates.sort(key=lambda c: -c["rerank_score"])
                candidates = candidates[:rerank_k]
            except Exception as e:  # noqa: BLE001
                print(f"[警告] Rerank 失败，退回融合结果: {e}")
                candidates = candidates[:rerank_k]
        else:
            candidates = candidates[:rerank_k]

        return candidates
