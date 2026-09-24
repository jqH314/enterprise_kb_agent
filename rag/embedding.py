"""
Embedding 模型
==============
两种 provider，配置里切换：
  - local : 本地跑 BGE（数据不出域，有 GPU 自动用 GPU，简历卖点）
  - openai: 调用兼容 OpenAI 的 embedding 接口

懒加载：首次调用才真正加载模型/建立连接。
"""
from __future__ import annotations

from functools import lru_cache

from config import config


# BGE 检索场景建议给 query 加的指令前缀（官方推荐）
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class LocalEmbedding:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedding] 加载本地模型 {model_name} (device={device}) ...")
        self.model = SentenceTransformer(model_name, device=device)
        self.is_bge = "bge" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE 查询加指令前缀效果更好
        q = _BGE_QUERY_INSTRUCTION + text if self.is_bge else text
        vec = self.model.encode([q], normalize_embeddings=True)
        return vec[0].tolist()


class OpenAIEmbedding:
    def __init__(self) -> None:
        from openai import OpenAI
        self.model = config.openai_embedding_model
        self.client = OpenAI(
            base_url=config.embedding_base_url or None,
            api_key=config.embedding_api_key or config.llm_api_key,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding():
    if config.embedding_provider == "local":
        return LocalEmbedding(config.local_embedding_model)
    return OpenAIEmbedding()
