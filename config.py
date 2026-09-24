"""
全局配置
========
所有配置项都从环境变量 / .env 读取，代码里不写死任何密钥。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---- HuggingFace 镜像设置（必须在 import 模型库之前生效）----
# 国内访问走镜像；Streamlit Cloud 等海外环境也能正常访问。
# 不强制离线：本地有缓存时自动用缓存，没有时自动从镜像下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # ---------- LLM ----------
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")

    # ---------- Embedding ----------
    # provider: local（本地 BGE，数据不出域，推荐）/ openai（兼容 OpenAI 的 embedding 接口）
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    local_embedding_model: str = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")

    # ---------- Rerank ----------
    use_rerank: bool = _get_bool("USE_RERANK", True)
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

    # ---------- 向量库 ----------
    chroma_dir: str = str(BASE_DIR / "data" / "chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "enterprise_kb")

    # ---------- RAG 检索参数 ----------
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    retrieve_top_k: int = int(os.getenv("RETRIEVE_TOP_K", "8"))      # 混合检索召回数
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))          # 精排后保留数

    # ---------- Agent ----------
    max_steps: int = int(os.getenv("MAX_STEPS", "6"))                # 最多工具调用轮次
    system_prompt: str = os.getenv("SYSTEM_PROMPT", "")

    # ---------- 路径 ----------
    docs_dir: str = str(BASE_DIR / "data" / "documents")
    workspace_dir: str = str(BASE_DIR / "workspace")                 # 沙箱工作目录

    # ---------- 沙箱 ----------
    sandbox_timeout: int = int(os.getenv("SANDBOX_TIMEOUT", "15"))  # 秒

    def ensure_dirs(self) -> None:
        for p in (self.chroma_dir, self.docs_dir, self.workspace_dir):
            Path(p).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.llm_api_key:
            raise ValueError(
                "未配置 LLM_API_KEY，请复制 .env.example 为 .env 并填入你的 API Key"
            )


config = Config()
