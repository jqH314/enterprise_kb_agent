# 企业知识库问答 Agent

一个基于 **RAG + Function Calling** 的企业内部文档智能问答系统。不只是"文档搜索"，而是一个能自主判断该查资料、该跑计算、答案可溯源的执行型 Agent。

**在线体验**：https://enterprisekbagent-pyhrqgt56s8awswtsokim5.streamlit.app

---

## ✨ 功能特性

- **多格式文档解析**：PDF / Word(docx) / Excel(xlsx) / CSV / TXT / Markdown
- **混合检索**：向量语义检索 + BM25 关键词检索，RRF 融合召回
- **重排序精排**：bge-reranker Cross-encoder 对召回结果二次排序
- **引用溯源**：答案标注来源文件名与页码，防止幻觉
- **工具调用 Agent**：自主决策调用知识库检索、沙箱代码执行、文件读取等工具
- **沙箱执行**：子进程隔离 + 超时控制 + 工作目录隔离，安全运行 LLM 生成的代码
- **本地 Embedding**：BGE 中文向量模型，数据不出域，GPU 自动加速
- **对话界面**：Streamlit Web UI，可折叠查看 Agent 思考轨迹

---

## 🏗 系统架构

```
用户问题
   │
   ▼
┌─────────────────────────┐
│  Agent 调度循环          │  ← 自实现 Function Calling 调度器
│  (core/agent.py)         │
└─────────────────────────┘
   │
   ├─── 调用工具 ──────────────┐
   │                           │
   ▼                           ▼
┌──────────┐          ┌──────────────┐
│ 知识库检索 │          │ 沙箱代码执行  │
│ (tools/) │          │ (tools/)     │
└────┬─────┘          └──────────────┘
     │
     ▼
┌─────────────────────────┐
│  RAG 检索管线            │
│  ┌───────┐  ┌─────────┐ │
│  │ 向量   │  │  BM25   │ │
│  │ 检索   │  │ 关键词  │ │
│  └───┬───┘  └────┬────┘ │
│      └─────┬──────┘      │
│         RRF 融合          │
│             │            │
│         Rerank 精排       │
│             │            │
│         返回 top-k 片段   │
└─────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10 ~ 3.12
- 推荐有 NVIDIA GPU（用 CPU 也能跑，模型很小）

### 安装

```bash
# 1. 安装 PyTorch（有显卡选 CUDA 版，否则 CPU 版即可）
pip install torch

# 2. 安装项目依赖
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 LLM API Key（推荐 DeepSeek，也兼容任意 OpenAI 格式接口）：

```env
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的key
LLM_MODEL=deepseek-chat
```

### 建索引

把你的文档（PDF/Word/Excel 等）放进 `data/documents/`，然后：

```bash
python main.py --build
```

首次运行会自动下载 BGE 模型（约 100MB），需要联网。

### 运行

```bash
# 命令行问答
python main.py --ask "公司的差旅报销标准是什么？"

# 或启动 Web 界面
streamlit run app.py
```

---

## 📁 项目结构

```
enterprise_kb_agent/
├── main.py                  # CLI 入口
├── app.py                   # Streamlit Web 界面
├── config.py                # 全局配置
├── .env.example
├── requirements.txt
├── core/
│   ├── llm.py               # LLM 客户端（Function Calling + 流式）
│   ├── agent.py             # Agent 主循环
│   ├── tool.py              # 工具注册系统（装饰器 + 自动生成 Schema）
│   └── sandbox.py           # 沙箱执行器
├── rag/
│   ├── loader.py            # 多格式文档加载
│   ├── splitter.py          # 递归文本切分
│   ├── embedding.py         # BGE 本地 Embedding
│   ├── vectorstore.py       # NumPy 向量存储与检索
│   ├── retriever.py         # 混合检索 + RRF + Rerank
│   └── pipeline.py          # RAG 流水线
├── tools/
│   ├── kb_search.py          # 知识库检索工具
│   ├── code_exec.py          # 沙箱 Python 执行
│   └── file_tools.py         # 受限文件操作
└── data/
    ├── documents/            # 原始文档
    └── chroma_db/            # 向量索引（自动生成）
```

---

## 🔍 检索流程说明

以"北京出差住宿标准是多少"为例：

1. **召回**：查询同时走向量检索（语义相似）和 BM25（关键词匹配），各取 top-8
2. **融合**：用 RRF（Reciprocal Rank Fusion）合并两路结果
3. **精排**：bge-reranker 对融合后的候选重新打分，取 top-4
4. **生成**：把 top-4 片段 + 用户问题交给 LLM，生成答案并标注来源

向量检索擅长"意思相近"，BM25 擅长"专有名词/编号精确命中"，两者互补。

---

## ☁️ 部署到 Streamlit Cloud

1. 把代码 push 到 GitHub
2. 打开 [share.streamlit.io](https://share.streamlit.io)，选择仓库，主文件选 `app.py`
3. 在 App Settings → Secrets 中填入：
   ```toml
   LLM_BASE_URL = "https://api.deepseek.com/v1"
   LLM_API_KEY = "sk-你的key"
   LLM_MODEL = "deepseek-chat"
   ```
4. 部署完成后点侧边栏「重建索引」即可

> 云端为 CPU 环境，BGE-small 模型较小，问答延迟可接受。

---

## 🔧 技术栈

| 模块 | 选型 |
|---|---|
| LLM | DeepSeek / 任意 OpenAI 兼容接口 |
| Embedding | BAAI/bge-small-zh-v1.5（本地） |
| Rerank | BAAI/bge-reranker-base（本地） |
| 向量检索 | NumPy 余弦相似度 |
| 关键词检索 | rank_bm25 |
| Web 界面 | Streamlit |

---

## 📌 已知限制

- 当前为单轮对话，暂不支持多轮上下文追问
- 文档解析以文本为主，不支持扫描件 OCR 和 PDF 表格还原
- 沙箱为进程级隔离，非容器级强隔离
- 向量存储基于 NumPy，适合万级以内文档规模

## License

MIT
