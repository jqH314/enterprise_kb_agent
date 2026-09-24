# 企业知识库问答 Agent（RAG + Agent 双范式）

一个面向企业内部资料的智能问答系统。不只是"文档检索问答"，而是一个**能自主判断该查资料、该跑计算、能溯源**的执行型 Agent。

> 适合简历项目 / AI 应用开发方向。每一个模块都可以在面试里展开讲原理。

---

## ✨ 核心亮点

| 模块 | 做了什么 | 面试可聊的点 |
|---|---|---|
| **混合检索** | 向量语义检索（BGE）+ BM25 关键词检索，RRF 融合 | 向量擅长语义、BM25 擅长专有名词/编号精确命中，互补 |
| **Rerank 精排** | bge-reranker Cross-encoder 对召回结果二次排序 | 双塔召回快但粗，Cross-encoder 精排准但慢，工程权衡 |
| **引用溯源** | 答案标注来源文件名/页码 | 企业场景刚需，防止幻觉 |
| **Function Calling Agent** | 自实现调度循环（非黑盒框架），多步推理 | ReAct / Tool Calling 原理，每一步都能讲 |
| **沙箱执行** | 子进程隔离 + 超时 + 工作目录隔离 + 环境变量白名单 | 让 LLM 安全跑代码，资源隔离思路 |
| **多格式解析** | PDF / Word / Excel / TXT / Markdown | 真实企业文档的多样性 |
| **本地 Embedding** | BGE 本地推理，数据不出域，GPU 自动加速 | 私有化部署卖点 |

---

## 🧱 技术栈

- **LLM**：DeepSeek / 任意 OpenAI 兼容接口（Function Calling）
- **Embedding / Rerank**：BAAI/bge-small-zh-v1.5 + bge-reranker-base（本地 GPU）
- **向量库**：Chroma（持久化，cosine 距离）
- **混合检索**：rank_bm25（BM25）+ RRF 融合
- **Agent**：自实现 Tool Calling 调度循环
- **沙箱**：Python subprocess + 超时 + 目录隔离
- **界面**：Streamlit

---

## 🗂 项目结构

```
enterprise_kb_agent/
├── main.py                  # CLI 入口（建索引 / 问答 / 交互）
├── app.py                   # Streamlit Web 界面
├── config.py                # 全局配置（读 .env）
├── .env.example
├── core/
│   ├── llm.py               # LLM 客户端（function calling + 流式）
│   ├── agent.py             # Agent 主循环（核心调度）
│   ├── tool.py              # 工具注册系统（装饰器 + 自动生成 Schema）
│   └── sandbox.py           # 沙箱执行器（隔离/超时/输出截断）
├── rag/
│   ├── loader.py            # 多格式文档加载
│   ├── splitter.py          # 递归字符切分（带 overlap）
│   ├── embedding.py         # 本地 BGE / OpenAI Embedding 可切换
│   ├── vectorstore.py       # Chroma 封装
│   ├── retriever.py          # 混合检索 + RRF 融合 + Rerank
│   └── pipeline.py          # RAG 流水线封装
├── tools/
│   ├── kb_search.py          # 知识库检索工具（Agent 调用）
│   ├── code_exec.py         # 沙箱 Python 执行工具
│   └── file_tools.py        # 受限文件读写（防路径逃逸）
└── data/
    ├── documents/           # 把你的文档放这里
    └── chroma_db/           # 向量库持久化
```

---

## 🚀 快速开始

### 1. 安装依赖（建议 Python 3.10–3.12）

```bash
# 先装 PyTorch（有 NVIDIA 显卡选 CUDA 版，否则 CPU 版；这里以 CUDA 12.1 为例）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 再装项目依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY（DeepSeek 控制台申请，很便宜）
```

### 3. 放入文档并建索引

把 PDF / Word / Excel 丢进 `data/documents/`，然后：

```bash
python main.py --build
```

### 4. 问答

```bash
# 单轮
python main.py --ask "公司的差旅报销标准是什么？"

# 或开 Web 界面
streamlit run app.py
```

---

## ☁️ 部署到 Streamlit Cloud（免费公网访问）

1. **把项目 push 到 GitHub**（记得 `.env` 已被 `.gitignore` 排除，不会泄露 key）
2. 打开 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 登录
3. 选择你的仓库，主文件选 `app.py`
4. 在 App  settings → Secrets 里填入：
   ```toml
   LLM_BASE_URL = "https://api.deepseek.com/v1"
   LLM_API_KEY = "sk-你的key"
   LLM_MODEL = "deepseek-chat"
   ```
5. 首次启动会自动下载 BGE 模型（约 1-2 分钟），之后就快了
6. 启动后点侧边栏「重建索引」即可把 `data/documents/` 里的文档入库

> 云端是 CPU 环境，bge-small 模型很小，问答速度可接受。

---

## 🧠 它是怎么工作的（面试讲解版）

以"对比 A、B 两个部门今年的报销标准差异"为例：

1. **用户提问** → Agent 把问题 + 工具清单发给 LLM
2. **LLM 决策**：先调 `search_knowledge_base("A部门报销标准")`，再调一次查 B 部门
3. **检索链路**：query 同时走向量库和 BM25 → RRF 融合 → Rerank 精排，返回 top-4 片段（带来源）
4. **LLM 拿到资料后**，发现需要计算差值，调 `run_python` 在沙箱里跑代码
5. **最终生成答案**，关键结论后标注【来源:xxx.pdf】

整个过程最多 `MAX_STEPS` 轮，防止死循环；Web 界面可折叠查看完整思考轨迹。

---

## 💼 简历项目描述（可直接套用）

> **企业知识库问答 Agent（RAG + Function Calling）** | Python / LLM
> - 基于 RAG 架构实现企业内部文档智能问答，支持 PDF/Word/Excel 多格式解析、递归切分、BGE 向量入库
> - 设计**向量 + BM25 混合检索 + RRF 融合 + bge-reranker 精排**的三段式检索链路，相比单路向量检索召回更全、答案更准
> - 基于 Function Calling 自实现 Agent 调度循环，让模型自主调用知识库检索、沙箱代码执行、文件读取等工具完成多步推理
> - 实现子进程沙箱（超时/目录隔离/环境变量白名单），安全执行 LLM 生成的代码；答案带文档来源引用，支持溯源
> - Streamlit 实现对话 UI，可可视化 Agent 思考轨迹

---

## 🔧 可扩展方向（面试加分项）

- 接入 **RAGAS** 做检索/生成质量自动评测
- 加 **查询改写 / HyDE** 优化首轮召回
- 向量库换 **Milvus / Qdrant** 支持千万级文档
- 加 **多轮对话记忆**（当前是单轮）
- 沙箱升级为 **Docker 容器级隔离**
