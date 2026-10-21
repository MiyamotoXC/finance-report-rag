# 金融研报 RAG（finance-report-rag）

> 一个**金融研报检索与问答** RAG 系统：把研报向量化入库，支持自然语言提问检索相关段落并由 LLM 生成带引用的答案；
> 额外提供**研报结构化抽取**（行业 / 评级 / 核心观点 / 关注风险），把非结构化研报转成可横向对比的字段表。
> 适用于"研报太多读不过来、想快速对比多家机构观点"的研究场景。

> 作者：[@MiyamotoXC](https://github.com/MiyamotoXC) ｜ 许可证：MIT ｜ 欢迎 Star / Fork 二次开发。

## 一、特性

- **研报检索问答**：云端向量化（DashScope Embedding）+ 余弦检索 + LLM 生成，答案标注来源、区分事实与观点
- **结构化抽取**：`extract.py` 把每篇研报抽成 行业/评级/核心观点/关注风险 的 JSON，便于横向对比
- **轻量自研**：向量检索用 `numpy` 余弦，不依赖 langchain/faiss，可读可改
- **密钥走环境变量**：`DASHSCOPE_API_KEY` 不入库

## 二、目录结构

```
finance-report-rag/
├── config.py          # 配置（密钥走环境变量）
├── ingest.py          # 解析 .md -> 切分 -> 向量化 -> numpy 索引落盘
├── qa.py              # 加载 -> 检索 -> LLM 生成（带引用）
├── extract.py         # 研报结构化抽取（行业/评级/观点/风险）
├── example_reports/   # 脱敏模拟研报（新能源 / 半导体 / 消费）
├── requirements.txt
└── LICENSE
```

## 三、快速开始

```bash
pip install -r requirements.txt
export DASHSCOPE_API_KEY="你的密钥"      # Windows: set DASHSCOPE_API_KEY=...

# 1) 构建索引
python ingest.py

# 2) 提问
python qa.py "哪些行业被给出增持评级？"

# 3) 结构化抽取（生成 reports.json 对比表）
python extract.py
```

## 四、配置（config.py）

| 项 | 默认 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | 环境变量 | Embedding 与 LLM 密钥（必需，不入库） |
| `EMBEDDING_MODEL` | `text-embedding-v1` | 向量模型 |
| `LLM_MODEL` | `qwen-plus` | 生成 / 抽取模型 |
| `DATA_DIR` | `example_reports` | 研报目录 |
| `INDEX_DIR` | `vector_store` | 索引落盘目录（已被 .gitignore 忽略） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 600 / 80 | 切分参数 |

## 五、能力边界（诚实说明）

- 示例研报为**脱敏的通用行业模板**，不含真实机构观点、公司或数据，仅作技术演示
- 结构化抽取依赖 LLM，字段可能不完整，关键决策以原文为准
- 向量检索为余弦相似度，未做重排（rerank），大规模语料建议接 reranker

## 六、License

代码以 MIT 发布。
