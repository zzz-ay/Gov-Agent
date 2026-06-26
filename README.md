# GOV-RAG 政策法规与公共服务知识库 RAG Agent 平台

GOV-RAG 是一个面向政策咨询、法规解释、合规风险和类案参考的多知识源 RAG Agent Demo 项目。

```text
真实公开数据驱动的政策法规与公共服务知识库 RAG Agent 平台
```

当前 Demo 重点不是导入海量数据，而是展示一套完整、可解释、可评测的 RAG Agent 工程闭环：

```text
多知识源数据 -> Source Router -> Hybrid Retrieval -> Evidence -> Agent Answer -> Logs/Feedback/Eval
```

## 1. 运行环境

```powershell
conda create -n name python=3.10
conda activate name
```

安装依赖：

```powershell
pip install -r requirements.txt
```

复制环境变量：

```powershell
copy .env.example .env
```

常用 `.env` 配置：

```env
LLM_PROVIDER=qwen
OPENAI_API_KEY=your_dashscope_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-plus

KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases
RERANKER_PROVIDER=local_rule
ENABLE_WEB_SEARCH=false
ENABLE_ADMIN_PANEL=true
ADMIN_PASSWORD=change-me
```

## 2. 项目定位

当前版本为多知识源平台：

```text
gov_policy      政策文件
laws            法律法规
legal_cases     法律案例
public_notices  公告通知，当前 Demo 暂不启用
```

三类已启用知识源分别回答不同问题：

| 知识源             | 作用                                           |
| ------------------ | ---------------------------------------------- |
| `gov_policy`     | 回答怎么办、条件是什么、材料是什么、流程是什么 |
| `laws`           | 回答依据是什么、法律怎么规定、合规边界是什么   |
| `legal_cases`    | 回答类似情况怎么处理、法院裁判思路、风险参考   |
| `public_notices` | 已预留，用于名单公示、公告通知、结果公开等     |

当前不启用公告源。后续如果要加入公告源，把真实公告文件放入：

```text
data/knowledge_sources/public_notices/
```

然后修改 `.env`：

```env
KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases,public_notices
```

最后重建向量库：

```powershell
python build_vectorstore.py
```

## 3. 业务逻辑

当前 Agent 主线如下：

```text
Intent Agent
  判断用户问题类型

Source Router Agent
  判断需要查哪些知识源：policy / law / case / notice

Retrieval Agent
  对指定知识源做 vector / BM25 / hybrid retrieval

Evidence Agent
  整理政策依据、法律依据、案例依据

Task Agent
  办事问题：整理条件、流程、材料
  法规问题：解释条款和适用边界
  风险问题：给出风险提示和依据
  类案问题：给出案例参考

Answer Agent
  生成最终回答

Safety / Compliance
  补充免责声明、核验提示和风险边界
```

Source Router 会把问题分为这些任务类型：

| 任务类型            | 示例问题                     | 默认知识源                                              |
| ------------------- | ---------------------------- | ------------------------------------------------------- |
| `policy_service`  | 就业补贴怎么申请？           | `gov_policy`                                          |
| `legal_basis`     | 劳动合同试用期怎么规定？     | `laws`                                                |
| `case_reference`  | 类似诈骗罪法院一般怎么判？   | `laws, legal_cases`                                   |
| `risk_compliance` | 虚假申请就业补贴有什么风险？ | `gov_policy, laws, legal_cases`                       |
| `public_notice`   | 补贴名单什么时候公示？       | 当前回退到`gov_policy`，后续可启用 `public_notices` |
| `general_qa`      | 通用问题                     | 按规则兜底                                              |

普通用户不会看到内部路由过程。管理员 Debug 模式下才显示 Source Router、业务流程和各 Agent 中间结果。

## 4. 数据目录

核心数据目录：

```text
data/raw_docs/                         政策文件
data/knowledge_sources/laws/           法律法规
data/knowledge_sources/legal_cases/    LeCaRDv2 小规模案例子集
data/knowledge_sources/public_notices/ 公告源预留目录
data/evaluation/                       评测样例
vectorstore/                           Chroma 向量库
```

当前 LeCaRDv2 只保留约 20-50 条代表性案例，适合本地 Demo。个人使用时不要把 55,000 多条候选案例全部导入，否则重建向量库容易卡死。

准备 LeCaRDv2 小样本：

```powershell
python scripts/prepare_lecardv2_subset.py --max-cases 30 --max-eval-cases 30 --overwrite
```

法律法规数据可放入：

```text
data/knowledge_sources/laws/
```

政策扩展文档可放入：

```text
data/raw_docs/expanded_policy/
```

更多数据接入说明见：

```text
docs/REAL_DATA_INGESTION.md
docs/POLICY_DATA_EXPANSION.md
docs/DEMO_RUNBOOK.md
```

## 5. 构建向量数据库

确认 `.env`：

```env
KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases
```

重建 Chroma：

```powershell
python build_vectorstore.py
```

构建完成后会写入：

```text
vectorstore/
```

每个 chunk 会带有规范化 metadata：

```text
knowledge_source_id
knowledge_source_name
knowledge_source_type
authority_level
source_file
source_url
region
topic
```

这使得系统可以按知识源过滤检索结果。

## 6. 启动前端

不建议直接使用默认 8501，因为 Windows 环境下可能被占用或启动较慢。推荐：

```powershell
streamlit run main.py --server.port 8510 --server.fileWatcherType none
```

访问：

```text
http://localhost:8510
```

前端能力：

```text
多会话列表
新建对话
历史对话切换
Fast Chat 快速问答
Agent Chat 深度分析
Debug 管理员调试模式
参考依据展开
用户反馈
管理员知识库管理
```

新建对话后，标题会根据第一条用户问题自动更新。

Streamlit 稳定配置写在：

```text
.streamlit/config.toml
```

当前配置关闭文件监听和使用统计，避免 Streamlit 在 Windows 下反复扫描 `data/`、`vectorstore/`、`.git/` 等目录：

```toml
[server]
fileWatcherType = "none"
runOnSave = false
folderWatchBlacklist = ["data", "vectorstore", "__pycache__", ".git"]

[browser]
gatherUsageStats = false
```

## 7. 启动 FastAPI

```powershell
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

API 文档：

```text
http://localhost:8000/docs
```

主要接口：

| 接口                                    | 作用               |
| --------------------------------------- | ------------------ |
| `POST /api/chat`                      | 普通问答           |
| `POST /api/chat/stream`               | 流式问答           |
| `POST /api/retrieval/search`          | 检索测试           |
| `POST /api/retrieval/route`           | Source Router 测试 |
| `POST /api/evaluation/retrieval`      | 检索评测           |
| `POST /api/evaluation/source-routing` | 路由评测           |
| `POST /api/evaluation/answer`         | 回答质量轻量评测   |
| `POST /api/feedback`                  | 用户反馈           |
| `GET /api/logs/chat`                  | 聊天日志           |
| `GET /api/logs/feedback`              | 反馈日志           |
| `GET /api/logs/eval-runs`             | 评测日志           |
| `GET /api/knowledge-sources`          | 知识源状态         |
| `GET /api/models/runtime`             | 模型运行配置       |

## 8. API 测试

路由测试：

```powershell
curl -X POST http://localhost:8000/api/retrieval/route ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"虚假申请就业补贴有什么风险？\"}"
```

预期会选择：

```text
gov_policy
laws
legal_cases
```

检索测试：

```powershell
curl -X POST http://localhost:8000/api/retrieval/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"劳动合同试用期怎么规定？\",\"top_k\":5,\"mode\":\"hybrid\",\"source_ids\":[\"laws\"]}"
```

问答测试：

```powershell
curl -X POST http://localhost:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"企业收集员工个人信息需要注意哪些合规要求？\",\"auto_route\":true}"
```

如果 Windows `curl` 中文转义不稳定，也可以用 Python TestClient 或 API docs 页面测试。

## 9. 评测

平台级轻量评测：

```powershell
python scripts/run_platform_eval_file.py
```

Source Router 评测：

```powershell
python scripts/run_source_routing_eval_file.py
```

当前预期：

```text
Routing accuracy: 1.0000
Source accuracy: 1.0000
Task accuracy: 1.0000
```

可选检索评测：

```powershell
python scripts/run_retrieval_eval_file.py --input data/evaluation/lecardv2_retrieval_eval.jsonl --modes hybrid --source-ids legal_cases
```

说明：如果评测文件中文在 PowerShell 中显示乱码，优先使用 Python 读取 UTF-8 文件，不要直接根据终端显示判断数据损坏。

## 10. 稳定性设计

本项目考虑到在内存有限机器上运行情况下，对重任务做了运行时保护：

1. LeCaRDv2 不全量导入，只保留约 20-50 条代表性案例。
2. Chroma 向量库通过离线脚本构建，避免前端页面里触发大规模重建。
3. BM25 使用磁盘缓存：

```text
data/processed/bm25_index.pkl
```

`.env` 默认关闭运行时自动重建：

```env
BM25_AUTO_REBUILD=false
```

如果 BM25 缓存存在但损坏或过期，前端运行时不会自动全量读取文档、切块和 tokenize，而是跳过运行时重建，避免内存被打满。需要更新 BM25 时，应离线重建缓存，不在用户页面里边跑边建。

## 11. 日志与反馈

运行数据写入：

```text
data/runtime/gov_rag_runtime.db
data/runtime/*.jsonl
```

包含：

```text
chat logs
feedback logs
eval runs
model runtime
source routing result
evidence result
source distribution
```

这些数据用于后续 badcase 分析和 RAG 质量改进。

## 12. 推荐演示问题

政策服务：

```text
就业补贴怎么申请？
办理居住证需要什么材料？
```

法规解释：

```text
劳动合同试用期怎么规定？
企业收集员工个人信息需要注意哪些合规要求？
```

风险合规：

```text
虚假申请就业补贴有什么风险？
```

类案参考：

```text
类似诈骗罪法院一般怎么判？
```

## 13. 项目亮点

- Source Router 可以自动选择知识源，避免所有问题混在一个知识库里查。
- 检索支持 vector、BM25、hybrid 和本地 rerank。
- 非办事类问题会跳过不必要的政务 Agent，减少无效链路。
- FastAPI 提供 chat、retrieval、route、evaluation、feedback、logs 等完整接口。
- Streamlit 前端支持多会话、用户问答、参考依据和管理员调试。
- 评测不只看检索命中，也看知识源路由和任务类型是否正确。
- 针对本地 Demo 场景增加 BM25 磁盘缓存和 Streamlit 文件监听黑名单，避免前端运行时 OOM。

## 14. 当前边界

- 当前公告源 `public_notices` 暂不启用，只保留接入能力。
- LeCaRDv2 当前是小规模代表性样本，不是完整大库。
- Answer Eval 是轻量规则评测，不是 LLM-as-judge。
- `bge_cross_encoder` 和 `dashscope_rerank` 是后续扩展方向，当前默认使用 `local_rule`。
- 具体政策办理、监管口径和司法判断仍需以主管部门、正式法律文书和专业意见为准。
