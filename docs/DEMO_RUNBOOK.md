# 政策法规 RAG Agent 平台演示 Runbook

```text
真实公开数据驱动的政策法规与公共服务知识库 RAG Agent 平台
```

当前 Demo 重点不是堆很多数据，而是展示完整工程闭环：

```text
多知识源 -> 自动路由 -> 混合检索 -> 证据组织 -> Agent 回答 -> 日志/反馈/评测
```

## 当前知识源

```text
gov_policy      政策文件，包含原毕业生政策和少量扩展政策
laws            法律法规
legal_cases     LeCaRDv2 小规模代表性案例子集
public_notices  公告通知，已预留知识源，当前可后续补数据
```

当前建议 `.env`：

```env
KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases
```

LeCaRDv2 案例保持在约 20-50 条，适合本地 Demo；

## 公告源暂不启用

当前 Demo 暂时不启用 `public_notices`，因为政策、法规、案例三类知识源已经可以完整展示平台主线。

公告源相关能力已经预留好，后续加入时按这个顺序做即可：

```text
1. 将公告、名单公示、结果公示、通知通告等真实文件放入 data/knowledge_sources/public_notices/
2. 确保文本中带有标题、发布机构、发布时间、来源链接等元数据
3. 在 .env 中改为 KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases,public_notices
4. 运行 python build_vectorstore.py 重建向量库
5. 用 “补贴申报名单什么时候公示？” 这类问题验证 Source Router 是否选择 public_notices
```

## 启动方式

启动 Streamlit 前端：

```bash
streamlit run main.py
```

默认访问：

```text
http://localhost:8501
```

启动 FastAPI：

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

API 文档：

```text
http://localhost:8000/docs
```

## 推荐演示问题

政策服务类：

```text
就业补贴怎么申请？
办理居住证需要什么材料？
```

法律法规类：

```text
劳动合同试用期怎么规定？
企业收集员工个人信息需要注意哪些合规要求？
```

风险合规类：

```text
虚假申请就业补贴有什么风险？
```

类案参考类：

```text
类似诈骗罪法院一般怎么判？
```

## Source Router 演示

可以直接验证路由结果：

```bash
python scripts/run_source_routing_eval_file.py
```

当前轻量样例应看到：

```text
Accuracy: 1.0000
Source accuracy: 1.0000
Task accuracy: 1.0000
```

路由任务类型：

```text
policy_service    政策办事
legal_basis       法规依据
case_reference    类案参考
risk_compliance   风险合规
public_notice     公告通知
general_qa        通用问答
```

## 平台级评测

默认轻量评测：

```bash
python scripts/run_platform_eval_file.py
```

可选检索评测：

```bash
python scripts/run_retrieval_eval_file.py --input data/evaluation/lecardv2_retrieval_eval.jsonl --modes hybrid --source-ids legal_cases
```

## 演示时强调的工程亮点

- 原项目不是重写，而是在已有 Streamlit + LangGraph RAG 项目上做平台化升级。
- 数据层从单一毕业生政策扩展为政策、法规、案例、公告四类知识源。
- Source Router Agent 会根据问题选择知识源，避免所有问题都混查。
- 法规/案例/风险问题会跳过不必要的政务办事 Agent，减少无效链路。
- 检索支持 vector、BM25、hybrid 和本地 rerank。
- API 层有 chat、retrieval、route、evaluation、feedback、logs。
- 评测不只看答案，还看 source routing 和 task type 是否正确。

## 后续收尾

```text
1. 补 public_notices 的少量真实公告样例。
2. 优化前端默认展示，用户侧隐藏内部路由和调试字段。
3. 整理 README、截图和最终演示视频/讲解稿。
4. 如果要正式交付，确认 git 仓库状态并提交一个稳定版本。
```
