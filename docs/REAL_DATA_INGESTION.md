# Real Data Ingestion

This project should use real public datasets instead of hand-written demo data
when it is presented as a policy, law, and compliance RAG system.

## Recommended Source Types

Start with sources that are naturally related to the original government-policy
RAG project:

```text
gov_policy      Existing government policy documents in data/raw_docs
laws            Public laws and regulations
legal_cases     Public legal-case retrieval datasets
public_notices  Public notices, announcements, or disclosure documents
```

## Laws

Put downloaded law/regulation source files outside the final knowledge-source
directory first, for example:

```text
data/imports/laws_raw/
```

Supported input formats:

```text
json
jsonl
csv
txt
md
```

The ingestion script recognizes common Chinese and English field names:

```text
title / 标题 / 法规标题 / 法律名称
content / text / 正文 / 内容 / 全文
source_org / 发布机关 / 制定机关 / 发布机构
publish_date / 发布日期 / 公布日期 / 施行日期
source_url / 来源链接 / 原文链接
region / 适用地区
topic / 主题 / 类别
doc_type / 文件类型 / 法规类型
authority_level / 效力级别 / 层级
```

Normalize the files into the `laws` knowledge source:

```bash
python scripts/ingest_laws.py --input data/imports/laws_raw
```

The normalized files are written to:

```text
data/knowledge_sources/laws
```

Then enable the source:

```env
KNOWLEDGE_SOURCES=gov_policy,laws
```

Rebuild the vector store:

```bash
python build_vectorstore.py
```

Test retrieval:

```text
POST /api/retrieval/search
```

Example request:

```json
{
  "query": "劳动合同试用期有什么规定",
  "top_k": 5,
  "mode": "hybrid"
}
```

## Why This Approach

The system does not bundle fake law text. Instead, it provides a repeatable
normalization pipeline so real public law datasets can be imported with clear
metadata and provenance.

Each imported document becomes a regular RAG document with:

```text
knowledge_source_id=laws
knowledge_source_type=law
authority_level=law
```

This keeps the original government-policy RAG intact while extending it into a
real public policy, law, and compliance knowledge-base system.

## Legal Cases

Legal-case datasets are useful for demonstrating retrieval quality and reranking
because they often include query-case relevance pairs.

### LeCaRDv2

LeCaRDv2 is the first real legal-case dataset integrated by this project. Keep
the downloaded raw files inside the project:

```text
data/imports/legal_cases_raw/lecardv2/
```

Expected local structure:

```text
data/imports/legal_cases_raw/lecardv2/
  repo/                  # GitHub repository: query and label files
  candidate.tar          # Google Drive candidate archive
  candidate_extracted/   # extracted candidate_55192 JSON files
```

Prepare a smaller RAG-ready subset for this demo project:

```bash
python scripts/prepare_lecardv2_subset.py --max-cases 30 --max-eval-cases 30 --overwrite
```

Outputs:

```text
data/knowledge_sources/legal_cases/
data/evaluation/lecardv2_retrieval_eval.jsonl
```

This imports a deliberately small, representative subset instead of all 55,192
candidate cases. For this local demo, keep the case corpus around 20-50 cases so
Chroma rebuilds remain practical and the laptop stays responsive.

Put downloaded case files outside the final knowledge-source directory first:

```text
data/imports/legal_cases_raw/
```

Supported input formats:

```text
json
jsonl
csv
txt
md
```

The ingestion script recognizes common Chinese and English field names:

```text
case_id / 案例编号 / 案件编号
title / case_name / 案由 / 案件名称
content / text / fact / 案情 / 事实 / 全文
court / 法院 / 审理法院
judgment_date / 裁判日期 / 判决日期
case_type / 案件类型
cause / 案由 / 罪名
source_url / 来源链接 / 原文链接
query / question / 检索问题 / 问题
expected_keywords / keywords / 关键词
```

Normalize the files into the `legal_cases` knowledge source:

```bash
python scripts/ingest_legal_cases.py --input data/imports/legal_cases_raw
```

If the dataset contains query/question fields, also export lightweight retrieval
eval cases:

```bash
python scripts/ingest_legal_cases.py ^
  --input data/imports/legal_cases_raw ^
  --eval-output data/evaluation/legal_cases_retrieval.jsonl
```

Then enable the source:

```env
KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases
```

Rebuild the vector store:

```bash
python build_vectorstore.py
```

Example retrieval request:

```json
{
  "query": "劳动合同解除后经济补偿如何认定",
  "top_k": 5,
  "mode": "hybrid"
}
```

Each imported case document receives:

```text
knowledge_source_id=legal_cases
knowledge_source_type=case
authority_level=case
```
