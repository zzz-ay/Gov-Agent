# Policy Data Expansion

The original policy corpus focuses on graduate public-service policies. To turn
the project into a broader public policy, law, and compliance RAG platform, the
policy layer should be expanded with real official documents.

## Target Policy Categories

```text
employment_entrepreneurship
social_security_medical_insurance
household_registration_residence_permit
talent_subsidy
housing_provident_fund
labor_employment
administrative_license_penalty
data_security_personal_information
enterprise_compliance
```

## Current Rule

Do not generate fake policy files.

If there is no reliable open dataset package, use a curated URL manifest of real
official policy pages and convert those pages into normalized RAG documents.

## URL Manifest

Manifest path:

```text
data/imports/policy_url_manifest.json
```

Format:

```json
{
  "items": [
    {
      "category": "employment_entrepreneurship",
      "title": "policy title",
      "url": "https://official.example/policy.html",
      "source_org": "official department",
      "publish_date": "2025-01-01",
      "region": "national",
      "topic": "就业创业",
      "doc_type": "policy",
      "authority_level": "policy"
    }
  ]
}
```

Fetch pages:

```bash
python scripts/fetch_policy_url_manifest.py
```

Output:

```text
data/raw_docs/expanded_policy/
```

These files are loaded as `gov_policy` because they are expanded policy
documents, not legal-case or law corpora.

Current seed batch:

```text
data/raw_docs/expanded_policy/
```

The seed batch currently contains official pages from China Government Network
and State Council Gazette, including employment, medical insurance, residence
permit, housing provident fund, data security, and enterprise compliance
documents. The manifest can be expanded category by category without changing
the ingestion code.

## Laws

The legal-law source should use the National Laws and Regulations Database when
the official API is reachable.

Script:

```bash
python scripts/fetch_flk_laws.py
```

Output:

```text
data/knowledge_sources/laws/flk/
```

If the FLK endpoint is unstable, retry in smaller batches with a terms file:

```bash
python scripts/fetch_flk_laws.py --terms-file data/imports/flk_law_terms.txt
```

Current imported FLK seed:

```text
data/knowledge_sources/laws/flk/
```

The importer uses FLK metadata plus the official FLK docx download endpoint.
Raw downloaded files are kept in:

```text
data/imports/laws_raw/flk/
```

The normalized `.txt` files contain metadata headers plus full law text, so they
can be indexed by the existing `laws` knowledge source.

## Already Completed

LeCaRDv2 legal cases have been downloaded, but the current demo intentionally
uses a small representative subset of about 20-50 cases to avoid slow Chroma
rebuilds and local machine freezes. The normalized case files live in:

```text
data/knowledge_sources/legal_cases/
```

The generated retrieval eval file is:

```text
data/evaluation/lecardv2_retrieval_eval.jsonl
```
