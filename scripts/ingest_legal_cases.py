import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "knowledge_sources" / "legal_cases"
DEFAULT_EVAL_OUTPUT = BASE_DIR / "data" / "evaluation" / "legal_cases_retrieval.jsonl"


FIELD_ALIASES = {
    "case_id": ["case_id", "id", "doc_id", "pid", "cid", "案例编号", "案件编号"],
    "title": ["title", "case_name", "name", "案由", "案件名称", "标题"],
    "content": ["content", "text", "body", "fact", "facts", "案情", "事实", "全文", "内容"],
    "court": ["court", "法院", "审理法院"],
    "judgment_date": ["judgment_date", "date", "裁判日期", "判决日期"],
    "case_type": ["case_type", "type", "案件类型", "案例类型"],
    "cause": ["cause", "案由", "罪名"],
    "source_url": ["source_url", "url", "link", "来源链接", "原文链接"],
    "query": ["query", "question", "q", "检索问题", "问题", "查询"],
    "expected_keywords": ["expected_keywords", "keywords", "关键词"],
}


def first_value(item: Dict[str, Any], field: str, default: str = "") -> str:
    for key in FIELD_ALIASES[field]:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = " ".join(str(part).strip() for part in value if str(part).strip())
        value = str(value).strip()
        if value:
            return value
    return default


def safe_filename(value: str, fallback: str) -> str:
    value = value.strip() or fallback
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value[:100].strip("._")
    return value or fallback


def iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for key in ["data", "records", "items", "cases", "documents", "candidates", "results"]:
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield payload


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                yield item


def iter_csv_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def record_from_text_file(path: Path) -> Dict[str, Any]:
    return {
        "case_id": path.stem,
        "title": path.stem,
        "content": path.read_text(encoding="utf-8"),
    }


def iter_records(input_path: Path) -> Iterable[Dict[str, Any]]:
    files: List[Path]
    if input_path.is_dir():
        files = [path for path in input_path.rglob("*") if path.is_file()]
    else:
        files = [input_path]

    for path in sorted(files):
        suffix = path.suffix.lower()
        if suffix == ".json":
            yield from iter_json_records(path)
        elif suffix == ".jsonl":
            yield from iter_jsonl_records(path)
        elif suffix == ".csv":
            yield from iter_csv_records(path)
        elif suffix in {".txt", ".md"}:
            yield record_from_text_file(path)
        else:
            print(f"[skip] unsupported file type: {path}")


def normalize_case(item: Dict[str, Any], index: int) -> Dict[str, str]:
    case_id = first_value(item, "case_id", default=f"case_{index:04d}")
    title = first_value(item, "title", default=case_id)
    content = first_value(item, "content")

    return {
        "case_id": case_id,
        "title": title,
        "content": content,
        "court": first_value(item, "court"),
        "judgment_date": first_value(item, "judgment_date"),
        "case_type": first_value(item, "case_type", default="legal_case"),
        "cause": first_value(item, "cause"),
        "source_url": first_value(item, "source_url"),
        "query": first_value(item, "query"),
        "expected_keywords": first_value(item, "expected_keywords"),
    }


def render_case_document(record: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Title: {record['title']}",
            f"Source Org: {record['court']}",
            f"Publish Date: {record['judgment_date']}",
            f"Source URL: {record['source_url']}",
            "Region: national",
            f"Topic: {record['cause'] or 'legal_case'}",
            "Document Type: legal_case",
            "Authority Level: case",
            f"Case ID: {record['case_id']}",
            f"Case Type: {record['case_type']}",
            "",
            "Content:",
            record["content"].strip(),
            "",
        ]
    )


def parse_keywords(value: str) -> List[str]:
    if not value:
        return []
    return [
        item.strip()
        for item in re.split(r"[,，;；\s]+", value)
        if item.strip()
    ]


def ingest_legal_cases(
    input_path: Path,
    output_dir: Path,
    eval_output: Path = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if eval_output:
        eval_output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    eval_written = 0
    eval_lines: List[str] = []

    for index, item in enumerate(iter_records(input_path), start=1):
        record = normalize_case(item, index)
        if not record["content"].strip():
            skipped += 1
            continue

        filename_seed = f"{record['case_id']}_{record['title']}"
        filename = safe_filename(filename_seed, fallback=f"case_{index:04d}") + ".txt"
        output_path = output_dir / filename

        if output_path.exists() and not overwrite:
            stem = output_path.stem
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{stem}_{counter}.txt"
                counter += 1

        output_path.write_text(render_case_document(record), encoding="utf-8")
        written += 1

        if eval_output and record["query"]:
            eval_case = {
                "question": record["query"],
                "gold_source_file": output_path.name,
                "expected_keywords": parse_keywords(record["expected_keywords"]),
            }
            eval_lines.append(json.dumps(eval_case, ensure_ascii=False))
            eval_written += 1

    if eval_output and eval_lines:
        mode = "w" if overwrite else "a"
        with eval_output.open(mode, encoding="utf-8") as f:
            for line in eval_lines:
                f.write(line + "\n")

    return {
        "written": written,
        "skipped": skipped,
        "eval_written": eval_written,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize real public legal-case datasets into the GOV-RAG legal_cases knowledge source."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory. Supports json, jsonl, csv, txt, and md.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Defaults to data/knowledge_sources/legal_cases.",
    )
    parser.add_argument(
        "--eval-output",
        default="",
        help="Optional JSONL eval-case output path. Use this when records contain query/question fields.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files with the same normalized case id/title.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    eval_output = Path(args.eval_output).resolve() if args.eval_output else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    result = ingest_legal_cases(
        input_path=input_path,
        output_dir=output_dir,
        eval_output=eval_output,
        overwrite=args.overwrite,
    )

    print("========== Legal Cases Ingestion Complete ==========")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Written: {result['written']}")
    print(f"Skipped empty records: {result['skipped']}")
    if eval_output:
        print(f"Eval output: {eval_output}")
        print(f"Eval cases written: {result['eval_written']}")
    print("")
    print("Next steps:")
    print("1. Set KNOWLEDGE_SOURCES=gov_policy,laws,legal_cases in .env")
    print("2. Run python build_vectorstore.py")
    print("3. Test POST /api/retrieval/search with mode=hybrid")


if __name__ == "__main__":
    main()
