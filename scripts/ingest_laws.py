import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "knowledge_sources" / "laws"


FIELD_ALIASES = {
    "title": ["title", "name", "law_title", "标题", "法规标题", "法律名称", "名称"],
    "content": ["content", "text", "body", "full_text", "正文", "内容", "全文"],
    "source_org": ["source_org", "issuer", "office", "authority", "发布机关", "制定机关", "发布机构"],
    "publish_date": ["publish_date", "date", "发布日期", "公布日期", "施行日期"],
    "source_url": ["source_url", "url", "link", "来源链接", "原文链接"],
    "region": ["region", "area", "适用地区", "地区"],
    "topic": ["topic", "category", "subject", "主题", "类别"],
    "doc_type": ["doc_type", "type", "document_type", "文件类型", "法规类型"],
    "authority_level": ["authority_level", "level", "效力级别", "层级"],
}


def first_value(item: Dict[str, Any], field: str, default: str = "") -> str:
    for key in FIELD_ALIASES[field]:
        value = item.get(key)
        if value is None:
            continue
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
        for key in ["data", "records", "items", "laws", "results"]:
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
        "title": path.stem,
        "content": path.read_text(encoding="utf-8"),
        "source_org": "",
        "publish_date": "",
        "source_url": "",
        "region": "national",
        "topic": "law",
        "doc_type": "law",
        "authority_level": "law",
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


def normalize_record(item: Dict[str, Any], index: int) -> Dict[str, str]:
    title = first_value(item, "title", default=f"law_{index:04d}")
    content = first_value(item, "content")

    return {
        "title": title,
        "source_org": first_value(item, "source_org"),
        "publish_date": first_value(item, "publish_date"),
        "source_url": first_value(item, "source_url"),
        "region": first_value(item, "region", default="national"),
        "topic": first_value(item, "topic", default="law"),
        "doc_type": first_value(item, "doc_type", default="law"),
        "authority_level": first_value(item, "authority_level", default="law"),
        "content": content,
    }


def render_law_document(record: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Title: {record['title']}",
            f"Source Org: {record['source_org']}",
            f"Publish Date: {record['publish_date']}",
            f"Source URL: {record['source_url']}",
            f"Region: {record['region']}",
            f"Topic: {record['topic']}",
            f"Document Type: {record['doc_type']}",
            f"Authority Level: {record['authority_level']}",
            "",
            "Content:",
            record["content"].strip(),
            "",
        ]
    )


def ingest_laws(input_path: Path, output_dir: Path, overwrite: bool = False) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    for index, item in enumerate(iter_records(input_path), start=1):
        record = normalize_record(item, index)
        if not record["content"].strip():
            skipped += 1
            continue

        filename = safe_filename(record["title"], fallback=f"law_{index:04d}") + ".txt"
        output_path = output_dir / filename

        if output_path.exists() and not overwrite:
            stem = output_path.stem
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{stem}_{counter}.txt"
                counter += 1

        output_path.write_text(render_law_document(record), encoding="utf-8")
        written += 1

    return {"written": written, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(
        description="Normalize real public law/regulation files into the GOV-RAG laws knowledge source."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or directory. Supports json, jsonl, csv, txt, and md.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory. Defaults to data/knowledge_sources/laws.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files with the same normalized title.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    result = ingest_laws(input_path=input_path, output_dir=output_dir, overwrite=args.overwrite)

    print("========== Laws Ingestion Complete ==========")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Written: {result['written']}")
    print(f"Skipped empty records: {result['skipped']}")
    print("")
    print("Next steps:")
    print("1. Set KNOWLEDGE_SOURCES=gov_policy,laws in .env")
    print("2. Run python build_vectorstore.py")
    print("3. Test POST /api/retrieval/search with mode=hybrid")


if __name__ == "__main__":
    main()
