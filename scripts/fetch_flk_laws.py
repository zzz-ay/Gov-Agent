import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

import httpx


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "knowledge_sources" / "laws" / "flk"
DEFAULT_DOWNLOAD_DIR = BASE_DIR / "data" / "imports" / "laws_raw" / "flk"

DEFAULT_TERMS = [
    "就业促进法",
    "劳动法",
    "劳动合同法",
    "社会保险法",
    "个人信息保护法",
    "数据安全法",
    "网络安全法",
    "行政许可法",
    "行政处罚法",
    "民法典",
    "居住证暂行条例",
    "住房公积金管理条例",
]


def strip_html(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value[:120].strip("._") or "law"


def flatten_content(node: Any, depth: int = 0) -> List[str]:
    lines: List[str] = []

    if not node:
        return lines

    if isinstance(node, str):
        text = strip_html(node)
        if text:
            lines.append(text)
        return lines

    if isinstance(node, list):
        for item in node:
            lines.extend(flatten_content(item, depth=depth))
        return lines

    if not isinstance(node, dict):
        return lines

    title = strip_html(str(node.get("title", "") or ""))
    content = strip_html(str(node.get("content", "") or node.get("text", "") or ""))

    if title:
        prefix = "#" * min(depth + 1, 6)
        lines.append(f"{prefix} {title}")

    if content and content != title:
        lines.append(content)

    for child in node.get("children", []) or []:
        lines.extend(flatten_content(child, depth=depth + 1))

    return lines


class FlkClient:
    def __init__(self, timeout: int = 20):
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Content-Type": "application/json;charset=utf-8",
            },
            verify=True,
        )
        self.timeout = timeout

    def search(self, term: str, page_size: int = 20) -> List[Dict[str, Any]]:
        payload = {
            "searchRange": 1,
            "searchType": 2,
            "searchContent": term,
            "pageNum": 1,
            "pageSize": page_size,
        }
        response = self.client.post(
            "https://flk.npc.gov.cn/law-search/search/list",
            json=payload,
        )
        response.raise_for_status()
        data = json.loads(response.content.decode("utf-8"))
        return data.get("rows", []) or []

    def detail(self, bbbs: str) -> Dict[str, Any]:
        response = self.client.get(
            "https://flk.npc.gov.cn/law-search/search/flfgDetails",
            params={"bbbs": bbbs},
        )
        response.raise_for_status()
        data = json.loads(response.content.decode("utf-8"))
        if data.get("code") != 200:
            raise RuntimeError(f"FLK detail failed: {data}")
        return data.get("data", {}) or {}

    def download_docx(self, bbbs: str, output_path: Path) -> bool:
        response = self.client.get(
            "https://flk.npc.gov.cn/law-search/download/pc",
            params={"bbbs": bbbs, "format": "docx"},
        )
        response.raise_for_status()
        payload = json.loads(response.content.decode("utf-8"))
        download_url = (payload.get("data") or {}).get("url")
        if payload.get("code") != 200 or not download_url:
            return False

        file_response = self.client.get(download_url)
        file_response.raise_for_status()
        if not file_response.content.startswith(b"PK"):
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(file_response.content)
        return True


def extract_docx_text(path: Path) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: List[str] = []

    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.parse(io.BytesIO(document_xml)).getroot()
    for paragraph in root.findall(".//w:p", namespace):
        parts: List[str] = []
        for node in paragraph.iter():
            if node.tag == f"{{{namespace['w']}}}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{{{namespace['w']}}}tab":
                parts.append("\t")
            elif node.tag == f"{{{namespace['w']}}}br":
                parts.append("\n")

        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)

    return "\n".join(paragraphs).strip()


def choose_result(term: str, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    clean_term = strip_html(term)
    exact = []
    contains = []

    for row in rows:
        title = strip_html(row.get("title", ""))
        if title == clean_term:
            exact.append(row)
        elif clean_term in title or title in clean_term:
            contains.append(row)

    return (exact or contains or rows or [None])[0]


def render_law(detail: Dict[str, Any], full_text: str = "") -> str:
    title = detail.get("title", "")
    content = full_text.strip()
    if not content:
        content_lines = flatten_content(detail.get("content"))
        content = "\n".join(content_lines).strip() or title

    return "\n".join(
        [
            f"Title: {title}",
            f"Source Org: {detail.get('zdjgName', '')}",
            f"Publish Date: {detail.get('gbrq', '')}",
            "Source URL: https://flk.npc.gov.cn/",
            "Region: national",
            f"Topic: {detail.get('flxz', 'law')}",
            f"Document Type: {detail.get('flxz', 'law')}",
            "Authority Level: law",
            f"FLK ID: {detail.get('bbbs', '')}",
            "",
            "Content:",
            content,
            "",
        ]
    )


def fetch_terms(
    terms: Iterable[str],
    output_dir: Path,
    download_dir: Path,
    sleep_seconds: float = 0.3,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    client = FlkClient()

    written = 0
    skipped = 0
    downloaded = 0

    for term in terms:
        term = term.strip()
        if not term:
            continue

        rows = client.search(term)
        row = choose_result(term, rows)
        if not row:
            print(f"[skip] no result: {term}")
            skipped += 1
            continue

        detail = client.detail(row["bbbs"])
        title = detail.get("title") or strip_html(row.get("title", "")) or term
        full_text = ""

        docx_path = download_dir / f"{safe_filename(title)}.docx"
        if client.download_docx(detail.get("bbbs", row["bbbs"]), docx_path):
            downloaded += 1
            full_text = extract_docx_text(docx_path)

        output_path = output_dir / f"{safe_filename(title)}.txt"
        output_path.write_text(render_law(detail, full_text=full_text), encoding="utf-8")
        print(f"[written] {title} -> {output_path}")
        written += 1
        time.sleep(sleep_seconds)

    return {"written": written, "skipped": skipped, "downloaded": downloaded}


def load_terms_file(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main():
    parser = argparse.ArgumentParser(description="Fetch laws/regulations from the National Laws and Regulations Database.")
    parser.add_argument("--terms-file", default="", help="Optional newline-delimited law title list.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    terms = load_terms_file(Path(args.terms_file)) if args.terms_file else DEFAULT_TERMS
    result = fetch_terms(
        terms=terms,
        output_dir=Path(args.output).resolve(),
        download_dir=Path(args.download_dir).resolve(),
        sleep_seconds=args.sleep,
    )

    print("========== FLK Laws Fetch Complete ==========")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Downloads: {Path(args.download_dir).resolve()}")
    print(f"Written: {result['written']}")
    print(f"Downloaded: {result['downloaded']}")
    print(f"Skipped: {result['skipped']}")


if __name__ == "__main__":
    main()
