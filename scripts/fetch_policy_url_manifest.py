import argparse
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = BASE_DIR / "data" / "imports" / "policy_url_manifest.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw_docs" / "expanded_policy"


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value[:120].strip("._") or "policy"


def extract_title(html: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if not match:
        return fallback
    title = unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    title = re.sub(r"(_|-).*?(中国政府网|政府网|政府|部门|网站).*$", "", title).strip()
    title = re.sub(r"__.*$", "", title).strip()
    return title or fallback


def extract_text(html: str) -> str:
    content_match = re.search(
        r"<div[^>]+id=[\"']UCAP-CONTENT[\"'][^>]*>(.*?)</div>",
        html,
        flags=re.I | re.S,
    )
    if content_match:
        html = content_match.group(1)

    parser = TextExtractor()
    parser.feed(html)
    text = parser.text()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str, timeout: int = 30) -> Dict[str, str]:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    html = response.text
    return {
        "title": extract_title(html, fallback=url),
        "content": extract_text(html),
    }


def render_policy(item: Dict, fetched: Dict[str, str]) -> str:
    title = item.get("title") or fetched["title"]
    return "\n".join(
        [
            f"Title: {title}",
            f"Source Org: {item.get('source_org', '')}",
            f"Publish Date: {item.get('publish_date', '')}",
            f"Source URL: {item.get('url', '')}",
            f"Region: {item.get('region', 'national')}",
            f"Topic: {item.get('topic', item.get('category', 'policy'))}",
            f"Document Type: {item.get('doc_type', 'policy')}",
            f"Authority Level: {item.get('authority_level', 'policy')}",
            "",
            "Content:",
            fetched["content"],
            "",
        ]
    )


def load_manifest(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("items", [])
    if isinstance(payload, list):
        return payload
    raise ValueError("Manifest must be a list or an object with an 'items' list.")


def fetch_manifest(manifest_path: Path, output_dir: Path) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_manifest(manifest_path)
    written = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        url = item.get("url", "").strip()
        if not url:
            continue

        category = safe_filename(item.get("category", "policy"))
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        try:
            fetched = fetch_url(url)
            title = item.get("title") or fetched["title"]
            output_path = category_dir / f"{idx:03d}_{safe_filename(title)}.txt"
            output_path.write_text(render_policy(item, fetched), encoding="utf-8")
            print(f"[written] {url} -> {output_path}")
            written += 1
        except Exception as exc:
            print(f"[failed] {url}: {exc}")
            failed += 1

    return {"written": written, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Fetch real policy webpages from a curated URL manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = fetch_manifest(Path(args.manifest).resolve(), Path(args.output).resolve())
    print("========== Policy URL Manifest Fetch Complete ==========")
    print(f"Manifest: {Path(args.manifest).resolve()}")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Written: {result['written']}")
    print(f"Failed: {result['failed']}")


if __name__ == "__main__":
    main()
