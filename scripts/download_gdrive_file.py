import argparse
import html
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


def extract_file_id(value: str) -> str:
    value = value.strip()
    match = re.search(r"/d/([^/]+)", value)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([^&]+)", value)
    if match:
        return match.group(1)
    return value


def get_confirm_token(response: requests.Response) -> str:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text)
    if match:
        return match.group(1)
    return ""


def get_warning_form_download(response: requests.Response):
    text = response.text
    form_match = re.search(r'<form[^>]+id="download-form"[^>]*>', text)
    if not form_match:
        form_match = re.search(r"<form[^>]+>", text)
    if not form_match:
        return None, {}

    form_tag = form_match.group(0)
    action_match = re.search(r'action="([^"]+)"', form_tag)
    if not action_match:
        return None, {}

    action_url = html.unescape(action_match.group(1))
    action_url = urljoin(response.url, action_url)

    params = {}
    for name, value in re.findall(
        r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
        text,
    ):
        params[html.unescape(name)] = html.unescape(value)

    return action_url, params


def save_response_content(response: requests.Response, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            total += len(chunk)
            if total and total % (100 * 1024 * 1024) < 1024 * 1024:
                print(f"downloaded {total / 1024 / 1024:.1f} MB")
    return total


def download_file(file_id: str, output_path: Path) -> int:
    url = "https://drive.google.com/uc"
    session = requests.Session()

    response = session.get(url, params={"export": "download", "id": file_id}, stream=True)
    response.raise_for_status()

    token = get_confirm_token(response)
    if token:
        response = session.get(
            url,
            params={"export": "download", "confirm": token, "id": file_id},
            stream=True,
        )
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        action_url, params = get_warning_form_download(response)
        if action_url and params:
            response = session.get(action_url, params=params, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

        if "text/html" in content_type.lower():
            preview = response.text[:500]
            raise RuntimeError(
                "Google Drive returned HTML instead of file bytes. "
                "The file may require browser confirmation, account access, or quota handling. "
                f"Preview: {preview}"
            )

    return save_response_content(response, output_path)


def main():
    parser = argparse.ArgumentParser(description="Download a public Google Drive file.")
    parser.add_argument("--file-id", required=True, help="Google Drive file id or shared URL.")
    parser.add_argument("--output", required=True, help="Output file path.")
    args = parser.parse_args()

    file_id = extract_file_id(args.file_id)
    output_path = Path(args.output).resolve()
    total = download_file(file_id, output_path)

    print("========== Google Drive Download Complete ==========")
    print(f"File id: {file_id}")
    print(f"Output: {output_path}")
    print(f"Bytes: {total}")


if __name__ == "__main__":
    main()
