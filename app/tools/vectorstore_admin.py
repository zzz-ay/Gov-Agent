from pathlib import Path
from typing import Dict, Any
import subprocess
import sys
import re


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DOCS_DIR = BASE_DIR / "data" / "raw_docs"
ADMIN_UPLOADS_DIR = RAW_DOCS_DIR / "admin_uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
BUILD_SCRIPT = BASE_DIR / "build_vectorstore.py"


SUPPORTED_DOC_SUFFIXES = {
    ".txt",
    ".pdf",
    ".md",
    ".docx",
}

SUPPORTED_UPLOAD_SUFFIXES = {
    ".txt",
    ".pdf",
    ".md",
    ".docx",
}


def is_meta_file(path: Path) -> bool:
    """
    判断是否为元数据文件。
    """
    return path.name.endswith(".meta.txt")


def sanitize_filename(filename: str) -> str:
    """
    简单清理上传文件名，避免特殊字符影响保存。
    """
    filename = filename.strip()

    filename = filename.replace("\\", "_").replace("/", "_")
    filename = re.sub(r"[^\w\u4e00-\u9fff.\-()（）]+", "_", filename)

    if not filename:
        filename = "uploaded_policy_file.txt"

    return filename


def ensure_admin_uploads_dir() -> Path:
    """
    确保管理员上传目录存在。
    """
    ADMIN_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return ADMIN_UPLOADS_DIR


def save_uploaded_policy_file(uploaded_file) -> Dict[str, Any]:
    """
    保存 Streamlit 上传的政策文件。

    支持：
    - txt
    - pdf
    - md
    - docx
    - meta.txt

    返回：
    {
        "success": True / False,
        "filename": "...",
        "saved_path": "...",
        "message": "..."
    }
    """
    if uploaded_file is None:
        return {
            "success": False,
            "filename": "",
            "saved_path": "",
            "message": "未检测到上传文件。",
        }

    original_filename = uploaded_file.name
    safe_filename = sanitize_filename(original_filename)

    # .meta.txt 特殊处理
    if safe_filename.endswith(".meta.txt"):
        suffix_ok = True
    else:
        suffix = Path(safe_filename).suffix.lower()
        suffix_ok = suffix in SUPPORTED_UPLOAD_SUFFIXES

    if not suffix_ok:
        return {
            "success": False,
            "filename": safe_filename,
            "saved_path": "",
            "message": "不支持的文件类型。仅支持 txt、pdf、md、docx、meta.txt。",
        }

    upload_dir = ensure_admin_uploads_dir()
    save_path = upload_dir / safe_filename

    # 如果文件已存在，自动追加序号
    if save_path.exists():
        stem = save_path.stem
        suffix = save_path.suffix

        # 兼容 .meta.txt
        if safe_filename.endswith(".meta.txt"):
            stem = safe_filename[:-9]
            suffix = ".meta.txt"

        counter = 1
        while True:
            new_filename = f"{stem}_{counter}{suffix}"
            new_path = upload_dir / new_filename
            if not new_path.exists():
                save_path = new_path
                safe_filename = new_filename
                break
            counter += 1

    try:
        content = uploaded_file.getbuffer()
        with open(save_path, "wb") as f:
            f.write(content)

        return {
            "success": True,
            "filename": safe_filename,
            "saved_path": str(save_path),
            "message": f"文件已保存到 {save_path}",
        }

    except Exception as e:
        return {
            "success": False,
            "filename": safe_filename,
            "saved_path": "",
            "message": f"文件保存失败：{e}",
        }


def count_policy_files() -> Dict[str, Any]:
    """
    统计 data/raw_docs 下的政策文件数量。
    """
    if not RAW_DOCS_DIR.exists():
        return {
            "raw_docs_exists": False,
            "total_files": 0,
            "doc_files": 0,
            "meta_files": 0,
            "pdf_files": 0,
            "txt_files": 0,
            "uploaded_files": 0,
            "files": [],
        }

    all_files = [
        p for p in RAW_DOCS_DIR.rglob("*")
        if p.is_file()
    ]

    meta_files = [
        p for p in all_files
        if is_meta_file(p)
    ]

    content_doc_files = [
        p for p in all_files
        if p.suffix.lower() in SUPPORTED_DOC_SUFFIXES and not is_meta_file(p)
    ]

    pdf_files = [
        p for p in content_doc_files
        if p.suffix.lower() == ".pdf"
    ]

    txt_files = [
        p for p in content_doc_files
        if p.suffix.lower() == ".txt"
    ]

    uploaded_files = []
    if ADMIN_UPLOADS_DIR.exists():
        uploaded_files = [
            p for p in ADMIN_UPLOADS_DIR.rglob("*")
            if p.is_file()
        ]

    display_files = []

    for p in content_doc_files:
        try:
            relative_path = str(p.relative_to(BASE_DIR))
        except ValueError:
            relative_path = str(p)

        display_files.append(relative_path)

    display_files.sort()

    return {
        "raw_docs_exists": True,
        "total_files": len(all_files),
        "doc_files": len(content_doc_files),
        "meta_files": len(meta_files),
        "pdf_files": len(pdf_files),
        "txt_files": len(txt_files),
        "uploaded_files": len(uploaded_files),
        "files": display_files,
    }


def get_vectorstore_status() -> Dict[str, Any]:
    """
    查看向量库状态。
    """
    exists = VECTORSTORE_DIR.exists()

    vector_files = []
    if exists:
        vector_files = [
            p for p in VECTORSTORE_DIR.rglob("*")
            if p.is_file()
        ]

    return {
        "vectorstore_exists": exists,
        "vectorstore_path": str(VECTORSTORE_DIR),
        "file_count": len(vector_files),
    }


def get_policy_library_status() -> Dict[str, Any]:
    """
    汇总政策库状态。
    """
    file_status = count_policy_files()
    vector_status = get_vectorstore_status()

    return {
        "base_dir": str(BASE_DIR),
        "raw_docs_dir": str(RAW_DOCS_DIR),
        "admin_uploads_dir": str(ADMIN_UPLOADS_DIR),
        "vectorstore_dir": str(VECTORSTORE_DIR),
        "build_script": str(BUILD_SCRIPT),
        "build_script_exists": BUILD_SCRIPT.exists(),
        "file_status": file_status,
        "vector_status": vector_status,
    }


def rebuild_vectorstore() -> Dict[str, Any]:
    """
    调用 build_vectorstore.py 重建向量库。

    注意：
    - 该操作会阻塞当前 Streamlit 请求；
    - 如果政策文件较多或 API 响应慢，可能耗时较长；
    - 使用当前 Python 解释器执行，避免环境不一致。
    """
    if not BUILD_SCRIPT.exists():
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"未找到构建脚本: {BUILD_SCRIPT}",
            "command": "",
        }

    command = [
        sys.executable,
        str(BUILD_SCRIPT),
    ]

    try:
        process = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "command": " ".join(command),
        }

    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "command": " ".join(command),
        }


if __name__ == "__main__":
    import json

    status = get_policy_library_status()
    print(json.dumps(status, ensure_ascii=False, indent=2))