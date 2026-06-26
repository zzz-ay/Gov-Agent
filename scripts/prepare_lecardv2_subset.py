import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LECARD_DIR = BASE_DIR / "data" / "imports" / "legal_cases_raw" / "lecardv2"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "knowledge_sources" / "legal_cases"
DEFAULT_EVAL_OUTPUT = BASE_DIR / "data" / "evaluation" / "lecardv2_retrieval_eval.jsonl"


def safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value[:120].strip("._") or "case"


def load_queries(query_path: Path) -> Dict[str, Dict]:
    queries = {}
    with query_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            item = json.loads(line)
            item["_ordinal_qid"] = str(idx)
            queries[str(item.get("id"))] = item
            queries[str(idx)] = item
    return queries


def iter_qrels(qrels_path: Path) -> Iterable[Tuple[str, str, int]]:
    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, pid, label = parts[:4]
            yield qid, pid, int(label)


def load_candidate(candidate_dir: Path, pid: str) -> Dict:
    path = candidate_dir / f"{pid}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_candidate_dir(lecard_dir: Path, candidate_dir: Path) -> None:
    if candidate_dir.exists():
        return

    archive_path = lecard_dir / "candidate.tar"
    if not archive_path.exists():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")

    extract_root = candidate_dir.parent
    extract_root.mkdir(parents=True, exist_ok=True)
    resolved_root = extract_root.resolve()

    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (extract_root / member.name).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe tar member path: {member.name}") from exc
        archive.extractall(extract_root)

    if not candidate_dir.exists():
        raise FileNotFoundError(f"Expected extracted candidate directory: {candidate_dir}")


def extract_court(text: str) -> str:
    match = re.search(r"([\u4e00-\u9fff]{2,30}人民法院)", text or "")
    return match.group(1) if match else ""


def case_title(item: Dict) -> str:
    pid = str(item.get("pid", ""))
    charge = "、".join(str(x) for x in item.get("charge", []) if str(x).strip())
    if charge:
        return f"LeCaRDv2案例{pid}_{charge}"
    return f"LeCaRDv2案例{pid}"


def charge_key(item: Dict) -> str:
    charges = [str(x).strip() for x in item.get("charge", []) if str(x).strip()]
    return charges[0] if charges else "unknown"


def render_case(item: Dict) -> str:
    pid = str(item.get("pid", ""))
    qw = str(item.get("qw", "") or "")
    fact = str(item.get("fact", "") or "")
    reason = str(item.get("reason", "") or "")
    result = str(item.get("result", "") or "")
    charge = "、".join(str(x) for x in item.get("charge", []) if str(x).strip())
    article = "、".join(str(x) for x in item.get("article", []) if str(x).strip())
    court = extract_court(qw)

    content_parts = [
        qw,
        f"案件事实：{fact}" if fact else "",
        f"裁判理由：{reason}" if reason else "",
        f"裁判结果：{result}" if result else "",
        f"罪名：{charge}" if charge else "",
        f"相关法条：{article}" if article else "",
    ]
    content = "\n".join(part for part in content_parts if part).strip()

    return "\n".join(
        [
            f"Title: {case_title(item)}",
            f"Source Org: {court}",
            "Publish Date:",
            "Source URL:",
            "Region: national",
            f"Topic: {charge or 'legal_case'}",
            "Document Type: legal_case",
            "Authority Level: case",
            f"Case ID: {pid}",
            "Case Type: LeCaRDv2",
            "",
            "Content:",
            content,
            "",
        ]
    )


def choose_pids(
    qrels_path: Path,
    candidate_dir: Path,
    max_cases: int,
) -> Tuple[List[str], Dict[str, List[str]]]:
    selected = []
    selected_set = set()
    seen_pids = set()
    seen_charges = set()
    fill_candidates: List[str] = []
    qid_to_pids: Dict[str, List[str]] = {}

    for qid, pid, label in iter_qrels(qrels_path):
        if label < 2:
            continue
        qid_to_pids.setdefault(qid, []).append(pid)

        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        fill_candidates.append(pid)

        try:
            item = load_candidate(candidate_dir, pid)
            key = charge_key(item)
        except FileNotFoundError:
            key = "missing"

        if key not in seen_charges and len(selected) < max_cases:
            selected.append(pid)
            selected_set.add(pid)
            seen_charges.add(key)

        if len(selected) >= max_cases:
            break

    if len(selected) < max_cases:
        for pid in fill_candidates:
            if pid in selected_set:
                continue
            selected.append(pid)
            selected_set.add(pid)
            if len(selected) >= max_cases:
                break

    return selected, qid_to_pids


def build_eval_cases(
    queries: Dict[str, Dict],
    qid_to_pids: Dict[str, List[str]],
    selected_pids: set,
    pid_to_filename: Dict[str, str],
    max_eval_cases: int,
) -> List[Dict]:
    eval_cases = []

    for qid, pids in qid_to_pids.items():
        query = queries.get(qid)
        if not query:
            continue

        gold_pid = next((pid for pid in pids if pid in selected_pids), None)
        if not gold_pid:
            continue

        question = str(query.get("fact") or query.get("query") or "").strip()
        if not question:
            continue

        eval_cases.append(
            {
                "question": question,
                "gold_source_file": pid_to_filename[gold_pid],
                "expected_keywords": [],
                "metadata": {
                    "dataset": "LeCaRDv2",
                    "qid": qid,
                    "gold_pid": gold_pid,
                },
            }
        )

        if len(eval_cases) >= max_eval_cases:
            break

    return eval_cases


def prepare_subset(
    lecard_dir: Path,
    output_dir: Path,
    eval_output: Path,
    max_cases: int,
    max_eval_cases: int,
    overwrite: bool,
    clean_output: bool,
) -> Dict[str, int]:
    repo_dir = lecard_dir / "repo"
    candidate_dir = lecard_dir / "candidate_extracted" / "candidate_55192"
    query_path = repo_dir / "query" / "query.json"
    qrels_path = repo_dir / "label" / "relevence_gold.trec"

    ensure_candidate_dir(lecard_dir, candidate_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output.parent.mkdir(parents=True, exist_ok=True)

    removed_existing = 0
    if clean_output:
        for path in output_dir.glob("lecardv2_*.txt"):
            path.unlink()
            removed_existing += 1

    queries = load_queries(query_path)
    selected_pids, qid_to_pids = choose_pids(
        qrels_path,
        candidate_dir=candidate_dir,
        max_cases=max_cases,
    )
    selected_set = set(selected_pids)
    pid_to_filename: Dict[str, str] = {}

    written = 0
    skipped = 0

    for pid in selected_pids:
        try:
            item = load_candidate(candidate_dir, pid)
        except FileNotFoundError:
            skipped += 1
            continue

        filename = safe_filename(f"lecardv2_{pid}_{case_title(item)}") + ".txt"
        output_path = output_dir / filename
        if output_path.exists() and not overwrite:
            pid_to_filename[pid] = output_path.name
            continue

        output_path.write_text(render_case(item), encoding="utf-8")
        pid_to_filename[pid] = output_path.name
        written += 1

    eval_cases = build_eval_cases(
        queries=queries,
        qid_to_pids=qid_to_pids,
        selected_pids=selected_set,
        pid_to_filename=pid_to_filename,
        max_eval_cases=max_eval_cases,
    )

    with eval_output.open("w", encoding="utf-8") as f:
        for item in eval_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return {
        "selected": len(selected_pids),
        "written": written,
        "skipped": skipped,
        "eval_cases": len(eval_cases),
        "removed_existing": removed_existing,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a smaller LeCaRDv2 legal-case subset for GOV-RAG."
    )
    parser.add_argument("--lecard-dir", default=str(DEFAULT_LECARD_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL_OUTPUT))
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--max-eval-cases", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing lecardv2_*.txt files from the output directory before writing the demo subset.",
    )
    args = parser.parse_args()

    result = prepare_subset(
        lecard_dir=Path(args.lecard_dir).resolve(),
        output_dir=Path(args.output).resolve(),
        eval_output=Path(args.eval_output).resolve(),
        max_cases=args.max_cases,
        max_eval_cases=args.max_eval_cases,
        overwrite=args.overwrite,
        clean_output=args.clean_output,
    )

    print("========== LeCaRDv2 Subset Prepared ==========")
    print(f"LeCaRDv2 dir: {Path(args.lecard_dir).resolve()}")
    print(f"Output: {Path(args.output).resolve()}")
    print(f"Eval output: {Path(args.eval_output).resolve()}")
    print(f"Selected candidate pids: {result['selected']}")
    print(f"Removed existing LeCaRDv2 files: {result['removed_existing']}")
    print(f"Written files: {result['written']}")
    print(f"Skipped missing files: {result['skipped']}")
    print(f"Eval cases: {result['eval_cases']}")
    print("")
    print("Next steps:")
    print("1. Set KNOWLEDGE_SOURCES=gov_policy,legal_cases in .env")
    print("2. Run python build_vectorstore.py")
    print("3. Run retrieval evaluation with data/evaluation/lecardv2_retrieval_eval.jsonl")


if __name__ == "__main__":
    main()
