#!/usr/bin/env python3
"""Stage RealSR validity jobs and optionally dispatch Codex judge subagents.

This script implements the VALIDITY_JUDGE.md workflow:
1. Copy each task's data, metadata, validity rubrics, and submission into a
   self-contained stage directory.
2. Write one prompt per chunk.
3. Optionally execute those prompts with `codex exec` via subprocess.run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import shutil
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CHUNK_SIZE = 3
DEFAULT_MAX_WORKERS = 1

CONSTANT_DISCIPLINE_RUBRIC = (
    "constant discipline / no cap evasion: the submission uses a compact "
    "symbolic form rather than hiding fitted or data-derived degrees of freedom. "
    "Using judgment, treat metadata caps as reference-baseline caps and allow "
    "up to +3 slack for fitted/data-derived scalar constants and Type II local "
    "fit parameters. Count apparent fitted/data-derived coefficients whether "
    "they appear in LAW_CONSTANTS, scalar OTHER_CONSTANTS, LOCAL_FITTABLE, or "
    "as hard-coded literal coefficients in predict/fit; do not mechanically "
    "count trivial structural constants such as 0, 1, -1, 2, 0.5, log bases, "
    "unit conversions, or numerical epsilons unless they are clearly fit from "
    "the data. Mark this rubric N if OTHER_CONSTANTS or module-level literals "
    "encode training/test aggregates, lookup tables, profiles, large literal "
    "arrays, per-age/per-group templates, or any other memorization/cap-evasion "
    "device instead of a scientific law."
)

DISPATCH_HELP = """\
Dispatch the judge subagents:
  If --dispatch codex is used, evaluate_validity.py calls `codex exec` once per
  generated prompt chunk. Each prompt contains the judging instructions plus the
  staged validity_rubrics.json path for every task in that chunk.

  Recommended chunking:
    --chunk-size 3

  Example:
    python harness/evaluate_validity.py \\
      --tasks-dir tasks \\
      --submissions submissions \\
      --stage-root validity_stage \\
      --output-root validity_out \\
      --chunk-size 3 \\
      --dispatch codex \\
      --max-workers 4

  The Codex command executed for each chunk is equivalent to:
    codex exec \\
      -C /path/to/SciLaws-Bench \\
      -s workspace-write \\
      --json \\
      -o <OUTPUT_DIR>/agent_logs/chunk_001.last.txt \\
      - < <STAGE_DIR>/prompts/chunk_001.md \\
      > <OUTPUT_DIR>/agent_logs/chunk_001.jsonl 2>&1

  Results:
    Each subagent writes <OUTPUT_DIR>/<stage_id>.json.
    evaluate_validity.py writes <OUTPUT_DIR>/validity_summary.csv and
    <OUTPUT_DIR>/validity_summary.json after dispatch.
    Dispatch logs are written under <OUTPUT_DIR>/agent_logs/.
    Dispatch process status is written to <OUTPUT_DIR>/dispatch_results.json.
"""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _as_abs(path: Path) -> str:
    return str(path.resolve())


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_under_repo(path: Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else repo_root / path


def _task_index(tasks_dir: Path) -> dict[str, dict[str, Path | str]]:
    index: dict[str, dict[str, Path | str]] = {}
    for metadata in sorted(tasks_dir.glob("*/*/metadata.yaml")):
        task_dir = metadata.parent
        task = task_dir.name
        ttype = task_dir.parent.name
        index[task] = {
            "task": task,
            "type": ttype,
            "task_dir": task_dir,
            "metadata": metadata,
            "data": task_dir / "data",
            "rubric": task_dir / "eval" / "validity_rubrics.json",
        }
    return index


def _submission_method(submission: Path, submissions_dir: Path) -> str | None:
    rel = submission.relative_to(submissions_dir)
    if len(rel.parts) == 1:
        return None
    return "__".join(rel.parts[:-1])


def _stage_id(method: str | None, task: str) -> str:
    return f"{method}__{task}" if method else task


def _find_rubric(task_info: dict[str, Path | str], tasks_dir: Path, scoring_dir: Path | None) -> Path:
    rubric = Path(task_info["rubric"])
    if rubric.exists():
        return rubric
    if scoring_dir is not None:
        alt = scoring_dir / str(task_info["type"]) / str(task_info["task"]) / "validity_rubrics.json"
        if alt.exists():
            return alt
    return rubric


def _write_staged_rubric(src: Path, dst: Path) -> int:
    payload = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rubrics = list(payload)
        if CONSTANT_DISCIPLINE_RUBRIC not in rubrics:
            rubrics.append(CONSTANT_DISCIPLINE_RUBRIC)
        dst.write_text(json.dumps(rubrics, indent=2) + "\n", encoding="utf-8")
        return len(rubrics)

    if isinstance(payload, dict):
        rubrics = payload.get("validity_rubrics")
        if not isinstance(rubrics, list):
            raise ValueError(f"{src} must contain a list key named validity_rubrics")
        staged_payload = dict(payload)
        staged_rubrics = list(rubrics)
        if CONSTANT_DISCIPLINE_RUBRIC not in staged_rubrics:
            staged_rubrics.append(CONSTANT_DISCIPLINE_RUBRIC)
        staged_payload["validity_rubrics"] = staged_rubrics
        dst.write_text(json.dumps(staged_payload, indent=2) + "\n", encoding="utf-8")
        return len(staged_rubrics)

    raise ValueError(f"{src} must be a JSON list or object")


def _iter_submissions(submissions_dir: Path):
    for path in sorted(submissions_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _build_prompt(stage_dir: Path, output_dir: Path, stage_ids: list[str]) -> str:
    task_lines = []
    for stage_id in stage_ids:
        task_dir = stage_dir / stage_id
        task_lines.append(
            f"- {stage_id}\n"
            f"  task_dir: {_as_abs(task_dir)}\n"
            f"  validity_rubrics: {_as_abs(task_dir / 'validity_rubrics.json')}"
        )
    tasks_block = "\n".join(task_lines)
    return f"""You are a VALIDITY JUDGE for RealSR v3.

Read only the staged task directories listed below. Write only JSON result files
under the output directory. Do not edit existing repository files.

OUTPUT_DIR: {_as_abs(output_dir)}

Staged tasks:
{tasks_block}

For EACH staged task:
1. Read task_dir/validity_rubrics.json. It may be either a list or an object
   with key "validity_rubrics"; score that rubric list.
2. Read task_dir/metadata.yaml for target.name, inputs in order, and type.
3. Read and execute task_dir/submission.py in an isolated namespace. Extract
   predict, USED_INPUTS, LAW_CONSTANTS, and for Type II also fit and
   LOCAL_FITTABLE. If import or contract execution fails, write a result JSON
   with validity_score=null and an "error" string, then continue.
4. Build X from data columns matching USED_INPUTS. If a USED_INPUT name does not
   match a data column, map positionally against metadata inputs.
   - Type I: evaluate data/test.csv.
   - Type II: group data/test_fit.csv by group_id, call fit(X_fit, y_fit,
     **LAW_CONSTANTS) for representative clusters, then call predict on
     data/test_test.csv with **LAW_CONSTANTS and fitted local params.
5. Build deterministic domain grids over each used input's data min/max range,
   holding other inputs at their median. For Type II, use one representative
   fitted local parameter set for grid checks.
6. Score each rubric as Y or N. Prefer computed evidence:
   - behavioral: finite-difference monotonicity, min/max range checks, sign,
     non-negativity, bounds, limits, mixed differences for separability.
   - structural: numeric probes where possible, otherwise source inspection.
     A functionally equivalent term counts; coefficient accuracy belongs to
     numeric_score, not validity_score.
   - constant-discipline / no-cap-evasion: use source inspection, metadata
     caps, and final submitted code. This is a judgment rubric, not a mechanical
     literal-count rule. Mark it N for fitted/data-derived lookup tables,
     profiles, train/test aggregates, large literal arrays, or obvious attempts
     to move degrees of freedom outside the stated caps.

Write exactly one JSON per task to OUTPUT_DIR/<stage_id>.json:
{{
  "task": "<stage_id>",
  "n_satisfied": <int or null>,
  "n_total": <int or null>,
  "validity_score": <number or null>,
  "error": <string or null>,
  "rubrics": [
    {{"i": 1, "verdict": "Y|N", "kind": "behavioral|structural",
      "evidence": "one-line computed or source evidence"}}
  ]
}}

After all tasks, reply one line per task:
<stage_id> validity=<score>
"""


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def stage_validity_jobs(args: argparse.Namespace) -> dict[str, Any]:
    tasks_dir = args.tasks_dir.resolve()
    submissions_dir = args.submissions.resolve()
    scoring_dir = args.scoring_dir.resolve() if args.scoring_dir else None

    run_id = args.run_id or _utc_stamp()
    stage_dir = args.stage_dir.resolve() if args.stage_dir else (args.stage_root / run_id).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (args.output_root / run_id).resolve()

    for path in (stage_dir, output_dir):
        if path.exists():
            if not args.overwrite:
                raise SystemExit(f"{path} already exists; pass --overwrite or choose a new --run-id")
            shutil.rmtree(path)
    stage_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    prompts_dir = stage_dir / "prompts"
    prompts_dir.mkdir()

    tasks = _task_index(tasks_dir)
    staged: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for submission in _iter_submissions(submissions_dir):
        task = submission.stem
        task_info = tasks.get(task)
        method = args.method_name or _submission_method(submission, submissions_dir)
        stage_id = _stage_id(method, task)
        if stage_id in seen_ids:
            skipped.append({
                "stage_id": stage_id,
                "task": task,
                "submission": _as_abs(submission),
                "skip": "duplicate_stage_id",
            })
            continue
        seen_ids.add(stage_id)

        if task_info is None:
            skipped.append({
                "stage_id": stage_id,
                "task": task,
                "submission": _as_abs(submission),
                "skip": "unknown_task",
            })
            continue

        rubric = _find_rubric(task_info, tasks_dir, scoring_dir)
        data_dir = Path(task_info["data"])
        if not rubric.exists():
            skipped.append({
                "stage_id": stage_id,
                "type": task_info["type"],
                "task": task,
                "submission": _as_abs(submission),
                "skip": "missing_validity_rubrics",
            })
            continue
        if not data_dir.exists():
            skipped.append({
                "stage_id": stage_id,
                "type": task_info["type"],
                "task": task,
                "submission": _as_abs(submission),
                "skip": "missing_data_dir",
            })
            continue

        dst = stage_dir / stage_id
        dst.mkdir()
        shutil.copy2(Path(task_info["metadata"]), dst / "metadata.yaml")
        n_staged_rubrics = _write_staged_rubric(rubric, dst / "validity_rubrics.json")
        shutil.copy2(submission, dst / "submission.py")
        shutil.copytree(data_dir, dst / "data")

        staged.append({
            "stage_id": stage_id,
            "method": method,
            "type": task_info["type"],
            "task": task,
            "task_dir": _as_abs(Path(task_info["task_dir"])),
            "submission": _as_abs(submission),
            "staged_task_dir": _as_abs(dst),
            "validity_rubrics": _as_abs(dst / "validity_rubrics.json"),
            "n_staged_rubrics": n_staged_rubrics,
        })

    chunk_size = max(1, int(args.chunk_size))
    chunks = []
    for idx, stage_ids in enumerate(_chunked([x["stage_id"] for x in staged], chunk_size), start=1):
        prompt_path = prompts_dir / f"chunk_{idx:03d}.md"
        prompt_path.write_text(_build_prompt(stage_dir, output_dir, stage_ids), encoding="utf-8")
        chunks.append({
            "chunk": idx,
            "prompt": _as_abs(prompt_path),
            "stage_ids": stage_ids,
        })

    manifest = {
        "run_id": run_id,
        "tasks_dir": _as_abs(tasks_dir),
        "submissions_dir": _as_abs(submissions_dir),
        "stage_dir": _as_abs(stage_dir),
        "output_dir": _as_abs(output_dir),
        "prompts_dir": _as_abs(prompts_dir),
        "chunk_size": chunk_size,
        "staged": staged,
        "skipped": skipped,
        "chunks": chunks,
    }
    manifest_path = stage_dir / "manifest.json"
    manifest["manifest"] = _as_abs(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _codex_command(args: argparse.Namespace, prompt_path: Path, last_message_path: Path) -> list[str]:
    repo_root = args.repo_root.resolve()
    cmd = [
        args.codex_bin,
        "exec",
        "-C",
        _as_abs(repo_root),
        "-s",
        args.codex_sandbox,
        "--json",
        "-o",
        _as_abs(last_message_path),
    ]
    if args.codex_approval:
        cmd.extend(["--ask-for-approval", args.codex_approval])
    if args.codex_model:
        cmd.extend(["-m", args.codex_model])
    add_dirs = [Path(p).resolve() for p in (args.codex_add_dir or [])]
    for candidate in (prompt_path.resolve().parent.parent, last_message_path.resolve().parent.parent):
        if not _is_relative_to(candidate, repo_root):
            add_dirs.append(candidate)
    seen_add_dirs: set[str] = set()
    for add_dir in add_dirs:
        add_dir_s = _as_abs(add_dir)
        if add_dir_s in seen_add_dirs:
            continue
        seen_add_dirs.add(add_dir_s)
        cmd.extend(["--add-dir", add_dir_s])
    for extra in args.codex_arg or []:
        cmd.append(extra)
    cmd.append("-")
    return cmd


def _run_codex_chunk(chunk: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    prompt_path = Path(chunk["prompt"])
    log_dir = output_dir / "agent_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = f"chunk_{int(chunk['chunk']):03d}"
    jsonl_path = log_dir / f"{chunk_name}.jsonl"
    last_message_path = log_dir / f"{chunk_name}.last.txt"
    cmd = _codex_command(args, prompt_path, last_message_path)
    timeout = args.codex_timeout_seconds if args.codex_timeout_seconds > 0 else None

    started = time.time()
    result: dict[str, Any] = {
        "chunk": chunk["chunk"],
        "prompt": _as_abs(prompt_path),
        "stage_ids": chunk["stage_ids"],
        "command": " ".join(shlex.quote(part) for part in cmd),
        "log": _as_abs(jsonl_path),
        "last_message": _as_abs(last_message_path),
    }
    if args.dry_run_dispatch:
        result.update({"returncode": None, "duration_seconds": 0.0, "status": "dry_run"})
        return result

    try:
        with prompt_path.open("rb") as prompt_fh, jsonl_path.open("wb") as log_fh:
            proc = subprocess.run(
                cmd,
                stdin=prompt_fh,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=args.repo_root,
                timeout=timeout,
                check=False,
            )
        result.update({
            "returncode": proc.returncode,
            "duration_seconds": round(time.time() - started, 3),
            "status": "ok" if proc.returncode == 0 else "failed",
        })
    except subprocess.TimeoutExpired:
        result.update({
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "status": "timeout",
            "error": f"codex exec exceeded {timeout} seconds",
        })
    except Exception as exc:
        result.update({
            "returncode": None,
            "duration_seconds": round(time.time() - started, 3),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })
    return result


def dispatch_codex(manifest: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(manifest["output_dir"])
    chunks = manifest["chunks"]
    max_workers = max(1, int(args.max_workers))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_codex_chunk, chunk, args, output_dir) for chunk in chunks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"dispatch[{int(result['chunk']):03d}] {result['status']} "
                f"returncode={result.get('returncode')} log={result['log']}",
                flush=True,
            )
    results.sort(key=lambda item: item["chunk"])
    result_path = output_dir / "dispatch_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print("DISPATCH_RESULTS:", _as_abs(result_path))
    return results


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _anti_hacking_verdict(result: dict[str, Any]) -> str:
    rubrics = result.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        return ""
    last = rubrics[-1]
    if not isinstance(last, dict):
        return ""
    verdict = str(last.get("verdict") or "").strip().upper()
    return verdict if verdict in {"Y", "N"} else ""


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary_stats(rows: list[dict[str, Any]], method: str | None, task_type: str | None) -> dict[str, Any]:
    selected = [
        row for row in rows
        if (method is None or row["method"] == method)
        and (task_type is None or row["type"] == task_type)
    ]
    scores = [float(row["validity_score"]) for row in selected]
    valid_results = sum(row.get("raw_validity_score") is not None for row in selected)
    mean_score = _mean(scores)
    return {
        "method": "ALL" if method is None else (method or "direct"),
        "type": task_type or "ALL",
        "n": len(selected),
        "scored": len(selected),
        "valid_results": valid_results,
        "anti_hacking_fail": sum(row.get("status") == "anti_hacking_fail" for row in selected),
        "mean_score": mean_score,
        "mean_scored": mean_score,
        "strict_mean": mean_score,
    }


def aggregate_validity_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(manifest["output_dir"])
    rows: list[dict[str, Any]] = []
    for item in manifest["staged"]:
        stage_id = item["stage_id"]
        result_path = output_dir / f"{stage_id}.json"
        row: dict[str, Any] = {
            "method": item.get("method") or "",
            "type": item.get("type") or "",
            "task": item.get("task") or "",
            "stage_id": stage_id,
            "validity_score": 0.0,
            "raw_validity_score": None,
            "anti_hacking_verdict": "",
            "n_satisfied": None,
            "n_total": None,
            "status": "missing_result",
            "error": "",
        }
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                raw_score = _numeric_score(result.get("validity_score"))
                score = raw_score if raw_score is not None else 0.0
                anti_verdict = _anti_hacking_verdict(result)
                if raw_score is not None and anti_verdict == "N":
                    score = 0.0
                row.update({
                    "validity_score": score,
                    "raw_validity_score": raw_score,
                    "anti_hacking_verdict": anti_verdict,
                    "n_satisfied": result.get("n_satisfied"),
                    "n_total": result.get("n_total"),
                    "status": (
                        "anti_hacking_fail"
                        if raw_score is not None and anti_verdict == "N"
                        else ("ok" if raw_score is not None else "validity_null")
                    ),
                    "error": result.get("error") or "",
                })
            except Exception as exc:
                row.update({
                    "status": "invalid_result_json",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        rows.append(row)

    summary_csv = output_dir / "validity_summary.csv"
    fieldnames = [
        "method",
        "type",
        "task",
        "stage_id",
        "validity_score",
        "raw_validity_score",
        "anti_hacking_verdict",
        "n_satisfied",
        "n_total",
        "status",
        "error",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    methods = sorted({row["method"] for row in rows})
    task_types = sorted({row["type"] for row in rows})
    stats = {
        "overall": _summary_stats(rows, None, None),
        "by_method": [_summary_stats(rows, method, None) for method in methods],
        "by_type": [_summary_stats(rows, None, task_type) for task_type in task_types],
        "by_method_type": [
            _summary_stats(rows, method, task_type)
            for method in methods
            for task_type in task_types
            if any(row["method"] == method and row["type"] == task_type for row in rows)
        ],
    }
    summary_json = output_dir / "validity_summary.json"
    summary_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    print("VALIDITY_SUMMARY_CSV:", _as_abs(summary_csv))
    print("VALIDITY_SUMMARY_JSON:", _as_abs(summary_json))
    overall = stats["overall"]
    print(
        "validity overall "
        f"n={overall['n']} valid_results={overall['valid_results']} "
        f"mean_score={overall['mean_score']} anti_hacking_fail={overall['anti_hacking_fail']}"
    )
    return {"rows": rows, "stats": stats, "summary_csv": _as_abs(summary_csv), "summary_json": _as_abs(summary_json)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage RealSR validity jobs and optionally dispatch Codex judge subagents.",
        epilog=DISPATCH_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=_repo_root(),
                        help="Repository root used as the Codex working directory.")
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--submissions", type=Path, default=None,
                        help="Directory with <task>.py files, or method subdirs containing <task>.py files.")
    parser.add_argument("--scoring-dir", type=Path, default=Path("scoring"),
                        help="Optional older scoring tree fallback for validity_rubrics.json.")
    parser.add_argument("--stage-root", type=Path, default=Path("validity_stage"))
    parser.add_argument("--output-root", type=Path, default=Path("validity_out"))
    parser.add_argument("--stage-dir", type=Path, default=None,
                        help="Exact stage dir to create; overrides --stage-root/--run-id.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Exact output dir to create; overrides --output-root/--run-id.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--method-name", default=None,
                        help="Prefix direct submissions as <method>__<task>.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dispatch", choices=("none", "codex"), default="none",
                        help="Optionally call a codeagent runner after staging.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip staging/dispatch; read --stage-dir/manifest.json and aggregate existing result JSON files.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Concurrent Codex exec processes when --dispatch codex is used.")
    parser.add_argument("--dry-run-dispatch", action="store_true",
                        help="Write dispatch commands to dispatch_results.json without running them.")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-sandbox", default="workspace-write")
    parser.add_argument("--codex-approval", default=None,
                        help="Optional approval policy if supported by this Codex CLI.")
    parser.add_argument("--codex-timeout-seconds", type=int, default=0,
                        help="Per-chunk Codex timeout. 0 means no timeout.")
    parser.add_argument("--codex-add-dir", action="append", default=[],
                        help="Additional writable/readable dir passed to `codex exec --add-dir`.")
    parser.add_argument("--codex-arg", action="append", default=[],
                        help="Extra raw argument passed to `codex exec`; repeat as needed.")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.tasks_dir = _resolve_under_repo(args.tasks_dir, args.repo_root)
    args.submissions = _resolve_under_repo(args.submissions, args.repo_root)
    args.scoring_dir = _resolve_under_repo(args.scoring_dir, args.repo_root)
    args.stage_root = _resolve_under_repo(args.stage_root, args.repo_root)
    args.output_root = _resolve_under_repo(args.output_root, args.repo_root)
    args.stage_dir = _resolve_under_repo(args.stage_dir, args.repo_root)
    args.output_dir = _resolve_under_repo(args.output_dir, args.repo_root)

    if args.aggregate_only:
        if args.stage_dir is None:
            raise SystemExit("--aggregate-only requires --stage-dir")
        manifest_path = args.stage_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        aggregate_validity_outputs(manifest)
        return 0

    if args.submissions is None:
        raise SystemExit("--submissions is required unless --aggregate-only is used")

    manifest = stage_validity_jobs(args)
    print("STAGE_DIR:", manifest["stage_dir"])
    print("OUTPUT_DIR:", manifest["output_dir"])
    print("PROMPTS_DIR:", manifest["prompts_dir"])
    print("MANIFEST:", manifest["manifest"])
    print("staged:", len(manifest["staged"]))
    print("skipped:", len(manifest["skipped"]))
    print("chunks:", len(manifest["chunks"]))
    for chunk in manifest["chunks"]:
        print(f"prompt[{chunk['chunk']:03d}]: {chunk['prompt']}")
    if args.dispatch == "codex":
        dispatch_codex(manifest, args)
        aggregate_validity_outputs(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
