#!/usr/bin/env python3
"""Batch runner for the baseline RealSR agent.

Runs `run_baseline.py` across many tasks/models/modes, writes incremental
summaries, and optionally dispatches the parallel structure judge after
simulator submissions finish.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
BASELINE_DIR = Path(__file__).resolve().parent
RUN_BASELINE = BASELINE_DIR / "run_baseline.py"
HARNESS_DIR = REPO / "harness"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(HARNESS_DIR))

from call_llm_api import resolve_model_and_source  # noqa: E402
import evaluate_numeric as _ev  # noqa: E402
import eval_formula as _ef  # noqa: E402


FIELDNAMES = [
    "run_id",
    "model",
    "mode",
    "type",
    "task",
    "task_dir",
    "max_turns",
    "include_test_range",
    "returncode",
    "status",
    "submitted",
    "qualified_submission",
    "numeric_score",
    "raw_numeric_score",
    "numeric_score_std",
    "metric",
    "score_status",
    "contract_ok",
    "rounds",
    "total_tokens",
    "n_experiments",
    "n_python_calls",
    "active_rows",
    "elapsed_seconds",
    "submission_path",
    "trajectory_path",
    "numeric_path",
    "log_path",
    "error",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tasks(tasks_dir: Path, task_types: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task_type in task_types:
        for meta in sorted((tasks_dir / task_type).glob("*/metadata.yaml")):
            task_dir = meta.parent
            out.append({
                "type": task_type,
                "task": task_dir.name,
                "task_dir": task_dir,
            })
    return out


def _read_summary_rows(summary_tsv: Path) -> list[dict[str, Any]]:
    if not summary_tsv.exists():
        return []
    with summary_tsv.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row.get("model", ""),
        row.get("mode", ""),
        row.get("type", ""),
        row.get("task", ""),
    )


def _float_or_blank(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(float(value))
    except Exception:
        return ""


def _json_from_mixed_output(text: str) -> dict[str, Any]:
    """Extract the first JSON object from scorer output.

    Some numeric submissions trigger low-level BLAS/LAPACK warnings that write
    directly to stdout around the JSON emitted by evaluate_numeric.py. The
    scorer result is still valid; parse the JSON prefix and ignore trailing
    warning text.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in scorer output")
    obj, _ = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("scorer output JSON is not an object")
    return obj


def _score_typeii_submission_subprocess(task_dir: Path, submission_path: Path) -> dict[str, Any]:
    """Run Type II numeric scoring in a child process.

    Type II scoring uses signal.alarm() for per-cluster fit timeouts. Python
    only allows signal handlers in the main thread, while run_batch scores
    jobs from ThreadPoolExecutor workers. Running the official scorer in a
    child process keeps that signal logic in the child's main thread.
    """
    cmd = [
        sys.executable,
        str(HARNESS_DIR / "evaluate_numeric.py"),
        "score",
        str(task_dir),
        str(submission_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        result = _json_from_mixed_output(proc.stdout)
    except Exception as exc:
        return {
            "status": "score_error",
            "contract_ok": False,
            "numeric_score": 0.0,
            "raw_numeric_score": None,
            "numeric_score_std": 0.0,
            "error": (
                f"numeric scorer subprocess failed to produce parseable JSON "
                f"(returncode={proc.returncode}): {type(exc).__name__}: {exc}"
            ),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    if proc.returncode != 0:
        result.setdefault("status", "score_error")
        result.setdefault("contract_ok", False)
        result.setdefault("numeric_score", 0.0)
        result.setdefault("raw_numeric_score", None)
        result["subprocess_returncode"] = proc.returncode
        result["stderr_tail"] = proc.stderr[-2000:]
    return result


def _score_real_submission(task_dir: Path, submission_path: Path,
                           include_test_range: bool, numeric_path: Path) -> dict[str, Any]:
    del include_test_range  # scoring uses the official task files, not prompt display options.
    if not submission_path.exists():
        return {
            "numeric_score": "",
            "raw_numeric_score": "",
            "numeric_score_std": "",
            "metric": "",
            "score_status": "missing_submission",
            "contract_ok": "",
        }
    try:
        meta = _ev.load_task(task_dir)
        task_type = meta.get("type", "typeII")
        ref_path = _ev.reference_metrics_path(task_dir)
        if not ref_path.exists():
            result = {
                "status": "no_reference",
                "contract_ok": None,
                "numeric_score": 0.0,
                "raw_numeric_score": None,
                "numeric_score_std": 0.0,
                "numeric_score_per_seed": (
                    [0.0] if task_type == "typeI" else [0.0] * _ev.N_SEEDS
                ),
                "error": f"{ref_path} missing",
            }
        elif task_type == "typeII":
            result = _score_typeii_submission_subprocess(task_dir, submission_path.resolve())
        else:
            ref_metrics = json.loads(ref_path.read_text(encoding="utf-8"))
            data = _ef.load_flat(task_dir)
            mod = _ev.load_submission(submission_path.resolve())
            result = _ev.score_one(
                mod, submission_path.name, data, ref_metrics, meta, {}
            )
        _write_json(numeric_path, result)
        score_obj = result.get("score") if isinstance(result.get("score"), dict) else {}
        return {
            "numeric_score": _float_or_blank(result.get("numeric_score")),
            "raw_numeric_score": _float_or_blank(result.get("raw_numeric_score")),
            "numeric_score_std": _float_or_blank(result.get("numeric_score_std")),
            "metric": result.get("metric") or score_obj.get("metric") or "",
            "score_status": result.get("status") or "",
            "contract_ok": str(bool(result.get("contract_ok"))).lower(),
        }
    except Exception as exc:
        payload = {
            "status": "score_error",
            "contract_ok": False,
            "numeric_score": 0.0,
            "raw_numeric_score": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(numeric_path, payload)
        return {
            "numeric_score": "0.0",
            "raw_numeric_score": "",
            "numeric_score_std": "0.0",
            "metric": "",
            "score_status": "score_error",
            "contract_ok": "false",
        }


def _score_info_from_numeric(numeric_path: Path) -> dict[str, Any] | None:
    result = _read_json(numeric_path)
    if not result:
        return None
    score_obj = result.get("score") if isinstance(result.get("score"), dict) else {}
    return {
        "numeric_score": _float_or_blank(result.get("numeric_score")),
        "raw_numeric_score": _float_or_blank(result.get("raw_numeric_score")),
        "numeric_score_std": _float_or_blank(result.get("numeric_score_std")),
        "metric": result.get("metric") or score_obj.get("metric") or "",
        "score_status": result.get("status") or "",
        "contract_ok": str(bool(result.get("contract_ok"))).lower(),
    }


def _row_from_traj(traj_path: Path) -> dict[str, Any]:
    payload = _read_json(traj_path)
    trial = payload.get("trial") or {}
    experiment_log = trial.get("experiment_log") or []
    active_rows = ""
    if trial.get("train_rows_current") is not None and experiment_log is not None:
        active_rows = str(trial.get("train_rows_current"))
    return {
        "status": trial.get("status") or "",
        "rounds": trial.get("rounds") or "",
        "total_tokens": trial.get("total_tokens") or "",
        "n_experiments": trial.get("n_experiments") or 0,
        "n_python_calls": trial.get("n_python_calls") or 0,
        "active_rows": active_rows,
    }


def _job_paths(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    model = job["model"]
    mode = job["mode"]
    task_type = job["type"]
    task_name = job["task"]

    sub_dir = args.run_dir / mode / "submissions" / model / task_type
    traj_dir = args.run_dir / mode / "trajectories" / model / task_type
    log_dir = args.run_dir / mode / "logs" / model / task_type
    numeric_dir = args.run_dir / mode / "numeric" / model / task_type
    for path in (sub_dir, traj_dir, log_dir, numeric_dir):
        path.mkdir(parents=True, exist_ok=True)

    submission_path = sub_dir / f"{task_name}.py"
    traj_path = traj_dir / f"{task_name}.traj.json"
    log_path = log_dir / f"{task_name}.log"
    numeric_path = numeric_dir / f"{task_name}.json"
    return {
        "sub_dir": sub_dir,
        "traj_dir": traj_dir,
        "log_dir": log_dir,
        "numeric_dir": numeric_dir,
        "submission_path": submission_path,
        "traj_path": traj_path,
        "log_path": log_path,
        "numeric_path": numeric_path,
    }


def _submission_nonempty(submission_path: Path) -> bool:
    return (
        submission_path.exists()
        and submission_path.read_text(encoding="utf-8", errors="replace").strip() != ""
    )


def _is_qualified(mode: str, submitted: bool, score_info: dict[str, Any]) -> bool:
    if mode == "real":
        return (
            submitted
            and score_info.get("contract_ok") == "true"
            and score_info.get("score_status") not in {"contract_fail", "exec_error", "score_error"}
        )
    return submitted


def _existing_row(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    model = job["model"]
    mode = job["mode"]
    task_type = job["type"]
    task_name = job["task"]
    task_dir = Path(job["task_dir"])
    include_range = bool(args.include_range and mode == "real")
    paths = _job_paths(job, args)
    submission_path = paths["submission_path"]
    traj_path = paths["traj_path"]
    log_path = paths["log_path"]
    numeric_path = paths["numeric_path"]

    submitted = _submission_nonempty(submission_path)
    if not submitted:
        return None

    score_info = {
        "numeric_score": "",
        "raw_numeric_score": "",
        "numeric_score_std": "",
        "metric": "",
        "score_status": "",
        "contract_ok": "",
    }
    if mode == "real":
        score_info = (
            _score_info_from_numeric(numeric_path)
            or _score_real_submission(task_dir, submission_path, include_range, numeric_path)
        )

    qualified = _is_qualified(mode, submitted, score_info)
    if not qualified:
        return None

    traj_info = _row_from_traj(traj_path)
    row = {
        "run_id": args.run_id,
        "model": model,
        "mode": mode,
        "type": task_type,
        "task": task_name,
        "task_dir": str(task_dir),
        "max_turns": args.max_turns,
        "include_test_range": str(include_range).lower(),
        "returncode": 0,
        "status": traj_info.get("status") or "completed_existing",
        "submitted": "1",
        "qualified_submission": "1",
        "rounds": traj_info.get("rounds", ""),
        "total_tokens": traj_info.get("total_tokens", ""),
        "n_experiments": traj_info.get("n_experiments", ""),
        "n_python_calls": traj_info.get("n_python_calls", ""),
        "active_rows": traj_info.get("active_rows", ""),
        "elapsed_seconds": "0.000",
        "submission_path": str(submission_path),
        "trajectory_path": str(traj_path),
        "numeric_path": str(numeric_path) if mode == "real" else "",
        "log_path": str(log_path),
        "error": "",
    }
    row.update(score_info)
    return row


def _upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = _row_key(row)
    for pos, old_row in enumerate(rows):
        if _row_key(old_row) == key:
            rows[pos] = row
            return
    rows.append(row)


def _run_one(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model = job["model"]
    mode = job["mode"]
    task_type = job["type"]
    task_name = job["task"]
    task_dir = Path(job["task_dir"])
    include_range = bool(args.include_range and mode == "real")
    paths = _job_paths(job, args)
    sub_dir = paths["sub_dir"]
    traj_dir = paths["traj_dir"]
    submission_path = paths["submission_path"]
    traj_path = paths["traj_path"]
    log_path = paths["log_path"]
    numeric_path = paths["numeric_path"]

    cmd = [
        sys.executable,
        str(RUN_BASELINE),
        str(task_dir),
        model,
        "--max-turns",
        str(args.max_turns),
        "--out",
        str(sub_dir),
        "--traj-out",
        str(traj_dir),
    ]
    if mode == "parallel":
        cmd.append("--simulator")
    elif include_range:
        cmd.append("--include-test-range")
    else:
        cmd.append("--no-include-test-range")

    env = os.environ.copy()
    if args.reasoning_effort:
        env["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
        env["REASONING_EFFORT"] = args.reasoning_effort
    if args.llm_timeout_seconds:
        env["LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout_seconds)

    start = time.time()
    error = ""
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                timeout=args.task_timeout_seconds or None,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            error = f"TimeoutExpired: {exc}"
            log.write("\nBATCH_TIMEOUT: " + error + "\n")
        except Exception as exc:
            returncode = 125
            error = f"{type(exc).__name__}: {exc}"
            log.write("\nBATCH_ERROR: " + error + "\n")

    elapsed = time.time() - start
    submitted = _submission_nonempty(submission_path)
    traj_info = _row_from_traj(traj_path)

    score_info = {
        "numeric_score": "",
        "raw_numeric_score": "",
        "numeric_score_std": "",
        "metric": "",
        "score_status": "",
        "contract_ok": "",
    }
    if mode == "real":
        score_info = _score_real_submission(
            task_dir, submission_path, include_range, numeric_path)

    qualified = _is_qualified(mode, submitted, score_info)

    row = {
        "run_id": args.run_id,
        "model": model,
        "mode": mode,
        "type": task_type,
        "task": task_name,
        "task_dir": str(task_dir),
        "max_turns": args.max_turns,
        "include_test_range": str(include_range).lower(),
        "returncode": returncode,
        "status": traj_info.get("status") or ("ok" if returncode == 0 else "run_error"),
        "submitted": "1" if submitted else "0",
        "qualified_submission": "1" if qualified else "0",
        "rounds": traj_info.get("rounds", ""),
        "total_tokens": traj_info.get("total_tokens", ""),
        "n_experiments": traj_info.get("n_experiments", ""),
        "n_python_calls": traj_info.get("n_python_calls", ""),
        "active_rows": traj_info.get("active_rows", ""),
        "elapsed_seconds": f"{elapsed:.3f}",
        "submission_path": str(submission_path),
        "trajectory_path": str(traj_path),
        "numeric_path": str(numeric_path) if mode == "real" else "",
        "log_path": str(log_path),
        "error": error,
    }
    row.update(score_info)
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["mode"], row["type"])].append(row)
        groups[(row["model"], row["mode"], "ALL")].append(row)
    out: list[dict[str, Any]] = []
    for (model, mode, task_type), items in sorted(groups.items()):
        scores = []
        for row in items:
            try:
                if row.get("numeric_score") != "":
                    scores.append(float(row["numeric_score"]))
            except Exception:
                pass
        status_counts = Counter(row.get("status") or "" for row in items)
        out.append({
            "model": model,
            "mode": mode,
            "type": task_type,
            "n": len(items),
            "submitted": sum(row.get("submitted") == "1" for row in items),
            "qualified": sum(row.get("qualified_submission") == "1" for row in items),
            "completion_rate": (
                sum(row.get("qualified_submission") == "1" for row in items) / len(items)
                if items else None
            ),
            "numeric_scored": len(scores),
            "numeric_mean_scored": (sum(scores) / len(scores) if scores else None),
            "numeric_strict_mean": (
                sum(float(row["numeric_score"]) if row.get("numeric_score") not in ("", None) else 0.0
                    for row in items) / len(items)
                if items else None
            ),
            "status_counts": dict(status_counts),
        })
    return {
        "rows": len(rows),
        "by_model_mode_type": out,
    }


def _write_summary(summary_tsv: Path, summary_json: Path,
                   rows: list[dict[str, Any]]) -> None:
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_tsv.with_suffix(summary_tsv.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    os.replace(tmp, summary_tsv)
    _write_json(summary_json, _summary(rows))


def _score_parallel(args: argparse.Namespace, models: list[str]) -> None:
    results = {}
    for model in models:
        submissions = args.run_dir / "parallel" / "submissions" / model
        if not submissions.exists():
            continue
        stage_dir = args.run_dir / "parallel_stage" / model
        output_dir = args.run_dir / "parallel_out" / model
        cmd = [
            sys.executable,
            str(REPO / "harness" / "evaluate_parallel.py"),
            "--repo-root",
            str(REPO),
            "--tasks-dir",
            "tasks",
            "--submissions",
            str(submissions),
            "--stage-dir",
            str(stage_dir),
            "--output-dir",
            str(output_dir),
            "--method-name",
            model,
            "--chunk-size",
            str(args.parallel_chunk_size),
            "--dispatch",
            "codex",
            "--max-workers",
            str(args.parallel_judge_workers),
            "--overwrite",
        ]
        if args.codex_model:
            cmd.extend(["--codex-model", args.codex_model])
        log_path = args.run_dir / "parallel_out" / f"{model}.evaluate_parallel.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        summary_path = output_dir / "parallel_summary.json"
        results[model] = {
            "returncode": proc.returncode,
            "log_path": str(log_path),
            "summary_json": str(summary_path),
            "summary": _read_json(summary_path),
        }
    _write_json(args.run_dir / "parallel_score_summary.json", results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=("real", "parallel"),
                        default=["real", "parallel"])
    parser.add_argument("--tasks-dir", type=Path, default=REPO / "tasks")
    parser.add_argument("--task-types", nargs="+", choices=("typeI", "typeII"),
                        default=["typeI", "typeII"])
    parser.add_argument("--tasks", nargs="+", default=[],
                        help="Optional task directory names to run.")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-range", action="store_true", default=True)
    parser.add_argument("--no-include-range", dest="include_range", action="store_false")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--llm-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--task-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--score-parallel", action="store_true")
    parser.add_argument("--parallel-judge-workers", type=int, default=4)
    parser.add_argument("--parallel-chunk-size", type=int, default=3)
    parser.add_argument("--codex-model", default="")
    parser.add_argument("--merge-existing-summary", action="store_true",
                        help="Load existing summary.tsv and replace rows for rerun jobs.")
    parser.add_argument("--skip-existing", action="store_true",
                        help=("Reuse existing qualified submissions/numeric results in "
                              "the run dir and run only missing or unqualified jobs."))
    args = parser.parse_args()

    args.run_id = args.run_id or f"batch_{_utc_stamp()}"
    args.run_dir = (args.run_dir or (BASELINE_DIR / "batch_runs" / args.run_id)).resolve()
    args.tasks_dir = args.tasks_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        resolve_model_and_source(model)

    tasks = _tasks(args.tasks_dir, args.task_types)
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [task for task in tasks if task["task"] in wanted]
        missing = sorted(wanted - {task["task"] for task in tasks})
        if missing:
            raise SystemExit(f"unknown tasks for selected --task-types: {missing}")
    if args.limit:
        tasks = tasks[:args.limit]
    jobs = []
    for mode in args.modes:
        for task in tasks:
            for model in args.models:
                jobs.append({**task, "model": model, "mode": mode})

    metadata = {
        "run_id": args.run_id,
        "run_dir": str(args.run_dir),
        "models": args.models,
        "modes": args.modes,
        "task_types": args.task_types,
        "max_turns": args.max_turns,
        "include_range": args.include_range,
        "reasoning_effort": args.reasoning_effort,
        "workers": args.workers,
        "job_order": "mode_task_model",
        "skip_existing": args.skip_existing,
        "n_jobs": len(jobs),
        "created_at_unix": time.time(),
    }
    _write_json(args.run_dir / "run_meta.json", metadata)

    summary_tsv = args.run_dir / "summary.tsv"
    summary_json = args.run_dir / "summary.json"
    rows: list[dict[str, Any]] = (
        _read_summary_rows(summary_tsv) if args.merge_existing_summary else []
    )
    if rows:
        _write_summary(summary_tsv, summary_json, rows)
    lock = threading.Lock()

    run_jobs = jobs
    skipped_existing = 0
    if args.skip_existing:
        run_jobs = []
        for job in jobs:
            existing = _existing_row(job, args)
            if existing is None:
                run_jobs.append(job)
            else:
                _upsert_row(rows, existing)
                skipped_existing += 1
        if skipped_existing:
            _write_summary(summary_tsv, summary_json, rows)

    print(f"RUN_DIR: {args.run_dir}", flush=True)
    print(
        f"JOBS: {len(jobs)}  run={len(run_jobs)}  "
        f"skip_existing={skipped_existing}  workers={args.workers}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_run_one, job, args) for job in run_jobs]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "run_id": args.run_id,
                    "model": "",
                    "mode": "",
                    "type": "",
                    "task": "",
                    "returncode": 126,
                    "status": "batch_error",
                    "submitted": "0",
                    "qualified_submission": "0",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with lock:
                _upsert_row(rows, row)
                _write_summary(summary_tsv, summary_json, rows)
            print(
                f"[{idx}/{len(run_jobs)}] {row.get('model')} {row.get('mode')} "
                f"{row.get('type')}/{row.get('task')} status={row.get('status')} "
                f"submitted={row.get('submitted')} score={row.get('numeric_score')}",
                flush=True,
            )

    if args.score_parallel and "parallel" in args.modes:
        _score_parallel(args, args.models)

    print(f"SUMMARY_TSV: {summary_tsv}", flush=True)
    print(f"SUMMARY_JSON: {summary_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
