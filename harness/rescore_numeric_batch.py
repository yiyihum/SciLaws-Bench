#!/usr/bin/env python3
"""Rescore an existing baseline_agent batch run with the current numeric scorer.

The script reads a run's summary.tsv, scores every listed submission with
`harness/evaluate_numeric.py`, and writes:

  - summary_rescored.tsv
  - summary_rescored.json
  - numeric_rescored/<model>/<type>/<task>.json

Completion means a qualified submission: the solver produced a submission that
compiled, passed the contract, executed, and received numeric_status == "ok".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(str(value))
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _score_zero(submission: Path, task_type: str, status: str, error: str = "") -> dict[str, Any]:
    return {
        "submission": submission.name,
        "status": status,
        "contract_ok": False,
        "numeric_score": 0.0,
        "raw_numeric_score": None,
        "numeric_score_std": 0.0,
        "numeric_score_per_seed": [0.0] if task_type == "typeI" else [0.0, 0.0, 0.0],
        "raw_numeric_score_per_seed": [None] if task_type == "typeI" else [None, None, None],
        "raw_metric": None,
        "error": error,
    }


def _score_submission(repo: Path, task_dir: Path, submission: Path, task_type: str) -> dict[str, Any]:
    if not submission.exists():
        return _score_zero(submission, task_type, "missing_submission", f"{submission} missing")

    cmd = [
        sys.executable,
        str(repo / "harness" / "evaluate_numeric.py"),
        "score",
        str(task_dir),
        str(submission),
    ]
    proc = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return _score_zero(
            submission,
            task_type,
            "scorer_error",
            proc.stderr.strip() or proc.stdout.strip(),
        )
    try:
        result = json.loads(proc.stdout)
    except Exception as exc:
        return _score_zero(
            submission,
            task_type,
            "invalid_scorer_json",
            f"{type(exc).__name__}: {exc}; stdout={proc.stdout[:500]!r}",
        )
    if _as_float(result.get("numeric_score")) is None:
        result["numeric_score"] = 0.0
        result.setdefault("status", "numeric_null")
    return result


def _summary_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [_as_float(row.get("numeric_score")) or 0.0 for row in rows]
    submitted = sum(str(row.get("submission_exists", "")).lower() == "true" for row in rows)
    qualified = sum(str(row.get("qualified_submission", "")).lower() == "true" for row in rows)
    numeric_status = Counter(row.get("numeric_status", "") for row in rows)
    agent_status = Counter(row.get("status", "") for row in rows)
    return {
        "n": len(rows),
        "submitted": submitted,
        "submission_rate": submitted / len(rows) if rows else None,
        "qualified_submissions": qualified,
        "completion_rate": qualified / len(rows) if rows else None,
        "invalid_submissions": len(rows) - qualified,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "zero_numeric": sum(score == 0.0 for score in scores),
        "numeric_status": dict(sorted(numeric_status.items())),
        "agent_status": dict(sorted(agent_status.items())),
    }


def _group_stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return {name: _summary_stats(items) for name, items in sorted(grouped.items())}


def rescore_batch(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / args.summary
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} missing")

    rows = list(csv.DictReader(summary_path.open(), delimiter="\t"))
    if not rows:
        raise SystemExit(f"{summary_path} has no rows")

    model = args.model or rows[0].get("model")
    if not model:
        raise SystemExit("--model is required when summary.tsv has no model column")

    out_root = run_dir / args.numeric_out / model
    new_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        task_type = row["type"]
        task = row["task"]
        task_dir = repo / "tasks" / task_type / task
        submission = run_dir / "submissions" / model / task_type / f"{task}.py"
        out_dir = out_root / task_type
        out_dir.mkdir(parents=True, exist_ok=True)

        result = _score_submission(repo, task_dir, submission, task_type)
        (out_dir / f"{task}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        score = _as_float(result.get("numeric_score")) or 0.0
        raw_score = result.get("raw_numeric_score")
        numeric_status = str(result.get("status") or ("ok" if result.get("contract_ok") else "unknown"))
        qualified = numeric_status == "ok"

        new_row = dict(row)
        new_row["numeric_score"] = str(score)
        new_row["numeric_status"] = numeric_status
        new_row["raw_numeric_score"] = "" if raw_score is None else str(raw_score)
        new_row["numeric_error"] = str(result.get("error") or result.get("note") or "")
        new_row["submission_exists"] = str(submission.exists()).lower()
        new_row["qualified_submission"] = str(qualified).lower()
        new_rows.append(new_row)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"{run_dir.name} {idx} / {len(rows)}", flush=True)

    out_summary = run_dir / args.output_summary
    fieldnames = list(rows[0].keys())
    for extra in [
        "numeric_status",
        "raw_numeric_score",
        "numeric_error",
        "submission_exists",
        "qualified_submission",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    with out_summary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(new_rows)

    stats = {
        "run": str(run_dir),
        "model": model,
        "summary_source": str(summary_path),
        "summary_rescored": str(out_summary),
        "numeric_rescored_dir": str(out_root),
        "overall": _summary_stats(new_rows),
        "by_type": _group_stats(new_rows, "type"),
    }
    out_json = run_dir / args.output_json
    out_json.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.replace_summary:
        backup = run_dir / args.backup_summary
        if not backup.exists():
            backup.write_bytes(summary_path.read_bytes())
        summary_path.write_bytes(out_summary.read_bytes())
        stats["summary_replaced"] = str(summary_path)
        stats["summary_backup"] = str(backup)
        out_json.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore a baseline batch run with current numeric scoring.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--summary", default="summary.tsv")
    parser.add_argument("--output-summary", default="summary_rescored.tsv")
    parser.add_argument("--output-json", default="summary_rescored.json")
    parser.add_argument("--numeric-out", default="numeric_rescored")
    parser.add_argument("--replace-summary", action="store_true")
    parser.add_argument("--backup-summary", default="summary_pre_numeric_zero.tsv")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    rescore_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
