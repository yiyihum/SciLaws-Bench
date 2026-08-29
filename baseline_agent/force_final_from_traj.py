#!/usr/bin/env python3
"""Force a final-formula submission from interrupted baseline trajectories.

This is a salvage path for runs that timed out after collecting useful
experiment/history context but never emitted `<final_formula>`. It does not run
more tools. For each selected trajectory it appends one hard-stop user message,
calls the model once (plus an optional parse retry), parses the resulting
`<final_formula>`, writes the submission, and updates the batch summary row.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASELINE_DIR = Path(__file__).resolve().parent
REPO = BASELINE_DIR.parent
HARNESS_DIR = REPO / "harness"

sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(HARNESS_DIR))

import agent_protocol as proto  # noqa: E402


FINAL_PROMPT_TYPEI = """\
Hard stop: do not run `<python>` or `<experiment>` again.

Use the full conversation history above, including your latest observations and
analysis, and submit the best final answer now. Output exactly one complete
`<final_formula>...</final_formula>` block and nothing else.

The module must define:
- `USED_INPUTS`
- `LAW_CONSTANTS = {}`
- `OTHER_CONSTANTS = {}`
- `LOCAL_FITTABLE = {}`
- `def predict(X):`

Use only valid Python. Make `predict(X)` return a finite numeric ndarray of
shape `(N,)`. If you are uncertain, choose the simplest physically reasonable
formula supported by your previous analysis and submit it now.
"""


FINAL_PROMPT_TYPEII = """\
Hard stop: do not run `<python>` or `<experiment>` again.

Use the full conversation history above, including your latest observations and
analysis, and submit the best final answer now. Output exactly one complete
`<final_formula>...</final_formula>` block and nothing else.

This is a multi-cluster Type II task. The module must define:
- `USED_INPUTS`
- `LAW_CONSTANTS = {}`
- `OTHER_CONSTANTS = {}`
- a non-empty `LOCAL_FITTABLE` dict with a few per-cluster parameters
- `def fit(X_fit, y_fit):` returning exactly those parameter keys
- `def predict(X, **params):` or `def predict(X, a, b, ...):`

Use the same functional form for every cluster. Do not use `group_id` in
`USED_INPUTS`, `fit`, or `predict`. Keep `fit()` fast and robust with sensible
fallbacks. If you are uncertain, choose the simplest shared mechanism supported
by your previous analysis and submit it now.
"""


RETRY_PROMPT = """\
Your previous response did not contain a parseable complete
`<final_formula>...</final_formula>` block with a `def predict`.

Output exactly one complete `<final_formula>...</final_formula>` block now.
Do not include prose and do not call `<python>` or `<experiment>`.
"""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_summary(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def _write_summary(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("mode", "")),
        str(row.get("type", "")),
        str(row.get("task", "")),
    )


def _paths(run_dir: Path, model: str, task_type: str, task: str) -> dict[str, Path]:
    base = run_dir / "parallel"
    return {
        "submission": base / "submissions" / model / task_type / f"{task}.py",
        "trajectory": base / "trajectories" / model / task_type / f"{task}.traj.json",
        "forced_trajectory": base / "trajectories" / model / task_type / f"{task}.forced_final.traj.json",
        "log": base / "logs" / model / task_type / f"{task}.force_final.log",
    }


def _select_jobs(run_dir: Path, model: str,
                 task_types: set[str] | None = None) -> list[dict[str, str]]:
    summary = run_dir / "summary.tsv"
    _, rows = _read_summary(summary)
    jobs: list[dict[str, str]] = []
    for row in rows:
        if row.get("model") != model or row.get("mode") != "parallel":
            continue
        if task_types is not None and row.get("type") not in task_types:
            continue
        if row.get("submitted") == "1":
            continue
        task_type = row.get("type") or ""
        task = row.get("task") or ""
        if not task_type or not task:
            continue
        paths = _paths(run_dir, model, task_type, task)
        if paths["submission"].exists() and paths["submission"].read_text(errors="replace").strip():
            continue
        jobs.append(row)
    return jobs


def _existing_forced_results(run_dir: Path, model: str) -> list[dict[str, Any]]:
    summary = run_dir / "summary.tsv"
    _, rows = _read_summary(summary)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("model") != model or row.get("mode") != "parallel":
            continue
        task_type = row.get("type") or ""
        task = row.get("task") or ""
        if not task_type or not task:
            continue
        paths = _paths(run_dir, model, task_type, task)
        submission = paths["submission"]
        forced_traj = paths["forced_trajectory"]
        if not submission.exists() or not submission.read_text(errors="replace").strip():
            continue
        if not forced_traj.exists():
            continue
        try:
            payload = json.loads(forced_traj.read_text(encoding="utf-8"))
            trial = payload.get("trial") or {}
        except Exception:
            trial = {}
        status = trial.get("status") or "completed_forced_final"
        if status != "completed_forced_final":
            continue
        out.append({
            "type": task_type,
            "task": task,
            "ok": True,
            "status": status,
            "elapsed_seconds": row.get("elapsed_seconds") or "0.000",
            "submission_path": str(submission),
            "trajectory_path": str(forced_traj),
            "log_path": str(paths["log"]),
            "total_tokens": str(trial.get("total_tokens") or row.get("total_tokens") or ""),
            "rounds": str(trial.get("rounds") or row.get("rounds") or ""),
            "error": "",
        })
    return out


def _prepare_messages(messages: list[dict[str, str]], tail_messages: int | None) -> list[dict[str, str]]:
    clean = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
             for m in messages if m.get("role") in {"system", "user", "assistant"}]
    if tail_messages is None or tail_messages <= 0 or len(clean) <= tail_messages + 2:
        return clean
    prefix = clean[:2]
    tail = clean[-tail_messages:]
    if tail and tail[0]["role"] == "assistant":
        # Avoid starting the truncated continuation with an assistant tool call
        # whose preceding user/tool output was dropped.
        tail = tail[1:]
    return prefix + [{
        "role": "user",
        "content": (
            "The middle of the trajectory was omitted to keep this forced-final "
            "request short. Use the original task instructions above and the "
            "most recent trajectory messages below to submit the best final formula."
        ),
    }] + tail


def _append_final_prompt(messages: list[dict[str, str]], task_type: str,
                         tail_messages: int | None = None) -> list[dict[str, str]]:
    out = _prepare_messages(messages, tail_messages)
    prompt = FINAL_PROMPT_TYPEII if task_type == "typeII" else FINAL_PROMPT_TYPEI
    out.append({"role": "user", "content": prompt})
    return out


def _usage_total(usages: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("prompt_tokens", "prompt_cached_tokens", "completion_tokens",
            "reasoning_tokens", "total_tokens")
    total = {k: 0 for k in keys}
    total["reasoning_content_chars"] = 0
    for usage in usages:
        for key in keys:
            total[key] += int(usage.get(key, 0) or 0)
        total["reasoning_content_chars"] += len(str(usage.get("reasoning_content") or ""))
    if usages:
        total["model"] = usages[-1].get("model")
        total["api_source"] = usages[-1].get("api_source")
    return total


def _call_once(messages: list[dict[str, str]], model: str, trial_id: str) -> tuple[str, dict[str, Any]]:
    # Import after main() applies environment knobs such as REASONING_EFFORT and
    # LLM_TIMEOUT_SECONDS; call_llm_api reads those at module import time.
    from call_llm_api import call_llm_api  # noqa: PLC0415

    content, reasoning, usage = call_llm_api(messages, model_name=model, trial_info={"trial_id": trial_id})
    usage = dict(usage or {})
    usage["reasoning_content"] = str(reasoning or "")
    usage["reasoning_content_chars"] = len(usage["reasoning_content"])
    return str(content or ""), usage


def run_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    model = args.model
    task_type = row["type"]
    task = row["task"]
    paths = _paths(run_dir, model, task_type, task)
    for key in ("submission", "forced_trajectory", "log"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    log_lines = [
        f"task={task_type}/{task}",
        f"model={model}",
        f"source_traj={paths['trajectory']}",
    ]
    try:
        payload = json.loads(paths["trajectory"].read_text(encoding="utf-8"))
        meta = dict(payload.get("meta") or {})
        trial = dict(payload.get("trial") or {})
        source_messages = trial.get("chat_history") or []
        messages = _append_final_prompt(source_messages, task_type, args.tail_messages)
        usages: list[dict[str, Any]] = []

        call_model = args.call_model or model
        response, usage = _call_once(messages, call_model, f"force_final_{model}_{task}")
        usages.append(usage)
        messages.append({"role": "assistant", "content": response})
        ok, submitted = proto.parse_final_formula(response)
        log_lines.append(f"attempt=1 ok={ok} finish={usage.get('finish_reason')} total_tokens={usage.get('total_tokens')}")

        if not ok and args.retries > 0:
            messages.append({"role": "user", "content": RETRY_PROMPT})
            response, usage = _call_once(messages, call_model, f"force_final_retry_{model}_{task}")
            usages.append(usage)
            messages.append({"role": "assistant", "content": response})
            ok, submitted = proto.parse_final_formula(response)
            log_lines.append(f"attempt=2 ok={ok} finish={usage.get('finish_reason')} total_tokens={usage.get('total_tokens')}")

        status = "completed_forced_final" if ok else "force_final_parse_failed"
        if ok:
            paths["submission"].write_text(submitted.strip() + "\n", encoding="utf-8")

        forced_payload = {
            "meta": {
                **meta,
                "mode": "force_final",
                "model": model,
                "source_checkpoint_path": str(paths["trajectory"]),
                "checkpoint_path": str(paths["forced_trajectory"]),
                "updated_at_unix": time.time(),
            },
            "trial": {
                "status": status,
                "submitted_equation": submitted if ok else "",
                "rounds": int(trial.get("rounds") or 0) + len(usages),
                "total_tokens": int(trial.get("total_tokens") or 0)
                + sum(int(u.get("total_tokens", 0) or 0) for u in usages),
                "usage_total": _usage_total(usages),
                "usage_per_turn": usages,
                "n_experiments": int(trial.get("n_experiments") or 0),
                "n_python_calls": int(trial.get("n_python_calls") or 0),
                "chat_history": messages,
                "source_status": trial.get("status"),
                "source_rounds": trial.get("rounds"),
                "source_total_tokens": trial.get("total_tokens"),
            },
        }
        _write_json(paths["forced_trajectory"], forced_payload)
        elapsed = time.time() - start
        log_lines.append(f"status={status} elapsed_seconds={elapsed:.3f}")
        paths["log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {
            "type": task_type,
            "task": task,
            "ok": ok,
            "status": status,
            "elapsed_seconds": f"{elapsed:.3f}",
            "submission_path": str(paths["submission"]),
            "trajectory_path": str(paths["forced_trajectory"]),
            "log_path": str(paths["log"]),
            "total_tokens": str(forced_payload["trial"]["total_tokens"]),
            "rounds": str(forced_payload["trial"]["rounds"]),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - batch salvage should keep going.
        elapsed = time.time() - start
        log_lines.append(f"error={type(exc).__name__}: {exc}")
        log_lines.append(f"elapsed_seconds={elapsed:.3f}")
        paths["log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {
            "type": task_type,
            "task": task,
            "ok": False,
            "status": "force_final_error",
            "elapsed_seconds": f"{elapsed:.3f}",
            "submission_path": str(paths["submission"]),
            "trajectory_path": str(paths["forced_trajectory"]),
            "log_path": str(paths["log"]),
            "total_tokens": "",
            "rounds": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _update_summaries(run_dir: Path, model: str, results: list[dict[str, Any]]) -> None:
    summary_tsv = run_dir / "summary.tsv"
    fieldnames, rows = _read_summary(summary_tsv)
    by_key = {(model, "parallel", r["type"], r["task"]): r for r in results}
    for row in rows:
        result = by_key.get(_row_key(row))
        if not result:
            continue
        row["returncode"] = "0" if result["ok"] else "125"
        row["status"] = result["status"]
        row["submitted"] = "1" if result["ok"] else "0"
        row["qualified_submission"] = "1" if result["ok"] else "0"
        row["rounds"] = result.get("rounds", row.get("rounds", ""))
        row["total_tokens"] = result.get("total_tokens", row.get("total_tokens", ""))
        row["elapsed_seconds"] = result.get("elapsed_seconds", row.get("elapsed_seconds", ""))
        row["submission_path"] = result.get("submission_path", row.get("submission_path", ""))
        row["trajectory_path"] = result.get("trajectory_path", row.get("trajectory_path", ""))
        row["log_path"] = result.get("log_path", row.get("log_path", ""))
        row["error"] = result.get("error", "")
    _write_summary(summary_tsv, fieldnames, rows)
    _write_json(run_dir / "summary.json", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="batch run dir containing summary.tsv")
    ap.add_argument("--model", default="or-glm52")
    ap.add_argument("--call-model", default=None,
                    help="model alias to call while still writing under --model output paths")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--task-types", nargs="*", default=None,
                    help="optional subset such as: --task-types typeII")
    ap.add_argument("--tail-messages", type=int, default=None,
                    help="keep system/task prompt plus only the last N trajectory messages")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--llm-timeout-seconds", type=float, default=None)
    args = ap.parse_args()

    if args.reasoning_effort:
        os.environ["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
        os.environ["REASONING_EFFORT"] = args.reasoning_effort
    if args.llm_timeout_seconds:
        os.environ["LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout_seconds)

    run_dir = Path(args.run_dir)
    task_types = set(args.task_types) if args.task_types else None
    jobs = _select_jobs(run_dir, args.model, task_types=task_types)
    print(f"RUN_DIR: {run_dir}")
    print(f"FORCE_FINAL_JOBS: {len(jobs)}  workers={args.workers}  retries={args.retries}", flush=True)
    if not jobs:
        return

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, job, args) for job in jobs]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            result = fut.result()
            results.append(result)
            print(
                f"[{idx}/{len(jobs)}] {args.model} parallel "
                f"{result['type']}/{result['task']} "
                f"status={result['status']} submitted={1 if result['ok'] else 0} "
                f"elapsed={result['elapsed_seconds']}",
                flush=True,
            )

    summary_results = _existing_forced_results(run_dir, args.model)
    seen = {(r["type"], r["task"]) for r in summary_results}
    summary_results.extend(
        r for r in results
        if (r["type"], r["task"]) not in seen
    )
    _update_summaries(run_dir, args.model, summary_results)
    report = {
        "created_at_utc": _utc_stamp(),
        "run_dir": str(run_dir),
        "model": args.model,
        "n_jobs": len(jobs),
        "n_ok": sum(1 for r in results if r["ok"]),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "results": sorted(results, key=lambda r: (r["type"], r["task"])),
    }
    _write_json(run_dir / "force_final_report.json", report)
    print(f"FORCE_FINAL_OK: {report['n_ok']}/{report['n_jobs']}")
    print(f"REPORT_JSON: {run_dir / 'force_final_report.json'}")


if __name__ == "__main__":
    main()
