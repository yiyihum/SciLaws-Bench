#!/usr/bin/env python3
"""Stage SciLaws-Bench parallel-structure jobs and optionally dispatch Codex judges.

Parallel-mode scoring is GT-structure recovery only. The judge receives the
hidden simulator `formula.py` and a solver `submission.py`, then assigns one of
the fixed ordinal scores:

  0.00 unrelated / invalid
  0.25 relevant variables or trend only
  0.50 main structure recovered
  0.75 main structure + most extra terms recovered
  1.00 full GT structure up to algebraic equivalence, coefficient signs/scales consistent

This intentionally does not compute prediction, validity, or test-set metrics.
Parallel rankings are structure-only.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import importlib.util
import inspect
import json
import math
import sys
import shutil
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_formula import check_row_independence  # noqa: E402


DEFAULT_CHUNK_SIZE = 3
DEFAULT_MAX_WORKERS = 1
ALLOWED_STRUCTURE_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)

BLOCKED_SUBMISSION_IMPORT_ROOTS = {
    "builtins",
    "glob",
    "importlib",
    "inspect",
    "io",
    "joblib",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "subprocess",
    "sys",
}

BLOCKED_SUBMISSION_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "file",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "raw_input",
    "setattr",
}

BLOCKED_SUBMISSION_CALL_ATTRS = {
    "fromfile",
    "genfromtxt",
    "get_handle",
    "load",
    "loadtxt",
    "open",
    "read_bytes",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_hdf",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_stata",
    "read_table",
    "read_text",
    "tofile",
}

DISPATCH_HELP = """\
Dispatch the structure judges:
  If --dispatch codex is used, evaluate_parallel.py calls `codex exec` once per
  generated prompt chunk. Each prompt contains staged task directories with:
    metadata.yaml
    formula.py       # hidden simulator GT
    submission.py    # solver answer

  Recommended chunking:
    --chunk-size 3

  Example:
    python harness/evaluate_parallel.py \\
      --tasks-dir tasks \\
      --submissions baseline_agent/batch_runs/.../submissions/gpt5.4/typeI \\
      --stage-root parallel_stage \\
      --output-root parallel_out \\
      --chunk-size 3 \\
      --dispatch codex \\
      --max-workers 4

  Results:
    Each subagent writes <OUTPUT_DIR>/<stage_id>.json.
    evaluate_parallel.py writes <OUTPUT_DIR>/parallel_summary.csv and
    <OUTPUT_DIR>/parallel_summary.json after dispatch.
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
            "formula": task_dir / "simulator" / "formula.py",
        }
    return index


def _submission_method(submission: Path, submissions_dir: Path) -> str | None:
    rel = submission.relative_to(submissions_dir)
    if len(rel.parts) == 1:
        return None
    return "__".join(rel.parts[:-1])


def _stage_id(method: str | None, task: str) -> str:
    return f"{method}__{task}" if method else task


def _find_formula(task_info: dict[str, Path | str], scoring_dir: Path | None) -> Path:
    formula = Path(task_info["formula"])
    if formula.exists():
        return formula
    if scoring_dir is not None:
        alt = scoring_dir / str(task_info["type"]) / str(task_info["task"]) / "simulator" / "formula.py"
        if alt.exists():
            return alt
        alt = scoring_dir / str(task_info["type"]) / str(task_info["task"]) / "formula.py"
        if alt.exists():
            return alt
    return formula


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
            f"  gt_formula: {_as_abs(task_dir / 'formula.py')}\n"
            f"  submission: {_as_abs(task_dir / 'submission.py')}"
        )
    tasks_block = "\n".join(task_lines)
    return f"""You are a PARALLEL STRUCTURE JUDGE for RealSR.

Read only the staged task directories listed below. Write only JSON result files
under the output directory. Do not edit existing repository files.

OUTPUT_DIR: {_as_abs(output_dir)}

Staged tasks:
{tasks_block}

Goal:
Score whether the submitted formula recovers the hidden simulator GT mechanism
structure. This is NOT a prediction-score task.

Allowed structure_score values are exactly:
- 0.00: unrelated / invalid
- 0.25: relevant variables or trend only
- 0.50: main structure recovered
- 0.75: main structure + most extra terms recovered
- 1.00: full GT structure up to algebraic equivalence, coefficient signs/scales consistent

For EACH staged task:
1. Read task_dir/metadata.yaml for target and input names.
2. Read task_dir/formula.py as the hidden simulator GT. Treat this as the
   reference mechanism, not as public solver context.
3. Read task_dir/submission.py as the solver answer.
4. Compare GT and submission by source inspection and, if useful, small
   diagnostic Python probes over metadata input ranges. Do not compute or report
   prediction scores, validity scores, test RMSE, or leaderboard-style
   performance.
5. Count algebraic equivalence as correct:
   - log base changes are equivalent if coefficients transform accordingly.
   - R^g/(R^g+RA^g) is equivalent to sigmoid(g*log(R/RA)).
   - Renaming helper variables or factoring terms is equivalent.
6. Penalize missing mechanism terms even if predictions are close. In
   particular, task-specific correction terms, interaction terms, saturation
   terms, piecewise regimes, offsets, exponents, and nested transforms are what
   distinguish 0.75/1.00 from 0.50.
7. Coefficients should affect only the jump from 0.75 to 1.00 unless the wrong
   sign/order of magnitude changes the mechanism.

Use this decision rule:
- 0.00: submission cannot be imported, lacks predict, uses wrong target shape,
  or is structurally unrelated to the GT mechanism.
- 0.25: uses relevant variables or gets a monotone/trend direction, but the main
  functional family is wrong.
- 0.50: recovers the main functional skeleton / outer family and key canonical
  variables, but misses important GT correction/interaction/extra terms.
- 0.75: recovers the main skeleton and most important extra terms, but has
  incomplete secondary terms or coefficient/sign/scale mismatches.
- 1.00: recovers the full GT structure up to algebraic equivalence, including
  key extra terms, with coefficient signs/scales consistent.

Write exactly one JSON per task to OUTPUT_DIR/<stage_id>.json:
{{
  "task": "<stage_id>",
  "structure_score": 0.0 | 0.25 | 0.5 | 0.75 | 1.0,
  "level": "unrelated_or_invalid | variables_or_trend_only | main_structure | main_plus_extra_terms | full_gt_structure",
  "error": <string or null>,
  "gt_summary": "one-line GT mechanism summary",
  "submission_summary": "one-line submitted mechanism summary",
  "matched": ["short evidence bullets"],
  "missed": ["short missed-structure bullets"],
  "coefficient_assessment": "short note on signs/scales/equivalence"
}}

After all tasks, reply one line per task:
<stage_id> structure=<score>
"""


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def stage_parallel_jobs(args: argparse.Namespace) -> dict[str, Any]:
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

        formula = _find_formula(task_info, scoring_dir)
        if not formula.exists():
            skipped.append({
                "stage_id": stage_id,
                "type": task_info["type"],
                "task": task,
                "submission": _as_abs(submission),
                "skip": "missing_simulator_formula",
            })
            continue

        dst = stage_dir / stage_id
        dst.mkdir()
        shutil.copy2(Path(task_info["metadata"]), dst / "metadata.yaml")
        shutil.copy2(formula, dst / "formula.py")
        shutil.copy2(submission, dst / "submission.py")

        staged.append({
            "stage_id": stage_id,
            "method": method,
            "type": task_info["type"],
            "task": task,
            "task_dir": _as_abs(Path(task_info["task_dir"])),
            "submission": _as_abs(submission),
            "formula": _as_abs(formula),
            "staged_task_dir": _as_abs(dst),
            "staged_formula": _as_abs(dst / "formula.py"),
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
        "score_values": list(ALLOWED_STRUCTURE_SCORES),
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


def _score_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            score = float(value.strip())
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        score = float(value)
    else:
        return None
    if not math.isfinite(score):
        return None
    return score if any(abs(score - allowed) <= 1e-9 for allowed in ALLOWED_STRUCTURE_SCORES) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _range_pair(meta: dict[str, Any]) -> tuple[float, float]:
    rng = (meta or {}).get("range")
    if isinstance(rng, dict):
        rng = rng.get("train") or rng.get("test")
    if isinstance(rng, (list, tuple)) and len(rng) >= 2:
        lo, hi = float(rng[0]), float(rng[1])
        if math.isfinite(lo) and math.isfinite(hi) and lo != hi:
            return (min(lo, hi), max(lo, hi))
        if math.isfinite(lo):
            return (lo - 1.0, lo + 1.0)
    return (0.0, 1.0)


def _sample_matrix(metadata: dict[str, Any], used_inputs: list[str], n: int = 8) -> np.ndarray:
    input_meta = {
        item.get("name"): item
        for item in (metadata.get("inputs") or [])
        if isinstance(item, dict) and item.get("name")
    }
    cols = []
    for name in used_inputs:
        meta = input_meta.get(name)
        if meta is None:
            raise ValueError(f"USED_INPUTS contains unknown metadata input {name!r}")
        lo, hi = _range_pair(meta)
        cols.append(np.linspace(lo, hi, n, dtype=float))
    return np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)


def _sample_y(metadata: dict[str, Any], n: int = 8) -> np.ndarray:
    lo, hi = _range_pair(metadata.get("target") or {})
    return np.linspace(lo, hi, n, dtype=float)


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _validate_submission_static(path: Path) -> tuple[bool, str]:
    """Reject submission modules that can read files or introspect evaluator state."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"submission syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BLOCKED_SUBMISSION_IMPORT_ROOTS:
                    return False, f"blocked import in submission: {alias.name!r}"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                return False, "relative imports are not allowed in submissions"
            root = node.module.split(".", 1)[0]
            if root in BLOCKED_SUBMISSION_IMPORT_ROOTS:
                return False, f"blocked import in submission: {node.module!r}"
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            attr = name.rsplit(".", 1)[-1]
            if name in BLOCKED_SUBMISSION_CALL_NAMES or attr in BLOCKED_SUBMISSION_CALL_ATTRS:
                return False, f"blocked call in submission: {name!r}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, f"blocked dunder attribute in submission: {node.attr!r}"
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                return False, f"blocked dunder name in submission: {node.id!r}"
    return True, ""


def _contract_check(stage_task_dir: Path, stage_id: str) -> tuple[bool, str]:
    """Deterministic submission contract gate for structure judging.

    This is not a prediction metric. It only verifies that the submitted module
    imports, exposes the required API, and can run on metadata-range probe rows.
    """
    metadata = yaml.safe_load((stage_task_dir / "metadata.yaml").read_text(encoding="utf-8")) or {}
    task_type = metadata.get("type") or ("typeII" if metadata.get("has_group_id") else "typeI")
    submission_path = stage_task_dir / "submission.py"
    static_ok, static_error = _validate_submission_static(submission_path)
    if not static_ok:
        return False, static_error
    module = _load_module(submission_path, f"_parallel_contract_{stage_id}")

    used_inputs = getattr(module, "USED_INPUTS", None)
    if not isinstance(used_inputs, list) or not used_inputs:
        return False, "USED_INPUTS must be a non-empty list"
    if not hasattr(module, "predict"):
        return False, "missing predict"

    predict_sig = inspect.signature(module.predict)
    predict_params = list(predict_sig.parameters)
    if task_type == "typeI" and predict_params != ["X"]:
        return False, f"typeI predict signature must be predict(X), got {predict_sig}"
    if task_type == "typeII" and (not predict_params or predict_params[0] != "X"):
        return False, f"typeII predict first argument must be X, got {predict_sig}"

    X = _sample_matrix(metadata, used_inputs, n=8)
    if task_type == "typeII":
        local = getattr(module, "LOCAL_FITTABLE", None)
        if not isinstance(local, dict) or not local:
            return False, "typeII LOCAL_FITTABLE must be a non-empty dict"
        if not hasattr(module, "fit"):
            return False, "typeII missing fit"
        fit_sig = inspect.signature(module.fit)
        if list(fit_sig.parameters)[:2] != ["X_fit", "y_fit"]:
            return False, f"typeII fit signature must start fit(X_fit, y_fit), got {fit_sig}"
        params = module.fit(X, _sample_y(metadata, n=len(X)))
        if not isinstance(params, dict):
            return False, "fit must return a dict"
        missing = set(local) - set(params)
        extra = set(params) - set(local)
        if missing or extra:
            return False, f"fit keys mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        predict_call = lambda Xr: module.predict(Xr, **params)  # noqa: E731
    else:
        predict_call = module.predict
    y = predict_call(X)

    arr = np.asarray(y, dtype=float)
    if arr.shape != (len(X),):
        return False, f"predict returned shape {arr.shape}, expected {(len(X),)}"
    if not np.all(np.isfinite(arr)):
        return False, "predict returned non-finite values"
    # Same row-independence rule as the numeric scorer: predict(X)[i] may
    # depend only on X[i].
    row_dependence = check_row_independence(predict_call, X, arr)
    if row_dependence:
        return False, f"row_dependence: {row_dependence}"
    return True, ""


def _summary_stats(rows: list[dict[str, Any]], method: str | None, task_type: str | None) -> dict[str, Any]:
    selected = [
        row for row in rows
        if (method is None or row["method"] == method)
        and (task_type is None or row["type"] == task_type)
    ]
    scores = [row["structure_score"] for row in selected if row["structure_score"] is not None]
    return {
        "method": "ALL" if method is None else (method or "direct"),
        "type": task_type or "ALL",
        "n": len(selected),
        "scored": len(scores),
        "mean_scored": _mean(scores),
        "strict_mean": sum(score if score is not None else 0.0 for score in (row["structure_score"] for row in selected)) / len(selected)
        if selected else None,
    }


def aggregate_parallel_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
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
            "structure_score": None,
            "judge_structure_score": None,
            "contract_ok": "",
            "level": "",
            "status": "missing_result",
            "error": "",
        }
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                score = _score_value(result.get("structure_score"))
                try:
                    contract_ok, contract_error = _contract_check(Path(item["staged_task_dir"]), stage_id)
                except Exception as exc:
                    contract_ok = False
                    contract_error = f"{type(exc).__name__}: {exc}"
                row.update({
                    "structure_score": score if contract_ok else 0.0,
                    "judge_structure_score": score,
                    "contract_ok": "true" if contract_ok else "false",
                    "level": result.get("level") or "",
                    "status": (
                        "ok" if contract_ok and score is not None
                        else "contract_invalid" if not contract_ok
                        else "invalid_structure_score"
                    ),
                    "error": contract_error if not contract_ok else (result.get("error") or ""),
                })
            except Exception as exc:
                row.update({
                    "status": "invalid_result_json",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        rows.append(row)

    summary_csv = output_dir / "parallel_summary.csv"
    fieldnames = [
        "method",
        "type",
        "task",
        "stage_id",
        "structure_score",
        "judge_structure_score",
        "contract_ok",
        "level",
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
    summary_json = output_dir / "parallel_summary.json"
    summary_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    print("PARALLEL_SUMMARY_CSV:", _as_abs(summary_csv))
    print("PARALLEL_SUMMARY_JSON:", _as_abs(summary_json))
    overall = stats["overall"]
    print(
        "parallel overall "
        f"n={overall['n']} scored={overall['scored']} "
        f"mean_scored={overall['mean_scored']} strict_mean={overall['strict_mean']}"
    )
    return {"rows": rows, "stats": stats, "summary_csv": _as_abs(summary_csv), "summary_json": _as_abs(summary_json)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage SciLaws-Bench parallel structure jobs and optionally dispatch Codex judges.",
        epilog=DISPATCH_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=_repo_root(),
                        help="Repository root used as the Codex working directory.")
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--submissions", type=Path, default=None,
                        help="Directory with <task>.py files, or method subdirs containing <task>.py files.")
    parser.add_argument("--scoring-dir", type=Path, default=Path("scoring"),
                        help="Optional hidden scoring tree fallback for simulator formula.py.")
    parser.add_argument("--stage-root", type=Path, default=Path("parallel_stage"))
    parser.add_argument("--output-root", type=Path, default=Path("parallel_out"))
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
                        help="Read an existing manifest/output dir and write summaries without staging.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest path for --aggregate-only.")
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
    args.manifest = _resolve_under_repo(args.manifest, args.repo_root)

    if args.aggregate_only:
        if args.manifest is None:
            raise SystemExit("--aggregate-only requires --manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        aggregate_parallel_outputs(manifest)
        return 0
    if args.submissions is None:
        raise SystemExit("--submissions is required unless --aggregate-only is used")

    manifest = stage_parallel_jobs(args)
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
        aggregate_parallel_outputs(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
