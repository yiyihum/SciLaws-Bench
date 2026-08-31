"""SciLaws-Bench scoring harness — two parallel scores (see README → How scores are defined).

  numeric_score  — deterministic, reference-relative. Computed by `evaluate_numeric.py`
                   (`score_one`), exposed here as `evaluate_on_test()`. Type II
                   is 3-seed averaged. Needs only the task's
                   data + reference_metrics.json — no API key, no network.
  validity_score — produced SEPARATELY by a code-enabled judge that executes the
                   formula on the data and scores the task's `validity_rubrics`
                   (see VALIDITY_JUDGE.md). Not computed here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import evaluate_numeric as _ev  # noqa: E402  official harness scorer (numeric)
import eval_formula as _ef      # noqa: E402  execution core (load_flat/clusters)

_DIAG_METRICS = ("rmse", "mae", "mse", "mdae", "smape", "mape", "log_mae", "r2")

__all__ = ["evaluate_on_test"]


def _load_submission_module(code: str) -> Tuple[Any, str]:
    """Write the submitted module text to a real temp .py and import it, then
    inject the harness-contract defaults the LLM may have omitted. A real file
    is used so source-reading tools work. Returns (module, temp_path); the
    caller unlinks temp_path."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="_llm_submission_")
    with os.fdopen(fd, "w") as fh:
        fh.write(code)
    spec = importlib.util.spec_from_file_location(f"_llm_submission_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "USED_INPUTS"):
        mod.USED_INPUTS = []
    if not hasattr(mod, "LAW_CONSTANTS"):
        mod.LAW_CONSTANTS = {}
    if not hasattr(mod, "OTHER_CONSTANTS"):
        mod.OTHER_CONSTANTS = {}
    if not hasattr(mod, "LOCAL_FITTABLE"):
        mod.LOCAL_FITTABLE = {}
    elif isinstance(mod.LOCAL_FITTABLE, (list, tuple, set)):
        mod.LOCAL_FITTABLE = {k: {} for k in mod.LOCAL_FITTABLE}
    return mod, path


def evaluate_on_test(submitted_code: str, task, **_ignored) -> Dict[str, Any]:
    """Score `submitted_code` against `task` — numeric_score only,
    via the official numeric harness `score_one` (Type II 3-seed averaged). validity_score
    is produced separately by the cc judge (VALIDITY_JUDGE.md).

    Returns: status, contract_ok, numeric_score, numeric_score_std,
    numeric_score_per_seed, metric, best_reference_id, raw_metric, metrics,
    violations, error.
    """
    task_dir = Path(task.task_dir)
    meta = _ev.load_task(task_dir)
    task_type = meta.get("type", "typeII")
    ref_path = _ev.reference_metrics_path(task_dir)   # private scoring/ tree
    if not ref_path.exists():
        return {"status": "no_reference", "contract_ok": None,
                "error": f"{ref_path} missing — run `evaluate_numeric.py reference`",
                "numeric_score": 0.0, "raw_numeric_score": None,
                "numeric_score_std": 0.0,
                "numeric_score_per_seed": [0.0] if task_type == "typeI" else [0.0] * _ev.N_SEEDS}
    ref_metrics = json.loads(ref_path.read_text())

    try:
        mod, tmp_path = _load_submission_module(submitted_code)
    except Exception as e:
        return {"status": "compile_error", "contract_ok": False,
                "error": f"{type(e).__name__}: {e}",
                "numeric_score": 0.0, "raw_numeric_score": None,
                "numeric_score_std": 0.0,
                "numeric_score_per_seed": [0.0] if task_type == "typeI" else [0.0] * _ev.N_SEEDS}
    try:
        # Context-ablation arms rename the columns the agent saw (x_1, ...). The
        # harness reads the REAL test.csv, so map USED_INPUTS back to real names
        # (order preserved → predict's positional X columns stay correct).
        v2r = getattr(task, "view_to_real", None)
        if v2r and getattr(mod, "USED_INPUTS", None):
            mod.USED_INPUTS = [v2r.get(c, c) for c in mod.USED_INPUTS]

        data = _ef.load_flat(task_dir) if task_type == "typeI" else _ef.load_clusters(task_dir)
        # numeric_score needs only reference_metrics.json (anchors); the reference
        # baseline .py files are not shipped, so the registry is empty.
        r = _ev.score_one(mod, "submission", data, ref_metrics, meta, {})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not r.get("contract_ok"):
        return {"status": "contract_fail", "contract_ok": False,
                "violations": r.get("violations"),
                "numeric_score": r.get("numeric_score", 0.0),
                "raw_numeric_score": r.get("raw_numeric_score"),
                "numeric_score_std": r.get("numeric_score_std", 0.0),
                "numeric_score_per_seed": r.get("numeric_score_per_seed"),
                "metric": _ev.task_metric(meta)}

    failed = bool(r.get("failed")) if task_type == "typeI" \
        else r.get("n_clusters_failed") == len(data["cluster_ids"])
    score = r.get("score") or {}
    metric = score.get("metric") or _ev.task_metric(meta)
    raw_metric = r.get("raw_metric")
    out: Dict[str, Any] = {
        "status": r.get("status") or ("ok" if not failed else "exec_error"),
        "contract_ok": True,
        "error": r.get("error"),
        "numeric_score": r.get("numeric_score"),
        "raw_numeric_score": r.get("raw_numeric_score"),
        "numeric_score_std": r.get("numeric_score_std"),
        "numeric_score_per_seed": r.get("numeric_score_per_seed"),
        "metric": metric,
        "best_reference_id": score.get("best_reference_id"),
        "raw_metric": raw_metric,
        "metrics": {metric: raw_metric} if isinstance(raw_metric, (int, float)) else {},
    }
    if task_type != "typeI":
        out["per_cluster_score"] = score.get("per_cluster_score")
        out["n_clusters_scored"] = score.get("n_clusters_scored")
    return out
