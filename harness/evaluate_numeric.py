"""evaluate_numeric.py — the deterministic numeric scoring driver.

The harness has just two files:
  eval_formula.py  — the execution core (run one formula -> raw metrics)
  evaluate_numeric.py — this file: everything built on top of that core.

Two modes:

  reference <task_dir>
      Run the task's reference bank + naive predictor through eval_formula,
      write formulas/reference_metrics.json (the skill-normalisation anchors,
      stored alongside the reference formulas they measure).
      A task-setup step — run once.

  score <task_dir> [submission.py]
      Run a submission and report numeric_score: deterministic,
      reference-relative test-set performance.
      With no submission path: self-test — score each reference baseline as
      if it were a submission (the best baseline should land numeric ≈ 0.5).

Usage:
    python harness/evaluate_numeric.py reference <task_dir>
    python harness/evaluate_numeric.py score     <task_dir> [submission.py]
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

from eval_formula import (  # noqa: E402
    METRICS, load_clusters, load_flat, run_formula, run_formula_flat,
)
# validity_score is intentionally not part of this numeric scorer. It is
# produced separately by the Claude-Code judge described in VALIDITY_JUDGE.md.

# A cluster is non-discriminative (excluded) if the best reference baseline is
# itself (near-)perfect on it — the score ratio sub/ref then blows up.
REF_EPS = 1e-4

# A Type II submission's fit() may be stochastic. The harness runs it under
# N_SEEDS fixed seeds and reports mean / std of numeric_score (Scheme B).
# A deterministic fit() gives identical runs → std 0. Type I has no fit() and
# is run once. BASE_SEED is fixed and documented for reproducibility.
BASE_SEED = 20260514
N_SEEDS = 3


def _is_higher_better(metric: str) -> bool:
    return METRICS[metric]["direction"] == "higher"


# ==========================================================================
# shared helpers
# ==========================================================================

def load_task(task_dir: Path) -> dict:
    if not (task_dir / "metadata.yaml").exists():
        raise SystemExit(f"no metadata.yaml under {task_dir}")
    return yaml.safe_load((task_dir / "metadata.yaml").open())


def scoring_dir(task_dir: Path) -> Path:
    """Map a PUBLIC task dir (.../tasks/<type>/<task>) to its PRIVATE scoring dir
    (.../scoring/<type>/<task>), which holds reference_metrics.json and
    validity_rubrics.json. The public tree (tasks/) is what solvers receive; the
    scoring anchors live in the withheld scoring/ tree."""
    parts = list(Path(task_dir).resolve().parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "tasks":
            parts[i] = "scoring"
            return Path(*parts)
    raise SystemExit(f"cannot locate scoring/ tree for {task_dir} "
                     f"(expected a 'tasks' component in the path)")


def reference_metrics_path(task_dir: Path) -> Path:
    public_eval = Path(task_dir) / "eval" / "reference_metrics.json"
    if public_eval.exists():
        return public_eval
    return scoring_dir(task_dir) / "reference_metrics.json"


def task_metric(meta: dict) -> str:
    """The task's declared metric — required, and must be in the registry.

    No silent default: a task that forgets `metric:` is a setup error, not
    something to paper over with rmse.
    """
    m = meta.get("metric")
    if m is None:
        raise SystemExit("metadata.yaml has no `metric:` field (required)")
    if m not in METRICS:
        raise SystemExit(f"metadata `metric: {m}` is not a known metric — "
                         f"choose one of {sorted(METRICS)}")
    return m


def load_task_registry(task_dir: Path) -> dict:
    """Import the task's `formulas` package, return its REGISTRY dict."""
    sys.path.insert(0, str(task_dir))
    import formulas  # noqa: PLC0415  (task-local package via sys.path)
    importlib.reload(formulas)
    return formulas.REGISTRY


def load_submission(path: Path):
    spec = importlib.util.spec_from_file_location(f"_submission_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# mode: reference — build the skill-normalisation anchors
# ==========================================================================

def _ref_entry(mod, extra: dict) -> dict:
    """Common metadata block for a reference baseline entry."""
    return {
        "kind": "reference",
        "paper_ref": getattr(mod, "PAPER_REF", None),
        "equation_loc": getattr(mod, "EQUATION_LOC", None),
        "law_constants":   {k: float(v) for k, v in mod.LAW_CONSTANTS.items()},
        "other_constants": {k: float(v) for k, v in mod.OTHER_CONSTANTS.items()},
        "local_fittable":  sorted(mod.LOCAL_FITTABLE.keys()),
        **extra,
    }


# Anti-dump caps are derived from the reference bank (§7.6). Static caps =
# the most any reference paper itself uses. fit_timeout = the slowest
# measured reference fit × a safety factor (the factor absorbs machine
# timing variance — the cap only needs to catch order-of-magnitude abuse).
FIT_TIMEOUT_FACTOR = 10
FIT_TIMEOUT_FLOOR  = 10   # seconds


def derive_caps(registry: dict, max_ref_fit_seconds: float, task_type: str) -> dict:
    """Derive the anti-dump caps from the reference bank.

    A submission may be as complex as the most complex published formula —
    no more. All four caps come from the bank, not from a task author.
    """
    law_counts   = [len(m.LAW_CONSTANTS) for m in registry.values()]
    local_counts = [len(m.LOCAL_FITTABLE) for m in registry.values()]
    init_sizes = [1]
    for m in registry.values():
        for spec in m.LOCAL_FITTABLE.values():
            init = spec.get("init") if isinstance(spec, dict) else None
            init_sizes.append(len(init) if isinstance(init, (list, tuple)) else 1)
    caps = {
        "max_law_constants":       max(law_counts) if law_counts else 0,
        "max_local_params":        max(local_counts) if local_counts else 0,
        "max_init_size_per_param": max(init_sizes),
    }
    if task_type == "typeI":
        caps["fit_timeout_seconds"] = None        # Type I has no fit()
    else:
        caps["fit_timeout_seconds"] = max(
            FIT_TIMEOUT_FLOOR,
            int(np.ceil(max_ref_fit_seconds * FIT_TIMEOUT_FACTOR)),
        )
    return caps


def mode_reference(task_dir: Path) -> int:
    meta = load_task(task_dir)
    task_type = meta.get("type", "typeII")
    target_name = meta["target"]["name"]
    task_metric(meta)                       # fail fast if metric is missing / unknown
    registry = load_task_registry(task_dir)
    baselines: dict[str, dict] = {}
    max_ref_fit_seconds = 0.0

    if task_type == "typeI":
        flat = load_flat(task_dir)
        print(f"[{meta['task_id']}] Type I — {len(flat['test_rows'])} flat test rows.",
              flush=True)
        for stem in sorted(registry):
            mod = registry[stem]
            res = run_formula_flat(mod, flat, target_name)
            baselines[stem] = _ref_entry(mod, {
                "failed": res["failed"], "error": res["error"],
                "metrics": res["metrics"],
            })
        n_units = len(flat["test_rows"])
    else:
        clusters = load_clusters(task_dir)
        n_units = len(clusters["cluster_ids"])
        print(f"[{meta['task_id']}] Type II — {n_units} test clusters.", flush=True)
        for stem in sorted(registry):
            mod = registry[stem]
            # References run UNtimed — we measure them to derive the cap.
            # seed=BASE_SEED: reference formulas are deterministic, but fixing
            # the seed makes even a stochastic reference reproducible.
            res = run_formula(mod, clusters, target_name,
                              fit_timeout_seconds=None, seed=BASE_SEED)
            max_ref_fit_seconds = max(max_ref_fit_seconds, res.get("max_fit_seconds", 0.0))
            baselines[stem] = _ref_entry(mod, {
                "n_clusters_fitted": res["n_clusters_fitted"],
                "n_clusters_failed": res["n_clusters_failed"],
                "max_fit_seconds": res.get("max_fit_seconds", 0.0),
                "per_cluster": {str(c): v["metrics"] for c, v in res["per_cluster"].items()},
            })

    caps = derive_caps(registry, max_ref_fit_seconds, task_type)
    out = {
        "task": meta["task_id"],
        "type": task_type,
        "metric_declared": meta.get("metric"),
        "reference_baseline_id": meta.get("reference_baseline_id"),
        ("n_test_rows" if task_type == "typeI" else "n_clusters"): n_units,
        "derived_caps": caps,
        "baselines": dict(sorted(baselines.items())),
    }
    out_path = reference_metrics_path(task_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[reference] wrote {out_path}")
    print(f"  derived caps: {caps}")

    metric = task_metric(meta)
    hdr = metric if task_type == "typeI" else f"mean-cluster-{metric}"
    print(f"\n  {'baseline':<26} {hdr:>20}")
    for name, b in out["baselines"].items():
        v = _baseline_metric(b, metric)
        if v is None:
            print(f"  {name:<26}  FAILED: {b.get('error')}")
        else:
            print(f"  {name:<26} {v:>20.4f}")
    return 0


# ==========================================================================
# mode: score — scoring layers on top of eval_formula
# ==========================================================================

def validate_contract(mod, caps: dict) -> list[str]:
    """Return contract violations (empty list = passes).

    `caps` is the `derived_caps` block from reference_metrics.json — the
    anti-dump caps derived from the reference bank (§7.6).
    """
    errs: list[str] = []
    for field in ("USED_INPUTS", "LAW_CONSTANTS", "OTHER_CONSTANTS", "LOCAL_FITTABLE"):
        if not hasattr(mod, field):
            errs.append(f"missing required field: {field}")
    if errs:
        return errs

    local = mod.LOCAL_FITTABLE
    is_type_ii = bool(local)

    if not hasattr(mod, "predict"):
        errs.append("missing predict()")
    else:
        if "group_id" in inspect.signature(mod.predict).parameters:
            errs.append("predict() signature contains 'group_id' (forbidden — anti-dump)")

    if is_type_ii and not hasattr(mod, "fit"):
        errs.append("LOCAL_FITTABLE non-empty but fit() missing (Type II requires fit())")
    if not is_type_ii and hasattr(mod, "fit"):
        errs.append("LOCAL_FITTABLE empty but fit() present (Type I must not define fit())")

    cap_law = caps.get("max_law_constants")
    if cap_law is not None and len(mod.LAW_CONSTANTS) > cap_law:
        errs.append(f"len(LAW_CONSTANTS)={len(mod.LAW_CONSTANTS)} exceeds "
                    f"max_law_constants={cap_law}")
    cap_local = caps.get("max_local_params")
    if cap_local is not None and len(local) > cap_local:
        errs.append(f"len(LOCAL_FITTABLE)={len(local)} exceeds max_local_params={cap_local}")
    cap_init = caps.get("max_init_size_per_param")
    if cap_init is not None:
        for name, spec in local.items():
            init = spec.get("init") if isinstance(spec, dict) else None
            if isinstance(init, (list, tuple)) and len(init) > cap_init:
                errs.append(f"LOCAL_FITTABLE['{name}']['init'] length {len(init)} "
                            f"exceeds max_init_size_per_param={cap_init}")
    return errs


def _score_one_cluster(sub_v: float, ref_v: float, metric: str) -> float:
    """Reference-relative score for one cluster (or the flat Type I test set).

    Lower-is-better  (perfect = 0):  score = 1 - 0.5 * sub / ref
    Higher-is-better (perfect = P):  score = 0.5 + 0.5 * (sub - ref) / (P - ref)
    Clipped to [0, 1]. ref -> 0.5, perfect -> 1.0, 2x the ref error -> 0.
    The reference is never (near-)perfect here — those units are excluded
    upstream by `_ref_nondiscriminative`, so the denominators are non-zero.
    """
    spec = METRICS[metric]
    if spec["direction"] == "higher":
        score = 0.5 + 0.5 * (sub_v - ref_v) / (spec["perfect"] - ref_v)
    else:
        score = 1.0 - 0.5 * sub_v / ref_v          # lower-is-better, perfect = 0
    return float(np.clip(score, 0.0, 1.0))


def _ref_nondiscriminative(ref_v: float, metric: str) -> bool:
    """True if the reference is itself (near-)perfect on this unit → exclude."""
    return abs(ref_v - METRICS[metric]["perfect"]) <= REF_EPS


def _baseline_metric(b: dict, metric: str) -> float | None:
    """The single raw-metric value scoring uses for a reference baseline.

    Type II → mean of the per-cluster metric (scoring is per-cluster, so the
    representative raw number is the per-cluster mean — never a cross-cluster
    pool). Type I → the metric on the flat test set. None if unavailable.
    """
    pc = b.get("per_cluster")
    if pc:                                               # Type II
        vals = [m[metric] for m in pc.values() if m and m.get(metric) is not None]
        return float(np.mean(vals)) if vals else None
    fm = b.get("metrics")                                # Type I
    if fm and fm.get(metric) is not None:
        return float(fm[metric])
    return None


def _best_reference(ref_metrics: dict, metric: str) -> tuple[str | None, str]:
    """Pick the best reference baseline in the bank.

    Best = argmin/argmax of `_baseline_metric` (mean per-cluster for Type II,
    flat-set metric for Type I). `reference_baseline_id` in metadata is only
    a label; the anchor is the empirically best baseline so the 0.5 mark is
    always the strongest paper.
    """
    higher = _is_higher_better(metric)
    cand: dict[str, float] = {}
    for name, b in ref_metrics["baselines"].items():
        if b.get("kind") != "reference":
            continue
        v = _baseline_metric(b, metric)
        if v is not None:
            cand[name] = v
    if not cand:
        return None, "no reference baseline produced a finite metric"
    best = max(cand, key=cand.get) if higher else min(cand, key=cand.get)
    return best, ""


def _zero_score_result(
    label: str,
    meta: dict,
    status: str,
    *,
    contract_ok: bool | None,
    error: str | None = None,
    violations: list[str] | None = None,
    note: str | None = None,
    score: dict | None = None,
) -> dict:
    metric = task_metric(meta)
    n_seeds = 1 if meta.get("type", "typeII") == "typeI" else N_SEEDS
    score_obj = score or {"metric": metric, "numeric_score": 0.0}
    score_obj.setdefault("metric", metric)
    score_obj.setdefault("numeric_score", 0.0)
    if note is not None:
        score_obj.setdefault("note", note)
    return {
        "submission": label,
        "contract_ok": contract_ok,
        "status": status,
        "error": error,
        "violations": violations,
        "note": note,
        "metric": metric,
        "raw_metric": None,
        "score": score_obj,
        "numeric_score": 0.0,
        "raw_numeric_score": None,
        "numeric_score_std": 0.0,
        "numeric_score_per_seed": [0.0] * n_seeds,
        "raw_numeric_score_per_seed": [None] * n_seeds,
    }


def compute_numeric_score(sub_result: dict, ref_metrics: dict, meta: dict) -> dict:
    """Type II reference-relative score: per-cluster score, equal-weight mean.

    score_N = clip(1 - 0.5 * sub_N / ref_N, 0, 1)    [lower-is-better]
    where ref_N is the per-cluster metric of the BEST reference baseline.
    Failed cluster -> score_N = 0. Cluster where the reference is itself
    (near-)perfect is excluded (non-discriminative).
    """
    metric = task_metric(meta)
    best_id, why = _best_reference(ref_metrics, metric)
    if best_id is None:
        return {"metric": metric, "numeric_score": 0.0, "raw_numeric_score": None,
                "note": why}
    ref_pc = ref_metrics["baselines"][best_id]["per_cluster"]

    per_cluster_score: dict[str, float] = {}
    excluded: list[str] = []

    for cid_str, ref_m in ref_pc.items():
        if ref_m is None or ref_m.get(metric) is None:
            excluded.append(cid_str)
            continue
        ref_v = ref_m[metric]
        if _ref_nondiscriminative(ref_v, metric):
            excluded.append(cid_str)
            continue
        sub_pc = sub_result["per_cluster"].get(int(cid_str))
        if sub_pc is None or sub_pc["failed"] or sub_pc["metrics"] is None:
            score = 0.0
        else:
            score = _score_one_cluster(sub_pc["metrics"][metric], ref_v, metric)
        per_cluster_score[cid_str] = score

    numeric_score = float(np.mean(list(per_cluster_score.values()))) if per_cluster_score else 0.0
    return {
        "metric": metric,
        "best_reference_id": best_id,
        "n_clusters_scored": len(per_cluster_score),
        "n_clusters_excluded_nondiscriminative": len(excluded),
        "per_cluster_score": per_cluster_score,
        "numeric_score": numeric_score,
        "raw_numeric_score": numeric_score if per_cluster_score else None,
        "note": None if per_cluster_score else "no discriminative clusters scored",
    }


def compute_numeric_score_flat(sub_result: dict, ref_metrics: dict, meta: dict) -> dict:
    """Type I reference-relative score — one number (no clusters).

    score = clip(1 - 0.5 * sub / ref, 0, 1) on the flat test set.
    Submission failed -> score = 0. Reference (near-)perfect -> score 0
    with a note (task degenerate for scoring).
    """
    metric = task_metric(meta)
    best_id, why = _best_reference(ref_metrics, metric)
    if best_id is None:
        return {"metric": metric, "numeric_score": 0.0, "raw_numeric_score": None,
                "note": why}
    ref_v = ref_metrics["baselines"][best_id]["metrics"][metric]

    note = None
    if _ref_nondiscriminative(ref_v, metric):
        numeric_score = 0.0
        note = "best reference is nondiscriminative / near-perfect"
    elif sub_result["failed"] or sub_result["metrics"] is None:
        numeric_score = 0.0
    else:
        numeric_score = _score_one_cluster(sub_result["metrics"][metric], ref_v, metric)

    return {
        "metric": metric,
        "best_reference_id": best_id,
        "numeric_score": numeric_score,
        "raw_numeric_score": None if note else numeric_score,
        "note": note,
    }


def score_one(mod, label: str, data: dict, ref_metrics: dict, meta: dict,
              ref_registry: dict, *, exclude_self_from_baselines: bool = False) -> dict:
    """Score one formula and return the deterministic numeric score.

    `data` is a flat dict (Type I) or clusters dict (Type II).
    `ref_registry` is kept for the self-test call signature; numeric scoring
    uses the frozen reference metrics, not the reference formula source.

    Set `exclude_self_from_baselines=True` when self-testing a reference
    baseline as if it were a submission.
    """
    caps = ref_metrics.get("derived_caps", {})
    errs = validate_contract(mod, caps)
    if errs:
        return _zero_score_result(
            label,
            meta,
            "contract_fail",
            contract_ok=False,
            violations=errs,
            note="contract violation",
        )

    task_type = meta.get("type", "typeII")
    target_name = meta["target"]["name"]

    if task_type == "typeI":
        # Type I — no fit(), deterministic: a single run.
        sub_result = run_formula_flat(mod, data, target_name)
        score = compute_numeric_score_flat(sub_result, ref_metrics, meta)
        numeric_per_seed = [score["numeric_score"]]
        raw_numeric_per_seed = [score.get("raw_numeric_score")]
        exec_info = {"failed": sub_result["failed"], "error": sub_result["error"],
                     "n_seeds": 1}
    else:
        # Type II — fit() may be stochastic: run N_SEEDS fixed seeds (Scheme B),
        # report mean / std of numeric_score. A deterministic fit() → std 0.
        sub_result = None
        score = None
        numeric_per_seed = []
        raw_numeric_per_seed = []
        for k in range(N_SEEDS):
            sr = run_formula(mod, data, target_name,
                             fit_timeout_seconds=caps.get("fit_timeout_seconds"),
                             seed=BASE_SEED + k)
            sc = compute_numeric_score(sr, ref_metrics, meta)
            numeric_per_seed.append(sc["numeric_score"])
            raw_numeric_per_seed.append(sc.get("raw_numeric_score"))
            if sub_result is None:
                sub_result, score = sr, sc          # detail reported from the first seed
        exec_info = {"n_clusters_fitted": sub_result["n_clusters_fitted"],
                     "n_clusters_failed": sub_result["n_clusters_failed"],
                     "n_seeds": N_SEEDS}

    numeric = float(np.mean(numeric_per_seed)) if numeric_per_seed else 0.0
    numeric_std = float(np.std(numeric_per_seed)) if numeric_per_seed else 0.0
    raw_numeric_vals = [v for v in raw_numeric_per_seed if isinstance(v, (int, float))]
    raw_numeric = float(np.mean(raw_numeric_vals)) if raw_numeric_vals else None

    # Raw-metric summary scoring is based on: Type I → the flat-set metric;
    # Type II → the mean per-cluster metric (no cross-cluster pooling).
    metric = task_metric(meta)
    if task_type == "typeI":
        fm = sub_result["metrics"]
        raw_metric = fm[metric] if fm and fm.get(metric) is not None else None
    else:
        vals = [c["metrics"][metric] for c in sub_result["per_cluster"].values()
                if c["metrics"] and c["metrics"].get(metric) is not None]
        raw_metric = float(np.mean(vals)) if vals else None

    failed = bool(sub_result["failed"]) if task_type == "typeI" \
        else exec_info["n_clusters_fitted"] == 0 and exec_info["n_clusters_failed"] > 0

    return {
        "submission": label,
        "contract_ok": True,
        "status": "exec_error" if failed else "ok",
        **exec_info,
        "raw_metric": raw_metric,
        "score": score,
        "numeric_score": numeric,
        "raw_numeric_score": raw_numeric,
        "numeric_score_std": numeric_std,
        "numeric_score_per_seed": numeric_per_seed,
        "raw_numeric_score_per_seed": raw_numeric_per_seed,
    }


def mode_score(task_dir: Path, submission: Path | None) -> int:
    meta = load_task(task_dir)
    task_type = meta.get("type", "typeII")
    ref_path = reference_metrics_path(task_dir)   # private scoring/ tree
    if not ref_path.exists():
        raise SystemExit(f"{ref_path} missing — run `evaluate_numeric.py reference {task_dir}` first.")
    ref_metrics = json.load(ref_path.open())

    data = load_flat(task_dir) if task_type == "typeI" else load_clusters(task_dir)

    if submission is not None:
        # numeric_score needs only reference_metrics.json (the anchors) — the
        # reference baseline .py files are NOT shipped, so pass an empty registry
        # (it was only used by the removed LLM judge).
        try:
            mod = load_submission(submission.resolve())
            result = score_one(mod, submission.name, data, ref_metrics, meta, {})
        except Exception as exc:
            result = _zero_score_result(
                submission.name,
                meta,
                "compile_error",
                contract_ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    # Self-test needs the reference baseline .py files, which are NOT included in
    # the release (only the scoring/ anchors). Pass a submission instead.
    try:
        registry = load_task_registry(task_dir)
    except Exception:
        print("[self-test unavailable] reference baseline formulas are not "
              "shipped (only scoring/.../reference_metrics.json).\n"
              "Score a submission instead:  evaluate_numeric.py score <task_dir> <sub.py>")
        return 0

    # Self-test: score each reference baseline as a submission.
    metric = task_metric(meta)
    raw_hdr = metric if task_type == "typeI" else f"mean-{metric}"
    print(f"[{meta['task_id']}] ({task_type}) self-test — each reference baseline "
          f"scored as a submission:\n")
    print(f"  {'submission':<26} {raw_hdr:>13} {'numeric':>9} {'±std':>8}")
    for stem in sorted(registry):
        result = score_one(registry[stem], stem, data, ref_metrics, meta, registry,
                           exclude_self_from_baselines=True)
        if not result["contract_ok"]:
            print(f"  {stem:<26}  CONTRACT FAIL: {result['violations']}")
            continue
        rv = result.get("raw_metric")
        pv = f"{rv:>13.4f}" if isinstance(rv, (int, float)) else f"{'FAILED':>13}"
        ns = result["numeric_score"]
        ns_s = f"{ns:>9.4f}" if ns is not None else f"{'—':>9}"
        sd = result["numeric_score_std"]
        sd_s = f"{sd:>8.4f}" if sd is not None else f"{'—':>8}"
        print(f"  {stem:<26} {pv} {ns_s} {sd_s}")
    best_id, _ = _best_reference(ref_metrics, metric)
    print(f"\n  (best reference baseline = {best_id} → expect its numeric ≈ 0.5)")
    return 0


# ==========================================================================
# CLI
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="RealSR numeric scoring harness")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_ref = sub.add_parser("reference", help="build reference anchors for a task")
    p_ref.add_argument("task_dir", type=Path)

    p_sc = sub.add_parser("score", help="score a submission (or self-test)")
    p_sc.add_argument("task_dir", type=Path)
    p_sc.add_argument("submission", type=Path, nargs="?", default=None)

    args = ap.parse_args()
    if args.mode == "reference":
        return mode_reference(args.task_dir.resolve())
    return mode_score(args.task_dir.resolve(), args.submission)


if __name__ == "__main__":
    raise SystemExit(main())
