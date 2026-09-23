"""eval_formula.py — the shared evaluation core.

Runs ONE formula module over the test set and returns raw metrics:
per-cluster (Type II) or on the flat test set (Type I). This is the common
engine called by evaluate_numeric.py (on each reference baseline and on a
submitted formula.py).

A formula module must expose the v2 contract:
  USED_INPUTS, LAW_CONSTANTS, OTHER_CONSTANTS, LOCAL_FITTABLE,
  predict(X, **params)   and (Type II) fit(X_fit, y_fit, **law_constants).

`run_formula` performs NO score normalisation and NO judging — it only
executes the formula and measures error. The reference-relative score
and the scoring channel lives in evaluate_numeric.py.
"""
from __future__ import annotations

import csv
import random
import signal
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [list(r) for r in reader]
    return header, rows


def group_by_cluster(header: list[str], rows: list[list[str]]) -> dict[int, list[list[str]]]:
    gid = header.index("group_id")
    out: dict[int, list[list[str]]] = {}
    for r in rows:
        out.setdefault(int(float(r[gid])), []).append(r)
    return out


def load_clusters(task_root: Path) -> dict:
    """Type II loader. Load test_fit.csv + test_test.csv, grouped by cluster.

    Returns a dict with: fit_header, test_header, fit_by_cluster,
    test_by_cluster, cluster_ids (sorted, present in both).
    """
    fit_header,  fit_rows  = load_csv(task_root / "data" / "test_fit.csv")
    test_header, test_rows = load_csv(task_root / "data" / "test_test.csv")
    fit_by_cluster  = group_by_cluster(fit_header,  fit_rows)
    test_by_cluster = group_by_cluster(test_header, test_rows)
    cluster_ids = sorted(set(fit_by_cluster) & set(test_by_cluster))
    return {
        "fit_header": fit_header, "test_header": test_header,
        "fit_by_cluster": fit_by_cluster, "test_by_cluster": test_by_cluster,
        "cluster_ids": cluster_ids,
    }


def load_flat(task_root: Path) -> dict:
    """Type I loader. Load train.csv + test.csv as flat tables (no clusters).

    The reference / submission formulas predict directly on test.csv.
    train.csv is carried for completeness (the SR system trains on it) but
    the v2 score does not use it — there is no naive baseline.
    """
    train_header, train_rows = load_csv(task_root / "data" / "train.csv")
    test_header,  test_rows  = load_csv(task_root / "data" / "test.csv")
    return {
        "train_header": train_header, "train_rows": train_rows,
        "test_header": test_header,   "test_rows": test_rows,
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _to_array(rows: list[list[str]], header: list[str], cols: list[str]) -> np.ndarray:
    idx = [header.index(c) for c in cols]
    return np.array([[float(row[i]) for i in idx] for row in rows], dtype=float)


def _col(rows: list[list[str]], header: list[str], name: str) -> np.ndarray:
    i = header.index(name)
    return np.array([float(r[i]) for r in rows], dtype=float)


# --------------------------------------------------------------------------
# metric registry
# --------------------------------------------------------------------------
# A task selects exactly ONE metric via metadata `metric:`. Each entry
# carries the compute fn, the optimisation direction, and the value a
# perfect prediction attains — the reference-relative score in evaluate_numeric.py
# needs all three. Higher-is-better metrics must be bounded above by
# `perfect`. Each compute fn receives finite-masked, equal-length
# (y_true, y_pred) with n >= 1, and returns a float or None (undefined for
# this data — e.g. mape when y_true has zeros).

def _mse(yt, yp):
    return float(np.mean((yp - yt) ** 2))


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def _mae(yt, yp):
    return float(np.mean(np.abs(yp - yt)))


def _mdae(yt, yp):
    return float(np.median(np.abs(yp - yt)))


def _r2(yt, yp):
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return None                               # constant truth — r2 undefined
    return float(1.0 - float(np.sum((yp - yt) ** 2)) / ss_tot)


def _smape(yt, yp):
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    safe = denom > 0
    if not safe.any():
        return None
    return float(np.mean(np.abs(yp[safe] - yt[safe]) / denom[safe]))


def _mape(yt, yp):
    nz = yt != 0
    if not nz.any():
        return None                               # all-zero truth — mape undefined
    return float(np.mean(np.abs((yp[nz] - yt[nz]) / yt[nz])))


def _log_mae(yt, yp):
    # log10-space error — for strictly-positive targets spanning decades.
    pos = yt > 0
    if not pos.any():
        return None
    yp_c = np.clip(yp[pos], 1e-300, None)         # non-positive prediction → huge log error
    return float(np.mean(np.abs(np.log10(yp_c) - np.log10(yt[pos]))))


METRICS: dict[str, dict] = {
    "rmse":    {"fn": _rmse,    "direction": "lower",  "perfect": 0.0},
    "mae":     {"fn": _mae,     "direction": "lower",  "perfect": 0.0},
    "mse":     {"fn": _mse,     "direction": "lower",  "perfect": 0.0},
    "mdae":    {"fn": _mdae,    "direction": "lower",  "perfect": 0.0},
    "smape":   {"fn": _smape,   "direction": "lower",  "perfect": 0.0},
    "mape":    {"fn": _mape,    "direction": "lower",  "perfect": 0.0},
    "log_mae": {"fn": _log_mae, "direction": "lower",  "perfect": 0.0},
    "r2":      {"fn": _r2,      "direction": "higher", "perfect": 1.0},
}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the full metric registry on one (y_true, y_pred) pair.

    Returns {metric_name: value|None, ..., "n_finite": int}. A task's
    declared metric is one key; the rest are kept for diagnostics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    n = int(mask.sum())
    if n == 0:
        return {**{name: None for name in METRICS}, "n_finite": 0}
    yt, yp = y_true[mask], y_pred[mask]
    out: dict = {}
    for name, spec in METRICS.items():
        try:
            out[name] = spec["fn"](yt, yp)
        except Exception:  # noqa: BLE001
            out[name] = None
    out["n_finite"] = n
    return out


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _Timeout()


# A law maps one input row to one prediction: predict(X)[i] may depend only on
# X[i]. Re-evaluating sampled rows one at a time catches submissions that
# sort / difference / aggregate across the test batch (a row permutation test
# would miss those that sort internally and restore the order).
ROW_PROBE_MAX = 32
ROW_PROBE_RTOL = 1e-4
ROW_PROBE_ATOL = 1e-9


def check_row_independence(predict_call, X: np.ndarray, y_pred: np.ndarray,
                           seed: int = 0) -> str | None:
    """Return None if sampled rows predict the same alone as in the batch,
    else a short description of the first violation."""
    n = len(X)
    if n <= 1:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(ROW_PROBE_MAX, n), replace=False)
    for i in idx:
        try:
            yi = np.asarray(predict_call(X[i:i + 1]), dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return f"predict() fails on a single row ({type(exc).__name__}: {exc})"
        if yi.shape != (1,):
            return f"predict() on a single row returned shape {yi.shape}"
        ref = float(y_pred[i])
        if not np.isclose(yi[0], ref, rtol=ROW_PROBE_RTOL,
                          atol=ROW_PROBE_ATOL * max(1.0, abs(ref))):
            return (f"row {int(i)}: predict alone = {yi[0]:.6g} but in the batch "
                    f"= {ref:.6g}; output depends on other rows")
    return None


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------

def run_formula(mod, clusters: dict, target_name: str,
                fit_timeout_seconds: int | None = None,
                seed: int | None = None) -> dict:
    """Execute one formula module over every test cluster.

    `seed`, if given, fixes the global NumPy / Python RNG before the run so
    a stochastic submission `fit()` is reproducible. evaluate_numeric.py runs each
    Type II submission under several seeds and reports mean / std.

    Returns:
      {
        "per_cluster": {cid: {"metrics": {...}, "failed": bool, "error": str|None}},
        "n_clusters_fitted": int,
        "n_clusters_failed": int,
        "max_fit_seconds": float,   # slowest per-cluster fit() wall-time
      }

    A cluster is `failed` if fit() raises / times out, or predict() returns
    non-finite. The score is computed per-cluster and averaged in
    evaluate_numeric.py — there is no cross-cluster pooling. `max_fit_seconds` lets
    evaluate_numeric.py derive the fit_timeout cap from the reference bank's
    measured fit cost.
    """
    fit_header  = clusters["fit_header"]
    test_header = clusters["test_header"]
    fit_by_cluster  = clusters["fit_by_cluster"]
    test_by_cluster = clusters["test_by_cluster"]
    cluster_ids = clusters["cluster_ids"]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    used = list(mod.USED_INPUTS)
    LAW  = dict(mod.LAW_CONSTANTS)
    is_type_ii = bool(mod.LOCAL_FITTABLE)

    per_cluster: dict[int, dict] = {}
    n_failed = 0
    max_fit_seconds = 0.0
    row_dependence: str | None = None

    for cid in cluster_ids:
        fr = fit_by_cluster[cid]
        tr = test_by_cluster[cid]
        try:
            if used:
                X_fit  = _to_array(fr, fit_header,  used)
                X_test = _to_array(tr, test_header, used)
            else:
                X_fit  = np.zeros((len(fr), 0), dtype=float)
                X_test = np.zeros((len(tr), 0), dtype=float)
            y_fit  = _col(fr, fit_header,  target_name)
            y_test = _col(tr, test_header, target_name)

            if is_type_ii:
                if fit_timeout_seconds:
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(int(fit_timeout_seconds))
                t0 = time.perf_counter()
                try:
                    local = mod.fit(X_fit, y_fit, **LAW)
                finally:
                    if fit_timeout_seconds:
                        signal.alarm(0)
                max_fit_seconds = max(max_fit_seconds, time.perf_counter() - t0)
            else:
                local = {}

            y_pred = np.asarray(mod.predict(X_test, **LAW, **local), dtype=float)
            if not np.all(np.isfinite(y_pred)):
                raise RuntimeError("predict returned non-finite values")
            if row_dependence is None:
                row_dependence = check_row_independence(
                    lambda Xr: mod.predict(Xr, **LAW, **local), X_test, y_pred,
                    seed=int(cid))

            m = metrics(y_test, y_pred)
            per_cluster[cid] = {"metrics": m, "failed": False, "error": None}
        except _Timeout:
            n_failed += 1
            per_cluster[cid] = {"metrics": None, "failed": True,
                                "error": f"fit() exceeded {fit_timeout_seconds}s"}
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            per_cluster[cid] = {"metrics": None, "failed": True,
                                "error": f"{type(exc).__name__}: {exc}"}

    return {
        "per_cluster": per_cluster,
        "n_clusters_fitted": len(cluster_ids) - n_failed,
        "n_clusters_failed": n_failed,
        "max_fit_seconds": max_fit_seconds,
        "row_dependence": row_dependence,
    }


# --------------------------------------------------------------------------
# core — Type I (flat, no clusters, no fit)
# --------------------------------------------------------------------------

def run_formula_flat(mod, flat: dict, target_name: str) -> dict:
    """Execute one Type I formula on the flat test set.

    Type I: LOCAL_FITTABLE is empty, there is no fit() — predict() is called
    once on the whole test set with only LAW_CONSTANTS.

    Returns:
      {"metrics": {...} | None, "failed": bool, "error": str|None}
    """
    test_header = flat["test_header"]
    test_rows   = flat["test_rows"]
    used = list(mod.USED_INPUTS)
    LAW  = dict(mod.LAW_CONSTANTS)

    try:
        X_test = (_to_array(test_rows, test_header, used) if used
                  else np.zeros((len(test_rows), 0), dtype=float))
        y_test = _col(test_rows, test_header, target_name)
        y_pred = np.asarray(mod.predict(X_test, **LAW), dtype=float)
        if not np.all(np.isfinite(y_pred)):
            raise RuntimeError("predict returned non-finite values")
        row_dependence = check_row_independence(
            lambda Xr: mod.predict(Xr, **LAW), X_test, y_pred)
        return {"metrics": metrics(y_test, y_pred), "failed": False, "error": None,
                "row_dependence": row_dependence}
    except Exception as exc:  # noqa: BLE001
        return {"metrics": None, "failed": True,
                "error": f"{type(exc).__name__}: {exc}", "row_dependence": None}
