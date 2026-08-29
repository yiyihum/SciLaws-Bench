"""Single-file simulator runtime for the v3 benchmark.

Task simulator directories contain `state.joblib` and `sample.csv`. The public
entry point is `load(sim_dir)`, which selects the Type I or Type II runtime
from the saved schema.

Some `state.joblib` files pickle `ResidualBootstrap` under the historical
`data_aug.sim2.bootstrap` path. This module registers that path as an alias to
itself so those states unpickle without a `data_aug` dependency.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
import pandas as pd


DEFAULT_MAX_ROWS = 50
DEFAULT_OVERSAMPLE = 20
HARD_LIMIT = 5000
FLOOR_FOR_LOG = 1e-12


def safe_float(value: Any) -> Optional[float]:
    """Convert finite numeric values to plain Python floats for JSON-like output."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def module_from_source(source: str, module_name: str, source_name: str):
    module = types.ModuleType(module_name)
    exec(compile(source, source_name, "exec"), module.__dict__)
    return module


def load_formula_file(task_dir: Path, formula_id: str, module_name: str):
    formula_path = task_dir / "formulas" / f"{formula_id}.py"
    spec = importlib.util.spec_from_file_location(module_name, formula_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load formula module from {formula_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_id_key(value: Any) -> str:
    """Normalize user-supplied group ids to the string keys stored in state."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return str(value)


def group_id_value(key: str) -> Any:
    """Return a compact JSON-like group id value for output rows."""
    try:
        return int(key)
    except ValueError:
        return key


def _transform_axes(X: np.ndarray, x_axes: list[str]) -> np.ndarray:
    """Apply per-column log10 to 'log10' axes; leave 'linear' columns alone."""
    Xt = np.array(X, dtype=float, copy=True)
    for j, ax in enumerate(x_axes):
        if ax == "log10":
            Xt[:, j] = np.log10(np.maximum(Xt[:, j], FLOOR_FOR_LOG))
    return Xt


class ResidualBootstrap:
    """k-NN residual resampler over real data.

    For a query point X*, draw a residual from the empirical residuals of the
    nearest real data points in standardized input space.
    """

    def __init__(self, x_axes: list[str], k_frac: float = 0.15,
                 k_min: int = 8, k_max: int = 64):
        self.x_axes = list(x_axes)
        self.k_frac = float(k_frac)
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self._Xs: np.ndarray | None = None
        self._res: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._tree = None

    def fit(self, X_real: np.ndarray, residuals: np.ndarray) -> "ResidualBootstrap":
        X_real = np.atleast_2d(np.asarray(X_real, dtype=float))
        res = np.asarray(residuals, dtype=float).ravel()
        finite = np.isfinite(res) & np.all(np.isfinite(X_real), axis=1)
        X_real, res = X_real[finite], res[finite]
        if len(res) == 0:
            raise ValueError("no finite (X, residual) pairs to fit bootstrap")
        Xt = _transform_axes(X_real, self.x_axes)
        self._mean = Xt.mean(axis=0)
        self._std = Xt.std(axis=0)
        self._std[self._std == 0] = 1.0
        self._Xs = (Xt - self._mean) / self._std
        self._res = res
        self._tree = None
        return self

    @property
    def n(self) -> int:
        return 0 if self._res is None else len(self._res)

    def _k(self) -> int:
        return int(np.clip(round(self.n * self.k_frac),
                           min(self.k_min, self.n), min(self.k_max, self.n)))

    def _ensure_tree(self):
        if self._tree is None and self.n >= self.k_min:
            try:
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self._Xs)
            except Exception:
                self._tree = "brute"
        return self._tree

    def _neighbour_idx(self, Xq_s: np.ndarray, k: int) -> np.ndarray:
        """Return (m, k) neighbour indices for standardized queries Xq_s."""
        tree = self._ensure_tree()
        if tree is not None and tree != "brute":
            _, idx = tree.query(Xq_s, k=k)
            return np.atleast_2d(idx).reshape(len(Xq_s), k)
        out = np.empty((len(Xq_s), k), dtype=int)
        for i, q in enumerate(Xq_s):
            d = np.sum((self._Xs - q) ** 2, axis=1)
            out[i] = np.argpartition(d, min(k, self.n) - 1)[:k]
        return out

    def sample(self, X_query: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draw one residual per query row with replacement from its k-NN."""
        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        m = len(X_query)
        if self.n < self.k_min:
            return self._res[rng.integers(0, self.n, size=m)]
        Xt = _transform_axes(X_query, self.x_axes)
        Xq_s = (Xt - self._mean) / self._std
        k = self._k()
        nbr = self._neighbour_idx(Xq_s, k)
        pick = rng.integers(0, k, size=m)
        chosen = nbr[np.arange(m), pick]
        return self._res[chosen]

    def predictive_interval(self, X_query: np.ndarray, lo: float = 0.1,
                            hi: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Per-query residual quantiles over the local neighbourhood."""
        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        m = len(X_query)
        if self.n < self.k_min:
            ql, qh = np.quantile(self._res, [lo, hi])
            return np.full(m, ql), np.full(m, qh)
        Xt = _transform_axes(X_query, self.x_axes)
        Xq_s = (Xt - self._mean) / self._std
        nbr = self._neighbour_idx(Xq_s, self._k())
        local = self._res[nbr]
        return (np.quantile(local, lo, axis=1),
                np.quantile(local, hi, axis=1))

    def local_std(self, X_query: np.ndarray) -> np.ndarray:
        """Per-query residual std over the local neighbourhood."""
        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        m = len(X_query)
        if self.n < self.k_min:
            return np.full(m, float(np.std(self._res)))
        Xt = _transform_axes(X_query, self.x_axes)
        Xq_s = (Xt - self._mean) / self._std
        nbr = self._neighbour_idx(Xq_s, self._k())
        return np.std(self._res[nbr], axis=1)

    def local_mean(self, X_query: np.ndarray) -> np.ndarray:
        """Per-query mean residual over the local neighbourhood."""
        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        m = len(X_query)
        if self.n < self.k_min:
            return np.full(m, float(np.mean(self._res)))
        Xt = _transform_axes(X_query, self.x_axes)
        Xq_s = (Xt - self._mean) / self._std
        nbr = self._neighbour_idx(Xq_s, self._k())
        return np.mean(self._res[nbr], axis=1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tree"] = None
        return state


def _register_pickle_alias() -> None:
    data_aug = sys.modules.get("data_aug")
    if data_aug is None:
        data_aug = types.ModuleType("data_aug")
        data_aug.__path__ = []
        sys.modules["data_aug"] = data_aug

    sim2 = sys.modules.get("data_aug.sim2")
    if sim2 is None:
        sim2 = types.ModuleType("data_aug.sim2")
        sim2.__path__ = []
        sys.modules["data_aug.sim2"] = sim2

    sys.modules.setdefault("data_aug.sim2.bootstrap", sys.modules[__name__])
    setattr(data_aug, "sim2", sim2)
    setattr(sim2, "bootstrap", sys.modules["data_aug.sim2.bootstrap"])


_register_pickle_alias()


@dataclass
class TypeISimulatorBundle:
    fetch_where: Callable
    fetch_data: Callable
    load_sample: Callable
    tool_description: Callable
    formula_info: Callable
    budget_status: Callable


def _load_typeI_formula(task_dir: Path, formula_id: str):
    return load_formula_file(
        task_dir, formula_id, f"_sim2rt_{task_dir.name}_{formula_id}"
    )


def _build_typeI_simulator(wrapper_path: str | Path) -> TypeISimulatorBundle:
    sim_dir = Path(wrapper_path).resolve().parent
    formula_id = sim_dir.name
    task_dir = sim_dir.parent.parent
    state_path = sim_dir / "state.joblib"
    sample_csv = sim_dir / "sample.csv"

    @lru_cache(maxsize=1)
    def _state() -> Dict[str, Any]:
        return joblib.load(state_path)

    @lru_cache(maxsize=1)
    def _F():
        src = _state().get("formula_source")
        if src:
            return module_from_source(
                src,
                f"_sim2rt_src_{formula_id}",
                f"<formula_source:{formula_id}>",
            )
        return _load_typeI_formula(task_dir, formula_id)

    @lru_cache(maxsize=1)
    def _real_df() -> pd.DataFrame:
        return pd.DataFrame(_state()["real_rows"])

    budget = {"used": 0}

    def _remaining() -> int:
        return max(0, int(_state()["fetch_budget_rows"]) - budget["used"])

    def budget_status() -> Dict[str, Any]:
        s = _state()
        return {"fetch_budget_rows": int(s["fetch_budget_rows"]),
                "used": budget["used"], "remaining": _remaining()}

    def _generate(X_used: np.ndarray, rng) -> np.ndarray:
        s = _state()
        F = _F()
        space = s["residual_space"]
        ns = float(s["noise_scale"])
        law = dict(s["law_constants"])
        y_clean = np.asarray(F.predict(X_used, **law))
        if np.iscomplexobj(y_clean):
            y_clean = np.real(y_clean)
        y_clean = y_clean.astype(float)
        r = s["bootstrap"].sample(X_used, rng) * ns if ns != 0 else np.zeros(len(X_used))
        if space == "log":
            floor = s.get("p_min_floor") or FLOOR_FOR_LOG
            lc = np.log10(np.maximum(y_clean, floor))
            return np.maximum(10.0 ** (lc + r), floor)
        return y_clean + r

    def load_sample() -> pd.DataFrame:
        return pd.read_csv(sample_csv)

    def fetch_data(seed: Optional[int] = None, n_samples: int = 1,
                   **input_vals) -> Dict[str, Any]:
        s = _state()
        used = s["used_inputs"]
        support = s["support"]
        target = s["target"]
        unknown = [k for k in input_vals if k not in used]
        if unknown:
            return {"error": f"unknown input(s) {unknown}; valid inputs: {used}"}

        arrays = {}
        for c in used:
            v = input_vals.get(c)
            if v is None:
                return {"error": f"missing input {c!r}; pass {c}=[...]"}
            try:
                arrays[c] = np.asarray(v, dtype=float).ravel()
            except Exception as e:
                return {"error": f"could not parse {c}: {type(e).__name__}: {e}"}

        n_pts = len(arrays[used[0]])
        if any(len(a) != n_pts for a in arrays.values()):
            return {"error": "all input arrays must have equal length"}
        if n_pts == 0:
            return {"source": "synthetic", "rows": [], "n_returned": 0, "n_clipped": 0,
                    "budget": budget_status()}

        n_per = max(1, int(n_samples))
        total = n_pts * n_per
        if _remaining() <= 0:
            return {"error": "fetch budget exhausted", "budget": budget_status()}
        if total > _remaining():
            return {"error": f"request ({total} rows) exceeds remaining fetch "
                             f"budget ({_remaining()})", "budget": budget_status()}

        n_clipped = 0
        for c in used:
            lo, hi = support[c]["min"], support[c]["max"]
            a = arrays[c]
            n_clipped += int(((a < lo) | (a > hi)).sum())
            arrays[c] = np.clip(a, lo, hi)

        X = np.repeat(np.stack([arrays[c] for c in used], axis=1), n_per, axis=0)
        rng = np.random.default_rng(seed)
        y = _generate(X, rng)
        budget["used"] += total

        rows = [{target: safe_float(y[i]),
                 **{c: safe_float(X[i, j]) for j, c in enumerate(used)}}
                for i in range(len(X))]
        return {"source": "synthetic", "n_returned": len(rows), "n_clipped": n_clipped,
                "rows": rows, "budget": budget_status()}

    def fetch_where(query: str, limit: Optional[int] = None,
                    max_rows_per_request: int = DEFAULT_MAX_ROWS,
                    seed: Optional[int] = None,
                    oversample: int = DEFAULT_OVERSAMPLE) -> Dict[str, Any]:
        s = _state()
        used = s["used_inputs"]
        target = s["target"]
        all_in = s["all_input_cols"]
        if not isinstance(query, str) or not query.strip():
            return {"error": "missing 'query' expression"}

        cap = max_rows_per_request if limit is None else min(int(limit), max_rows_per_request)
        cap = max(1, min(cap, HARD_LIMIT))
        if _remaining() <= 0:
            return {"error": "fetch budget exhausted", "budget": budget_status()}
        cap = min(cap, _remaining())
        n_draw = min(cap * max(1, int(oversample)), HARD_LIMIT, len(_real_df()) * 50 or HARD_LIMIT)

        rng = np.random.default_rng(seed)
        real = _real_df()
        idx = rng.integers(0, len(real), size=n_draw)
        pool = real.iloc[idx].reset_index(drop=True).copy()
        X_used = pool[used].to_numpy(float)
        pool[target] = _generate(X_used, rng)
        pool = pool[[target] + [c for c in all_in]]

        try:
            matched = pool.query(query)
        except Exception as e:
            return {"error": f"query failed: {type(e).__name__}: {e}",
                    "available_columns": list(pool.columns)}

        n_match = int(len(matched))
        matched = matched.iloc[:cap]
        budget["used"] += int(len(matched))

        out = {"where_evaluated": query, "n_matching_in_pool": n_match,
               "n_returned": int(len(matched)),
               target: [safe_float(v) for v in matched[target]],
               **{c: [safe_float(v) for v in matched[c]] for c in all_in},
               "budget": budget_status()}
        if n_match > cap:
            out["_truncated_to"] = cap
        return out

    def formula_info() -> Dict[str, Any]:
        s = _state()
        return {"formula_id": s.get("formula_id"), "target": s["target"],
                "metric": s["metric"], "used_inputs": s["used_inputs"],
                "formula_source": s.get("formula_source"),
                "formula_doc": s.get("formula_doc"),
                "equation_loc": s.get("equation_loc"),
                "paper_ref": s.get("paper_ref"),
                "law_constants": s.get("law_constants"),
                "phase": "typeI"}

    def tool_description() -> str:
        s = _state()
        used = s["used_inputs"]
        support = s["support"]
        target = s["target"]
        lines = [
            "Synthetic data oracle for this task. Two fetch tools, both metered "
            f"against a finite budget of {int(s['fetch_budget_rows'])} total rows:",
            "  fetch_data(**inputs, n_samples=1, seed=None) — evaluate at explicit "
            "input points (pass each input as a list).",
            "  fetch_where(query, limit=20) — query a fresh synthetic pool "
            f"(pandas df.query over [{', '.join(used + [target])}]).",
            "  load_sample() — a fixed generator sample (free, does not cost budget).",
            "Queryable input ranges:",
        ]
        for c in used:
            lines.append(f"  {c} ∈ [{support[c]['min']:.4g}, {support[c]['max']:.4g}]")
        lines.append("Each fetch is a FRESH draw: formula(X) + a residual resampled "
                     "from nearby real data (heteroscedastic, non-Gaussian).")
        return "\n".join(lines)

    return TypeISimulatorBundle(
        fetch_where=fetch_where,
        fetch_data=fetch_data,
        load_sample=load_sample,
        tool_description=tool_description,
        formula_info=formula_info,
        budget_status=budget_status,
    )


@dataclass
class TypeIISimulatorBundle:
    list_groups: Callable
    fetch_data: Callable
    fetch_where: Callable
    load_sample: Callable
    budget_status: Callable
    tool_description: Callable
    form_info: Callable


def _load_typeII_form(task_dir: Path, form_id: str):
    return load_formula_file(task_dir, form_id, f"_t2rt_{form_id}")


def _build_typeII_simulator(wrapper_path: str | Path) -> TypeIISimulatorBundle:
    sim_dir = Path(wrapper_path).resolve().parent
    form_id = sim_dir.name
    task_dir = sim_dir.parent.parent
    state_path = sim_dir / "state.joblib"
    sample_csv = sim_dir / "sample.csv"

    @lru_cache(maxsize=1)
    def _state():
        return joblib.load(state_path)

    @lru_cache(maxsize=1)
    def _F():
        src = _state().get("formula_source")
        if src:
            return module_from_source(
                src,
                f"_t2rt_src_{form_id}",
                f"<formula_source:{form_id}>",
            )
        return _load_typeII_form(task_dir, form_id)

    budget = {"used": 0}

    def _remaining():
        return max(0, int(_state()["fetch_budget_rows"]) - budget["used"])

    def budget_status():
        s = _state()
        return {"fetch_budget_rows": int(s["fetch_budget_rows"]), "used": budget["used"],
                "remaining": _remaining()}

    def _gen(gid: str, X_used: np.ndarray, rng) -> np.ndarray:
        s = _state()
        F = _F()
        group = s["groups"][gid]
        space = s["residual_space"]
        ns = float(group.get("noise_scale", s["noise_scale"]))
        y = np.asarray(F.predict(X_used, **group["params"]), float)
        r = group["bootstrap"].sample(X_used, rng) * ns
        if space == "log":
            floor = s.get("p_min_floor") or FLOOR_FOR_LOG
            return np.maximum(10.0 ** (np.log10(np.maximum(y, floor)) + r), floor)
        return y + r

    def list_groups():
        s = _state()
        return {"n_groups": len(s["groups"]), "used_inputs": s["used_inputs"],
                "groups": {gid: {"n": g["n"], "support": g["support"]}
                           for gid, g in s["groups"].items()}}

    def load_sample() -> pd.DataFrame:
        return pd.read_csv(sample_csv)

    def fetch_data(group_id=None, seed: Optional[int] = None,
                   n_samples: int = 1, **inputs):
        s = _state()
        used = s["used_inputs"]
        target = s["target"]
        unknown = [k for k in inputs if k not in used]
        if unknown:
            return {"error": f"unknown input(s) {unknown}; valid inputs: {used}"}

        arrays = {}
        for c in used:
            v = inputs.get(c)
            if v is None:
                return {"error": f"missing input {c!r}; pass {c}=[...]"}
            arrays[c] = np.asarray(v, float).ravel()

        n_pts = len(arrays[used[0]])
        if any(len(a) != n_pts for a in arrays.values()):
            return {"error": "all input arrays must have equal length"}
        if n_pts == 0:
            return {"group_id": group_id, "n_returned": 0, "n_clipped": 0,
                    "rows": [], "budget": budget_status()}

        if isinstance(group_id, (list, tuple, np.ndarray, pd.Series)):
            group_ids = np.asarray(group_id, dtype=object).ravel()
            if len(group_ids) != n_pts:
                return {"error": "group_id must be scalar or have one value per input point"}
            gid_keys = [group_id_key(g) for g in group_ids]
        else:
            gid = group_id_key(group_id)
            gid_keys = [gid] * n_pts

        unknown_groups = sorted({g for g in gid_keys if g not in s["groups"]})
        if unknown_groups:
            return {"error": f"unknown group_id {unknown_groups}; call list_groups()"}

        n_per = max(1, int(n_samples))
        total = n_pts * n_per
        if total > _remaining():
            return {"error": f"request ({total}) exceeds remaining budget ({_remaining()})",
                    "budget": budget_status()}

        n_clipped = 0
        clipped = {c: arrays[c].copy() for c in used}
        for i, gid in enumerate(gid_keys):
            support = s["groups"][gid]["support"]
            for c in used:
                lo, hi = support[c]["min"], support[c]["max"]
                value = clipped[c][i]
                if value < lo or value > hi:
                    n_clipped += 1
                    clipped[c][i] = np.clip(value, lo, hi)

        X = np.repeat(np.stack([clipped[c] for c in used], axis=1), n_per, axis=0)
        gids_repeated = np.repeat(np.asarray(gid_keys, dtype=object), n_per)
        y = np.empty(len(X), dtype=float)
        rng = np.random.default_rng(seed)
        for gid in sorted(set(gid_keys)):
            mask = gids_repeated == gid
            y[mask] = _gen(gid, X[mask], rng)
        budget["used"] += total

        rows = [{target: safe_float(y[i]),
                 **{c: safe_float(X[i, j]) for j, c in enumerate(used)},
                 "group_id": group_id_value(str(gids_repeated[i]))}
                for i in range(len(X))]
        return {"group_id": group_id, "n_returned": len(rows), "n_clipped": n_clipped,
                "rows": rows, "budget": budget_status()}

    def fetch_where(group_id=None, query: str = "", limit: Optional[int] = None,
                    max_rows_per_request: int = DEFAULT_MAX_ROWS,
                    seed: Optional[int] = None, oversample: int = 20):
        s = _state()
        used = s["used_inputs"]
        target = s["target"]
        all_in = s["all_input_cols"]
        gid = str(group_id)
        if gid not in s["groups"]:
            return {"error": f"unknown group_id {group_id!r}; call list_groups()"}
        if not isinstance(query, str) or not query.strip():
            return {"error": "missing 'query'"}

        cap = max(1, min(max_rows_per_request if limit is None
                         else min(int(limit), max_rows_per_request),
                         HARD_LIMIT, _remaining()))
        if cap <= 0:
            return {"error": "fetch budget exhausted", "budget": budget_status()}

        real = pd.DataFrame(s["groups"][gid]["real_rows"])
        n_draw = min(cap * max(1, int(oversample)), HARD_LIMIT, max(len(real) * 50, HARD_LIMIT))
        rng = np.random.default_rng(seed)
        pool = real.iloc[rng.integers(0, len(real), size=n_draw)].reset_index(drop=True).copy()
        pool[target] = _gen(gid, pool[used].to_numpy(float), rng)
        pool = pool[[target] + list(all_in)]

        try:
            matched = pool.query(query).iloc[:cap]
        except Exception as e:
            return {"error": f"query failed: {type(e).__name__}: {e}", "columns": list(pool.columns)}

        budget["used"] += int(len(matched))
        return {"group_id": gid, "where": query, "n_returned": int(len(matched)),
                target: [safe_float(v) for v in matched[target]],
                **{c: [safe_float(v) for v in matched[c]] for c in all_in},
                "budget": budget_status()}

    def form_info():
        s = _state()
        return {"form_id": s.get("form_id"), "formula_source": s.get("formula_source"),
                "formula_doc": s.get("formula_doc"), "equation_loc": s.get("equation_loc"),
                "paper_ref": s.get("paper_ref"),
                "group_params": {gid: g["params"] for gid, g in s["groups"].items()}}

    def tool_description():
        s = _state()
        return ("Multi-group synthetic oracle: recover the UNIVERSAL form behind all groups.\n"
                f"  {len(s['groups'])} groups, inputs {s['used_inputs']}, "
                f"budget {int(s['fetch_budget_rows'])} rows.\n"
                "  list_groups() — group ids + per-group input ranges (free).\n"
                "  fetch_data(group_id, **inputs, n_samples=1) — eval that group at points.\n"
                "  fetch_where(group_id, query, limit=20) — query a fresh pool for that group.\n"
                "  load_sample() — fixed noisy sample across all groups (free).\n"
                "Each group has its OWN local params + heteroscedastic noise; the FORM is shared.")

    return TypeIISimulatorBundle(
        list_groups,
        fetch_data,
        fetch_where,
        load_sample,
        budget_status,
        tool_description,
        form_info,
    )


def load(sim_dir: str | Path):
    """Build a simulator bundle from a task's `simulator/` directory."""
    sim_dir = Path(sim_dir).resolve()
    state_path = sim_dir / "state.joblib"
    if not state_path.exists():
        raise FileNotFoundError(f"no state.joblib under {sim_dir}")

    schema = str(joblib.load(state_path).get("schema_version", ""))
    anchor = sim_dir / "_wrapper.py"
    if "typeII" in schema:
        return _build_typeII_simulator(anchor)
    return _build_typeI_simulator(anchor)


__all__ = ["load", "ResidualBootstrap"]
