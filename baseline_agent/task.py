"""Task wrappers for the LLM-as-agent framework.

Two task types (orthogonal axes):

  Cluster axis (`has_group_id`):
    type1   single-cluster — one (X, y) population, no `group_id` column
    type2   multi-cluster — `group_id` column; test groups may not overlap with train

  Data axis (data mode):
    fix         RealSRTask: train/test read from data/train.csv + data/test.csv
    simulator   SimulatorTask: train is a lab-notebook (initial bootstrap +
                appended on each <experiment> call); test is the real
                data/test.csv.

Submission contract is identical across all 4 cells:
    def discovered_equation(X)                    # always accepted
    def discovered_equation(X, group_ids)         # type2 only
Fully numeric closed forms — no eval-time fitting; the LLM uses the agent's
<python> sandbox to fit constants and bake the values in.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# `prompts` (the fixed prompt/contract spec) lives in the sibling harness.
_HARNESS = Path(__file__).resolve().parent.parent / "harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))


GROUP_COL = "group_id"
EXPERIMENT_PREVIEW_ROWS = 10
MAX_EXPERIMENT_INPUT_POINT_FRACTION = 0.10
MAX_EXPERIMENT_SAMPLES = 3
MAX_EXPERIMENT_CALLS = 10


# Context-ablation modes (paper: Pure Numerical Discovery Without Scientific Context).
# - full              : current default — full problem_statement, real names, full var meta
# - no_var_detail     : full problem_statement, real names, var symbol/unit/description stripped
# - anonymous_names   : full problem_statement, names → x_1..x_d/y, var meta stripped
# - numeric_only      : generic problem_statement, names → x_1..x_d/y, var meta stripped
CONTEXT_MODES = ("full", "no_var_detail", "anonymous_names", "numeric_only")

GENERIC_PROBLEM_STATEMENT = (
    "You are given a tabular dataset with one numeric target column `y` "
    "and `d` numeric input columns `x_1, ..., x_d`. Discover a closed-form "
    "expression for `y` in terms of the inputs."
)


def _apply_context_mode(
    context_mode: str,
    problem_statement: str,
    target_col: str,
    target_meta: Dict[str, Any],
    input_cols: List[str],
    input_meta: List[Dict[str, Any]],
    train: pd.DataFrame,
    test: Optional[pd.DataFrame],
) -> Tuple[str, str, Dict[str, Any], List[str], List[Dict[str, Any]],
           pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, str]]]:
    """Apply the 4-arm context ablation. Returns the (possibly rewritten) prompt
    inputs plus a `view_to_real` mapping for arms that anonymise names.

    The on-disk CSV columns and ordering are unchanged; we only rewrite the
    DataFrames the agent sees. `task.input_cols` order is preserved so
    `train_arrays`/`test_arrays` keep producing X with the same column order
    as before — only the names change.
    """
    if context_mode not in CONTEXT_MODES:
        raise ValueError(
            f"unknown context_mode={context_mode!r}; expected one of {CONTEXT_MODES}"
        )
    view_to_real: Optional[Dict[str, str]] = None

    if context_mode in ("no_var_detail", "anonymous_names", "numeric_only"):
        target_meta = {"name": target_meta.get("name", target_col)}
        input_meta = [{"name": m["name"]} for m in input_meta]

    if context_mode in ("anonymous_names", "numeric_only"):
        new_input_cols = [f"x_{i+1}" for i in range(len(input_cols))]
        new_target_col = "y"
        rename_map = {**dict(zip(input_cols, new_input_cols)),
                      target_col: new_target_col}
        train = train.rename(columns=rename_map)
        if test is not None:
            test = test.rename(columns=rename_map)
        view_to_real = {new_target_col: target_col}
        for new, real in zip(new_input_cols, input_cols):
            view_to_real[new] = real
        input_cols = new_input_cols
        target_col = new_target_col
        target_meta = {"name": new_target_col}
        input_meta = [{"name": n} for n in new_input_cols]

    if context_mode == "numeric_only":
        problem_statement = GENERIC_PROBLEM_STATEMENT

    return (problem_statement, target_col, target_meta,
            input_cols, input_meta, train, test, view_to_real)


# ---- shared load / query helpers -------------------------------------------


def _load_meta(task_dir: str | Path):
    """Resolve the task dir and load + validate `metadata.yaml`.

    Returns (td, meta, target, target_col, inputs, has_group_id) — the common
    preamble shared by RealSRTask.load and SimulatorTask.load.
    """
    td = Path(task_dir).resolve()
    if not td.is_dir():
        raise FileNotFoundError(f"Task dir not found: {td}")
    meta_path = td / "metadata.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.yaml missing at {meta_path}")
    meta = yaml.safe_load(meta_path.read_text()) or {}

    target = meta.get("target") or {}
    target_col = target.get("name")
    if not target_col:
        raise ValueError(f"{meta_path}: target.name missing")
    inputs = list(meta.get("inputs") or [])
    if not inputs:
        raise ValueError(f"{meta_path}: inputs[] missing or empty")
    has_group_id = bool(meta.get("has_group_id", False))
    return td, meta, target, target_col, inputs, has_group_id


def _parse_baseline_metrics(meta: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """id -> {mae, mse, rmse, smape?} from metadata.yaml `baselines`."""
    out: Dict[str, Dict[str, float]] = {}
    for b in meta.get("baselines") or []:
        bid = b.get("id")
        tm = b.get("test_metrics") or {}
        if bid and tm:
            out[str(bid)] = {k: float(v) for k, v in tm.items()
                             if isinstance(v, (int, float))}
    return out


_METRIC_KEYS = ("mae", "mse", "rmse", "smape", "log_mae", "mape", "mdae", "r2")


def _load_reference_baselines(
    task_dir: Path,
) -> Tuple[Dict[str, Dict[str, float]], Optional[str]]:
    """v2 fallback: baselines live in `formulas/reference_metrics.json`, not in
    `metadata.yaml`. Returns ({baseline_id: {metric: value}}, metric_declared).
    Non-metric keys (e.g. n_finite) and null values are dropped."""
    p = Path(task_dir) / "formulas" / "reference_metrics.json"
    if not p.exists():
        return {}, None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}, None
    out: Dict[str, Dict[str, float]] = {}
    for bid, b in (d.get("baselines") or {}).items():
        m = (b or {}).get("metrics") or {}
        # Type II baselines carry no flat `metrics` block — scoring is per
        # cluster, so the representative value is the mean over clusters (this
        # matches the harness's per-cluster averaging and metadata best_baseline).
        if not m and isinstance((b or {}).get("per_cluster"), dict):
            import numpy as np
            pc = b["per_cluster"].values()
            for k in _METRIC_KEYS:
                col = [c[k] for c in pc
                       if isinstance(c, dict) and isinstance(c.get(k), (int, float))]
                if col:
                    m[k] = float(np.mean(col))
        vals = {k: float(v) for k, v in m.items()
                if k in _METRIC_KEYS and isinstance(v, (int, float))}
        if vals:
            out[str(bid)] = vals
    return out, d.get("metric_declared")


def _best_baseline(baseline_metrics: Dict[str, Dict[str, float]],
                   metric: str = "smape") -> Tuple[Optional[str], Optional[float]]:
    """(best_baseline_id, best_value) for `metric`.

    Lower is better for all metrics except `r2` (higher is better). Falls back
    along metric → smape → mae → rmse → mse if the requested metric is missing
    for any baseline. Returns (None, None) if none qualifies.
    """
    for m in (metric, "smape", "mae", "rmse", "mse"):
        vals = [(bid, tm[m]) for bid, tm in baseline_metrics.items() if m in tm]
        if len(vals) == len(baseline_metrics) and vals:
            pick = max if m == "r2" else min
            bid, v = pick(vals, key=lambda kv: kv[1])
            return bid, float(v)
    return None, None


def _expand_group_ids(gids: Any, n_samples: int) -> Optional[List[int]]:
    """Repeat each supplied group id `n_samples` times → flat int list, or None."""
    if gids is None:
        return None
    n_per = max(1, int(n_samples))
    return list(np.repeat(np.asarray(gids, dtype=int).ravel(), n_per))


@dataclass
class RealSRTask:
    task_dir: Path
    task_id: str
    domain: str
    problem_statement: str
    metadata: Dict[str, Any] = field(repr=False)

    train: pd.DataFrame = field(repr=False)
    test: pd.DataFrame = field(repr=False)

    target_col: str
    target_meta: Dict[str, Any] = field(repr=False)        # {symbol, unit, description, range}
    input_cols: List[str]
    input_meta: List[Dict[str, Any]] = field(repr=False)   # parallel to input_cols
    has_group_id: bool                                     # False = type1, True = type2

    # Baseline test_metrics: id -> {mae, mse, rmse, smape?}
    baseline_metrics: Dict[str, Dict[str, float]] = field(repr=False)

    # Context ablation state
    context_mode: str = "full"
    view_to_real: Optional[Dict[str, str]] = None
    real_input_cols: List[str] = field(default_factory=list, repr=False)
    real_target_col: str = ""

    # Declared headline metric (v2 reference_metrics.json `metric_declared`).
    headline_metric: str = "smape"
    # metadata `priors` — literature candidate constants (incl. distractors)
    # surfaced to the LLM (full context mode only). `_role` is stripped.
    priors: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    # opt-in: surface the per-variable test range in the prompt.
    show_test_range: bool = False

    @classmethod
    def load(
        cls,
        task_dir: str | Path,
        context_mode: str = "full",
        show_test_range: bool = False,
    ) -> "RealSRTask":
        td, meta, target, target_col, inputs, has_group_id = _load_meta(task_dir)
        input_cols = [i["name"] for i in inputs]

        train = pd.read_csv(td / "data" / "train.csv").reset_index(drop=True)
        # Type I has a flat data/test.csv. Type II is multi-cluster and ships
        # NO test.csv — it holds out whole clusters into test_fit.csv (the
        # per-cluster fit window) + test_test.csv (the held-out scoring window).
        # The official harness reads those two directly at eval time; here we
        # only need a `test` frame for prompt stats / logging (never shown to
        # the agent), so the held-out scoring window test_test.csv stands in.
        flat_test = td / "data" / "test.csv"
        if flat_test.is_file():
            test = pd.read_csv(flat_test).reset_index(drop=True)
        else:
            test = pd.read_csv(td / "data" / "test_test.csv").reset_index(drop=True)

        for df, name in [(train, "train"), (test, "test")]:
            for col in [target_col] + input_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"{name}.csv missing column {col!r}; got {list(df.columns)}"
                    )
            if has_group_id and GROUP_COL not in df.columns:
                raise ValueError(
                    f"{name}.csv missing {GROUP_COL!r} but metadata has_group_id=true"
                )

        # Drop non-numeric (categorical) input columns from the MODELLED inputs.
        # The LLM gets X as a float matrix and the harness scores predict() over
        # numeric USED_INPUTS, so a string column (e.g. a `region` label) cannot
        # be a numeric input — including it made train_arrays() crash on
        # to_numpy(float). It stays present in `train_df` (a distractor the agent
        # may inspect) but is excluded from X, the prompt's X[:, i] list, and
        # predict(). Tasks with all-numeric inputs are unaffected.
        drop = [c for c in input_cols
                if not pd.api.types.is_numeric_dtype(train[c])]
        # GROUP_COL is the cluster label, never a predictive feature — if a task's
        # metadata erroneously lists it under `inputs`, exclude it (using it as a
        # feature would be a per-cluster memorisation leak; predict() is also
        # forbidden a group_id arg by the harness contract).
        if has_group_id and GROUP_COL in input_cols and GROUP_COL not in drop:
            drop.append(GROUP_COL)
        if drop:
            input_cols = [c for c in input_cols if c not in drop]
            inputs = [m for m in inputs if m["name"] not in drop]
            print(f"[task] dropped non-feature input(s) {drop} from the modelled X "
                  f"(non-numeric kept in train_df as distractor; group_id excluded "
                  f"as cluster label); numeric inputs: {input_cols}", file=sys.stderr)

        baseline_metrics = _parse_baseline_metrics(meta)
        # v2 metadata declares the scoring metric directly (`metric:`); fall
        # back to reference_metrics.json `metric_declared`, then smape.
        headline_metric = meta.get("metric") or "smape"
        if not baseline_metrics:
            baseline_metrics, declared = _load_reference_baselines(td)
            if declared and not meta.get("metric"):
                headline_metric = declared

        real_input_cols = list(input_cols)
        real_target_col = target_col

        # v2 prompt slot is `context` (de-leaked background); fall back to the
        # legacy `problem_statement` field for older tasks that predate the move.
        problem_statement = str(meta.get("context") or meta.get("problem_statement") or "")
        (problem_statement, target_col, target, input_cols, inputs,
         train, test, view_to_real) = _apply_context_mode(
            context_mode=context_mode,
            problem_statement=problem_statement,
            target_col=target_col,
            target_meta=target,
            input_cols=input_cols,
            input_meta=inputs,
            train=train,
            test=test,
        )

        return cls(
            task_dir=td,
            task_id=meta.get("task_id", td.name),
            domain=str(meta.get("domain", "")),
            problem_statement=problem_statement,
            metadata=meta,
            train=train,
            test=test,
            target_col=target_col,
            target_meta=target,
            input_cols=input_cols,
            input_meta=inputs,
            has_group_id=has_group_id,
            baseline_metrics=baseline_metrics,
            context_mode=context_mode,
            view_to_real=view_to_real,
            real_input_cols=real_input_cols,
            real_target_col=real_target_col,
            headline_metric=headline_metric,
            priors=list(meta.get("priors") or []),
            show_test_range=show_test_range,
        )

    # ---- agent-facing -------------------------------------------------------

    def is_multi_group(self) -> bool:
        return self.has_group_id

    def get_task_prompt(self, max_turns: int = 30) -> str:
        from prompts import build_task_prompt
        return build_task_prompt(
            task_id=self.task_id,
            domain=self.domain,
            problem_statement=self.problem_statement,
            target_col=self.target_col,
            target_meta=self.target_meta,
            input_cols=self.input_cols,
            input_meta=self.input_meta,
            has_group_id=self.has_group_id,
            n_train_rows=len(self.train),
            n_test_rows=len(self.test),
            n_train_groups=int(self.train[GROUP_COL].nunique()) if self.has_group_id else 0,
            n_test_groups=int(self.test[GROUP_COL].nunique()) if self.has_group_id else 0,
            max_turns=max_turns,
            metric=self.headline_metric,
            include_test_range=self.show_test_range,
            is_simulator=False,
            caps=self.metadata.get("caps") or {},
        )

    def train_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """(X_train, y_train, group_ids_or_None)."""
        X = self.train[self.input_cols].to_numpy(dtype=float)
        y = self.train[self.target_col].to_numpy(dtype=float)
        g = (self.train[GROUP_COL].to_numpy(dtype=int)
             if self.has_group_id else None)
        return X, y, g

    def test_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        X = self.test[self.input_cols].to_numpy(dtype=float)
        y = self.test[self.target_col].to_numpy(dtype=float)
        g = (self.test[GROUP_COL].to_numpy(dtype=int)
             if self.has_group_id else None)
        return X, y, g

    # ---- baselines ----------------------------------------------------------

    def best_baseline(self, metric: str = "smape") -> Tuple[Optional[str], Optional[float]]:
        return _best_baseline(self.baseline_metrics, metric)


def _safe_float(v: Any) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return float("nan")
        return f
    except (TypeError, ValueError):
        return float("nan")


# ---- simulator-backed task -------------------------------------------------


def _resolve_simulator_dir(task_dir: Path, simulator_name: str) -> Path:
    """Resolve current flat simulator layout, with a fallback for old named dirs."""
    root = task_dir / "simulator"
    if simulator_name in ("", ".", "default", "simulator") and (root / "state.joblib").exists():
        return root
    named = root / simulator_name
    if (named / "state.joblib").exists():
        return named
    if (root / "state.joblib").exists():
        return root
    raise FileNotFoundError(
        f"simulator state missing; expected {root/'state.joblib'}"
        f" or {named/'state.joblib'}"
    )


def _load_simulator_module(task_dir: Path, simulator_name: str) -> Any:
    sim_dir = _resolve_simulator_dir(task_dir, simulator_name)
    try:
        from sim_runtime import load as load_simulator
    except Exception as e:
        raise RuntimeError(f"failed to import harness sim_runtime: {type(e).__name__}: {e}") from e
    try:
        return load_simulator(sim_dir)
    except Exception as e:
        raise FileNotFoundError(
            f"failed to load simulator from {sim_dir}: {type(e).__name__}: {e}"
        ) from e


def _simulator_state(task_dir: Path, simulator_name: str) -> Dict[str, Any]:
    """Read simulator state.joblib. Returns {} if the file is not present."""
    try:
        sim_dir = _resolve_simulator_dir(task_dir, simulator_name)
    except FileNotFoundError:
        return {}
    sj = sim_dir / "state.joblib"
    if not sj.exists():
        return {}
    try:
        import sim_runtime  # noqa: F401  registers the legacy pickle alias
        import joblib
        return dict(joblib.load(sj))
    except Exception:
        return {}


def _simulator_input_subset(task_dir: Path, simulator_name: str) -> List[str]:
    """The inputs the simulator actually varies (state['used_inputs'] or
    train_support keys)."""
    s = _simulator_state(task_dir, simulator_name)
    used = s.get("used_inputs")
    if used:
        return list(used)
    support = s.get("support")
    if support:
        return list(support.keys())
    return list((s.get("train_support") or {}).keys())


@dataclass
class SimulatorTask:
    """Simulator-backed task with an experiment-only agent-facing train log.

    The held-out test set still comes from the task data for scoring, but the
    agent does not see `data/train.csv` in simulator mode. Its `train_df`,
    `X_train`, and `y_train` start empty and grow only through `<experiment>`.
    The real train file is retained internally only to derive hidden budget caps
    and support-compatible metadata.
    """
    task_dir: Path
    task_id: str
    domain: str
    problem_statement: str
    metadata: Dict[str, Any] = field(repr=False)

    simulator: Any = field(repr=False)
    simulator_name: str
    sim_state: Dict[str, Any] = field(repr=False)        # simulator/<name>/state.json contents

    target_col: str
    target_meta: Dict[str, Any] = field(repr=False)
    input_cols: List[str]
    input_meta: List[Dict[str, Any]] = field(repr=False)
    has_group_id: bool

    baseline_metrics: Dict[str, Dict[str, float]] = field(repr=False)

    # Real test set (data/test.csv).
    _test_df: pd.DataFrame = field(repr=False)
    # Lab notebook: starts empty and grows on each <experiment>.
    _train_df: pd.DataFrame = field(repr=False)
    _real_train_df: pd.DataFrame = field(repr=False)     # hidden budget anchor

    # Declared headline metric (v2 reference_metrics.json `metric_declared`).
    headline_metric: str = "smape"
    priors: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    show_test_range: bool = True
    experiment_log: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _experiment_counter: int = 0

    @classmethod
    def load(
        cls,
        task_dir: str | Path,
        simulator_name: str,
        show_test_range: bool = True,
    ) -> "SimulatorTask":
        td, meta, target, target_col, meta_inputs, has_group_id = _load_meta(task_dir)

        sim = _load_simulator_module(td, simulator_name)
        if not hasattr(sim, "fetch_data"):
            raise ValueError(
                f"simulator {simulator_name!r} under {td/'simulator'} "
                f"must expose `fetch_data(...)`."
            )

        # Inputs covered by the simulator (state.json input_support keys).
        sim_var_keys = _simulator_input_subset(td, simulator_name)
        if sim_var_keys:
            input_cols = [c for c in (i["name"] for i in meta_inputs) if c in sim_var_keys]
            if not input_cols:
                input_cols = list(sim_var_keys)
        else:
            input_cols = [i["name"] for i in meta_inputs]
        input_meta = [m for m in meta_inputs if m["name"] in input_cols]

        # Train / test from data/, exactly like RealSRTask.
        train_df = pd.read_csv(td / "data" / "train.csv").reset_index(drop=True)
        flat_test = td / "data" / "test.csv"
        if flat_test.is_file():
            test_df = pd.read_csv(flat_test).reset_index(drop=True)
        else:
            test_df = pd.read_csv(td / "data" / "test_test.csv").reset_index(drop=True)
        for df, name in [(train_df, "train"), (test_df, "test")]:
            for col in [target_col] + input_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"data/{name}.csv missing column {col!r}; "
                        f"got {list(df.columns)}"
                    )

        # Baselines from metadata (computed on real data/test.csv — same test
        # set this task evaluates against, so discovery_score is comparable
        # with fix-mode trials).
        baseline_metrics = _parse_baseline_metrics(meta)
        # v2 metadata declares the scoring metric directly (`metric:`); fall
        # back to reference_metrics.json `metric_declared`, then smape.
        headline_metric = meta.get("metric") or "smape"
        if not baseline_metrics:
            baseline_metrics, declared = _load_reference_baselines(td)
            if declared and not meta.get("metric"):
                headline_metric = declared

        visible_cols = [target_col, *input_cols]
        if has_group_id and GROUP_COL in train_df.columns:
            visible_cols.append(GROUP_COL)
        empty_train_df = train_df.iloc[0:0][visible_cols].copy()

        return cls(
            task_dir=td,
            task_id=meta.get("task_id", td.name),
            domain=str(meta.get("domain", "")),
            problem_statement=str(meta.get("context") or meta.get("problem_statement") or ""),
            metadata=meta,
            simulator=sim,
            simulator_name=simulator_name,
            sim_state=_simulator_state(td, simulator_name),
            target_col=target_col,
            target_meta=target,
            input_cols=input_cols,
            input_meta=input_meta,
            has_group_id=has_group_id,
            baseline_metrics=baseline_metrics,
            _test_df=test_df,
            _train_df=empty_train_df,
            _real_train_df=train_df,
            headline_metric=headline_metric,
            priors=list(meta.get("priors") or []),
            show_test_range=show_test_range,
        )

    # ---- mirror RealSRTask agent-facing API --------------------------------

    def is_multi_group(self) -> bool: return self.has_group_id

    @property
    def train(self) -> pd.DataFrame:
        return self._train_df

    @property
    def test(self) -> pd.DataFrame:
        return self._test_df

    def get_task_prompt(self, max_turns: int = 30) -> str:
        from prompts import build_task_prompt
        return build_task_prompt(
            task_id=self.task_id,
            domain=self.domain,
            problem_statement=self.problem_statement,
            target_col=self.target_col,
            target_meta=self.target_meta,
            input_cols=self.input_cols,
            input_meta=self.input_meta,
            has_group_id=self.has_group_id,
            n_train_rows=len(self._train_df),
            n_test_rows=len(self._test_df),
            n_train_groups=0,
            n_test_groups=(int(self._test_df[GROUP_COL].nunique())
                           if self.has_group_id and GROUP_COL in self._test_df else 0),
            max_turns=max_turns,
            metric=self.headline_metric,
            include_test_range=self.show_test_range,
            is_simulator=True,
            oracle_description=self._build_oracle_description(),
            caps=self.metadata.get("caps") or {},
        )

    def _build_oracle_description(self) -> str:
        """Construct a short neutral simulator note from state.joblib."""
        lines: List[str] = []
        if self.has_group_id:
            groups = self.sim_state.get("groups") or {}
            if groups:
                tg = sorted(groups.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                sup_per_g = {g: v.get("support", {}) for g, v in groups.items()}
            else:
                tg = sorted(int(g) for g in (self.sim_state.get("train_groups") or []))
                sup_per_g = self.sim_state.get("train_support_per_group") or {}
            if tg:
                lines.append(
                    f"- `<experiment>` requires `group_id=[...]` (one int per "
                    f"input point) and accepts only values from the available simulator groups: "
                    f"{tg if len(tg) <= 20 else str(tg[:20]) + '...'}."
                )
            if sup_per_g:
                lines.append(
                    "- Per-group experiment support (input is clipped to the chosen "
                    "group's support; out-of-range values are reported in "
                    "`n_clipped`):"
                )
                for g in tg[:5]:
                    rngs = sup_per_g.get(g) or sup_per_g.get(str(g)) or {}
                    bits = ", ".join(
                        f"{c} ∈ [{r['min']:.4g}, {r['max']:.4g}]"
                        for c, r in rngs.items() if isinstance(r, dict)
                    )
                    lines.append(f"  * group {g}: {bits}")
                if len(tg) > 5:
                    lines.append(f"  * (... and {len(tg) - 5} more train groups)")
        else:
            sup = self.sim_state.get("support") or self.sim_state.get("train_support") or {}
            for col, rng in sup.items():
                if isinstance(rng, dict) and "min" in rng and "max" in rng:
                    lines.append(
                        f"- `<experiment>` accepts `{col}` in "
                        f"[{rng['min']:.4g}, {rng['max']:.4g}]; "
                        f"out-of-range values are silently moved to the nearest endpoint and reported in `n_clipped`."
                    )
        if not lines:
            lines.append("- The simulator returns noisy observations at the requested input values.")
        caps = self.experiment_caps()
        lines.append(
            f"- Experiment budget caps: at most {caps['max_experiment_calls']} "
            f"`<experiment>` calls per trial; each call may request at most "
            f"{caps['max_input_points_per_call']} input points, at most "
            f"{caps['max_samples_per_point']} samples per point, and at most "
            f"{caps['max_rows_per_call']} returned rows per call. "
            "Requests above these caps are rejected without calling the simulator."
        )
        lines.append("- `<experiment>` appends returned rows to `X_train`, `y_train`, and `train_df`.")
        return "\n".join(lines)

    def run_experiment(self, n_samples: int = 1,
                       seed: Optional[int] = None,
                       **input_vals) -> Dict[str, Any]:
        """Probe the simulator at LLM-chosen input values; appends synthetic
        observations to the cumulative train log (X_train / y_train / train_df).

        Args:
            n_samples: how many independent draws per input point (default 1)
            seed: RNG seed for reproducibility (optional)
            **input_vals: one keyword per simulator input, value = list of floats.
                          For T2 (multi-group) tasks also pass `group_id` as a list
                          of integers (one per input point).
        """
        primary_input = self.input_cols[0]
        # Legacy alias for the asteroid task: forward `D_km_center=...` to the
        # actual primary input col when they differ.
        if primary_input not in input_vals and "D_km_center" in input_vals:
            input_vals[primary_input] = input_vals.pop("D_km_center")
        if input_vals.get(primary_input) is None:
            return {"error": (f"missing input values; pass `{primary_input}` as a "
                              "list of floats.")}
        if self.has_group_id and input_vals.get("group_id") is None:
            return {"error": ("multi-group task: pass `group_id` as a list of "
                              "integers (one per input point).")}
        caps = self.experiment_caps()
        if len(self.experiment_log) >= caps["max_experiment_calls"]:
            return {
                "error": (
                    f"experiment call cap reached: already ran "
                    f"{len(self.experiment_log)} successful experiments; maximum is "
                    f"{caps['max_experiment_calls']}."
                ),
                "experiment_caps": caps,
            }
        try:
            n_samples_i = int(n_samples)
        except Exception:
            return {
                "error": (
                    f"n_samples must be an integer in [1, "
                    f"{caps['max_samples_per_point']}]."
                ),
                "experiment_caps": caps,
            }
        if n_samples_i < 1 or n_samples_i > caps["max_samples_per_point"]:
            return {
                "error": (
                    f"n_samples={n_samples_i} exceeds experiment cap; use an "
                    f"integer in [1, {caps['max_samples_per_point']}]."
                ),
                "experiment_caps": caps,
            }

        def _request_count(v: Any) -> int:
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                return len(v)
            return 1

        input_counts = {k: _request_count(v) for k, v in input_vals.items()}
        n_points = max(input_counts.values()) if input_counts else 1
        if n_points < 1:
            return {"error": "experiment request must include at least one input point."}
        if n_points > caps["max_input_points_per_call"]:
            return {
                "error": (
                    f"requested {n_points} input points, exceeding cap "
                    f"{caps['max_input_points_per_call']}; request fewer points."
                ),
                "experiment_caps": caps,
            }

        requested_rows = n_points * n_samples_i
        if requested_rows > caps["max_rows_per_call"]:
            return {
                "error": (
                    f"requested up to {requested_rows} simulator rows, exceeding "
                    f"per-call cap {caps['max_rows_per_call']}; reduce input "
                    "points or n_samples."
                ),
                "experiment_caps": caps,
            }
        try:
            result = self.simulator.fetch_data(
                n_samples=n_samples_i,
                seed=seed,
                **input_vals,
            )
        except Exception as e:
            return {"error": f"simulator call failed: {type(e).__name__}: {e}"}
        if "error" in result:
            return result
        rows = result.get("rows") or []
        if len(rows) > caps["max_rows_per_call"]:
            return {
                "error": (
                    f"simulator returned {len(rows)} rows, exceeding per-call cap "
                    f"{caps['max_rows_per_call']}; no rows were appended."
                ),
                "experiment_caps": caps,
            }
        self._experiment_counter += 1
        experiment_id = self._experiment_counter
        n_before = len(self._train_df)
        appended_df: Optional[pd.DataFrame] = None
        if rows:
            cols = [self.target_col, *self.input_cols]
            # Phase2 simulators include group_id in rows. Phase1 sims on T2 tasks
            # don't — fall back to the LLM-supplied group_id list when available.
            phase2 = bool(rows) and (GROUP_COL in rows[0])
            if self.has_group_id and phase2:
                cols.append(GROUP_COL)
            new_df = pd.DataFrame(rows)[cols]
            if self.has_group_id and not phase2 and GROUP_COL not in new_df.columns:
                expanded = _expand_group_ids(input_vals.get("group_id"), n_samples_i)
                if expanded is not None and len(expanded) == len(new_df):
                    new_df[GROUP_COL] = expanded
            appended_df = new_df
            self._train_df = pd.concat([self._train_df, new_df], ignore_index=True)

        row_indices = list(range(n_before, n_before + len(rows)))
        preview_rows: List[Dict[str, Any]] = []
        if appended_df is not None:
            for raw in appended_df.head(EXPERIMENT_PREVIEW_ROWS).to_dict(orient="records"):
                item: Dict[str, Any] = {}
                for c in [self.target_col, *self.input_cols]:
                    item[c] = _safe_float(raw.get(c))
                if self.has_group_id and GROUP_COL in raw:
                    item[GROUP_COL] = int(raw[GROUP_COL])
                preview_rows.append(item)

        log_entry = {
            "experiment_id": experiment_id,
            "n_samples": n_samples_i,
            "seed": seed,
            "input_counts": input_counts,
            "n_returned": int(result.get("n_returned", len(rows))),
            "n_appended": len(rows),
            "n_clipped": int(result.get("n_clipped", 0)),
            "row_indices_returned": row_indices,
            "train_rows_total": len(self._train_df),
            "experiment_rows_total": len(self._train_df),
            "experiment_calls_total": len(self.experiment_log) + 1,
            "experiment_caps": caps,
        }
        self.experiment_log.append(log_entry)

        out: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "n_returned": int(result.get("n_returned", len(rows))),
            "n_appended_to_train": len(rows),
            "row_indices_returned": row_indices,
            "row_ids_returned": list(range(n_before + 1, n_before + 1 + len(rows))),
            "preview_rows": preview_rows,
            "preview_rows_shown": len(preview_rows),
            "preview_truncated": len(rows) > len(preview_rows),
            "n_rows_omitted_from_preview": max(0, len(rows) - len(preview_rows)),
            "train_rows_total": len(self._train_df),
            "experiment_rows_total": len(self._train_df),
            "experiment_calls_total": len(self.experiment_log),
            "experiment_caps": caps,
            "_n_unique_rows_seen": len(self._train_df),
            "n_clipped": int(result.get("n_clipped", 0)),
            "full_data_note": (
                "All returned rows are appended to train_df/X_train/y_train for "
                "the next <python> call. Use row_indices_returned with "
                "train_df.iloc[...] to inspect the full batch."
            ),
        }
        if self.has_group_id:
            phase2 = bool(rows) and (GROUP_COL in rows[0])
            if not phase2:
                expanded = _expand_group_ids(input_vals.get("group_id"), n_samples_i)
                if expanded is not None and len(expanded) == len(rows):
                    out["_note"] = ("phase1 simulator on a multi-group task: "
                                    "the simulator was fit without group structure, "
                                    "so y depends only on the inputs (group_id echoed but unused).")
        return out

    def experiment_caps(self) -> Dict[str, Any]:
        original_train_rows = len(self._real_train_df)
        max_points = max(10, int(math.floor(original_train_rows * MAX_EXPERIMENT_INPUT_POINT_FRACTION)))
        max_rows_per_call = max_points * MAX_EXPERIMENT_SAMPLES
        return {
            "max_experiment_calls": MAX_EXPERIMENT_CALLS,
            "max_input_points_per_call": max_points,
            "max_samples_per_point": MAX_EXPERIMENT_SAMPLES,
            "max_rows_per_call": max_rows_per_call,
            "max_total_rows_if_all_calls_use_full_budget": max_rows_per_call * MAX_EXPERIMENT_CALLS,
            "preview_rows": EXPERIMENT_PREVIEW_ROWS,
        }

    def reset_pool(self) -> None:
        """Reset the experiment-only train log to empty."""
        self._train_df = self._real_train_df.iloc[0:0][self._train_df.columns].copy()
        self.experiment_log.clear()
        self._experiment_counter = 0

    def train_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        df = self.train
        X = df[self.input_cols].to_numpy(dtype=float) if len(df) else np.empty((0, len(self.input_cols)))
        y = df[self.target_col].to_numpy(dtype=float) if len(df) else np.empty((0,))
        g = (df[GROUP_COL].to_numpy(dtype=int)
             if self.has_group_id and GROUP_COL in df.columns and len(df) else None)
        return X, y, g

    def test_arrays(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        X = self._test_df[self.input_cols].to_numpy(dtype=float)
        y = self._test_df[self.target_col].to_numpy(dtype=float)
        g = (self._test_df[GROUP_COL].to_numpy(dtype=int)
             if self.has_group_id and GROUP_COL in self._test_df.columns else None)
        return X, y, g

    def best_baseline(self, metric: str = "smape") -> Tuple[Optional[str], Optional[float]]:
        """Same fallback chain as RealSRTask. baseline_metrics come from
        metadata.yaml (computed on real data/test.csv — same test set the LLM
        is scored on, so discovery_score is comparable to fix-data trials)."""
        return _best_baseline(self.baseline_metrics, metric)


# ---- factory ---------------------------------------------------------------


def load_task(task_dir: str | Path, simulator: Optional[str] = None,
              context_mode: str = "full",
              show_test_range: Optional[bool] = None,
              **kwargs) -> "RealSRTask | SimulatorTask":
    """Load a task. `simulator=None` → fix-data mode (RealSRTask); a string
    selects a simulator under `<task_dir>/simulator/<name>/`. `context_mode`
    is honoured for fix-data only (the 4-arm context ablation)."""
    if simulator is None:
        show_ranges = True if show_test_range is None else show_test_range
        return RealSRTask.load(
            task_dir,
            context_mode=context_mode,
            show_test_range=show_ranges,
        )
    show_ranges = False if show_test_range is None else show_test_range
    return SimulatorTask.load(
        task_dir,
        simulator_name=simulator,
        show_test_range=show_ranges,
        **kwargs,
    )
