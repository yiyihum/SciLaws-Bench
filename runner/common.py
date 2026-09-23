"""Shared helpers for the SCILAWS-PARALLEL external runner.

The runner never re-implements the benchmark: prompts, the agent loop, the
<experiment>/<python>/<final_formula> protocol, the sandbox and the query caps
all come from the canonical files in `harness/` and `baseline_agent/`, which are
kept byte-identical to upstream commit CANONICAL_COMMIT (checked by preflight).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO / "harness"
BASELINE_DIR = REPO / "baseline_agent"
MANIFEST_DIR = REPO / "manifests"
DEFAULT_TASKS_DIR = REPO / "tasks"
DEFAULT_RUNS_DIR = REPO / "runs"

CANONICAL_REPO = "https://github.com/yiyihum/SciLaws-Bench"
CANONICAL_COMMIT = "9239f66b921cb89c7a9d14061f782fdce49dcfb5"
HF_DATASET = "RealSR/SciLaws-Bench"
HF_REVISION = "15b93258fabc8f1785c5a5ffd4d3e51fc15dd860"

# The only per-task files the solver-side runner ships. Everything else in the
# public dataset (simulator/formula.py, simulator/sample.csv, eval/) is grader
# material and must not be present in the runner's task tree.
TASK_FILE_WHITELIST = ("metadata.yaml", "data/", "simulator/state.joblib")

# Status vocabulary (runs/<run>/tasks/<type>/<task>/state.json).
TERMINAL_STATUSES = ("submitted", "no_submission", "infra_failed")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write (tmp + rename) so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def load_manifest(name: str) -> Any:
    return json.loads((MANIFEST_DIR / name).read_text())


def task_list() -> List[Dict[str, str]]:
    """The frozen 118-task grid: [{"task": ..., "type": "typeI"|"typeII"}]."""
    return load_manifest("tasks_118.json")["tasks"]


def is_task_file(rel: str) -> bool:
    return rel == "metadata.yaml" or rel == "simulator/state.joblib" or (
        rel.startswith("data/") and rel.count("/") == 1)


def load_config(path: Path) -> Dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    required = ("run_name", "model_alias", "model_id", "provider", "base_url",
                "api_key_env", "reasoning_effort", "max_turns")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"config {path}: missing keys {missing}")
    cfg.setdefault("llm_timeout_seconds", 120)
    cfg.setdefault("task_timeout_seconds", 7200)
    cfg.setdefault("max_attempts", 3)
    cfg.setdefault("jobs", 40)
    return cfg


def config_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The config fields that must not change inside one run directory."""
    keys = ("model_alias", "model_id", "provider", "base_url", "reasoning_effort",
            "max_turns", "llm_timeout_seconds")
    return {k: cfg.get(k) for k in keys}


def child_env(cfg: Dict[str, Any], attempt: int | None = None) -> Dict[str, str]:
    """Environment for one task process. Mirrors run_batch.py: the canonical
    client reads reasoning effort and API timeout from these variables at
    import time."""
    env = os.environ.copy()
    env["OPENAI_REASONING_EFFORT"] = str(cfg["reasoning_effort"])
    env["REASONING_EFFORT"] = str(cfg["reasoning_effort"])
    env["LLM_TIMEOUT_SECONDS"] = str(cfg["llm_timeout_seconds"])
    if cfg["provider"] == "openai":
        env["OPENAI_BASE_URL"] = str(cfg["base_url"])
        key_env = cfg["api_key_env"]
        if key_env != "OPENAI_API_KEY" and os.environ.get(key_env):
            env["OPENAI_API_KEY"] = os.environ[key_env]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if attempt is not None:
        env["RUNNER_ATTEMPT"] = str(attempt)
    return env


def git_info(repo: Path = REPO) -> Dict[str, Any]:
    def _git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return ""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "runner_commit": _git("rev-parse", "HEAD") or None,
        "runner_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "runner_dirty": bool(status),
        "canonical_upstream_commit": CANONICAL_COMMIT,
        "canonical_is_ancestor": subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", CANONICAL_COMMIT, "HEAD"],
            capture_output=True).returncode == 0,
    }


def package_versions() -> Dict[str, str]:
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "PyYAML", "openai"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "MISSING"
    return out
