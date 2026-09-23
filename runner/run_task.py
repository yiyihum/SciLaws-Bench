"""One attempt of one SCILAWS-PARALLEL task, in its own process.

This is a thin shim around the canonical `baseline_agent/run_baseline.py`:

  1. register the configured model alias in the canonical client registry
     (`call_llm_api.api_source_mapping` + `_REASONING_MODELS`) so the canonical
     reasoning branch is taken (reasoning_effort, max_completion_tokens=65536,
     no temperature) -- an unregistered alias would otherwise raise, and a
     model missing from `_REASONING_MODELS` would silently lose its effort;
  2. assert the client actually picked up the effort / timeout from the env;
  3. call `run_baseline.main()` with the exact Table-2 argument shape:
       <task_dir> <alias> --max-turns 30 --out D --traj-out D --simulator

`--probe-request` builds one request through the same code path with the
OpenAI client replaced by a recorder, and prints the captured kwargs (zero API).
`provider: fake` swaps the LLM for a scripted stub (zero API, self-test only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import BASELINE_DIR, HARNESS_DIR, load_config  # noqa: E402

for _p in (str(BASELINE_DIR), str(HARNESS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def register_model(cfg: dict):
    """Register cfg's model in the canonical client and verify the wiring."""
    import call_llm_api as cla  # reads OPENAI_REASONING_EFFORT at import time

    alias, model_id = cfg["model_alias"], cfg["model_id"]
    if cfg["provider"] == "openai":
        cla.api_source_mapping[alias] = ("oa", model_id)
        if cfg.get("reasoning_model", True):
            cla._REASONING_MODELS.add(alias)
    effort = str(cfg["reasoning_effort"]).strip().lower()
    if cfg.get("reasoning_model", True) and cla._OPENAI_REASONING_EFFORT != effort:
        raise SystemExit(f"FATAL: canonical client effort={cla._OPENAI_REASONING_EFFORT!r}, "
                         f"config wants {effort!r} (env not propagated)")
    if float(cla._LLM_TIMEOUT_SECONDS) != float(cfg["llm_timeout_seconds"]):
        raise SystemExit(f"FATAL: canonical client timeout={cla._LLM_TIMEOUT_SECONDS}, "
                         f"config wants {cfg['llm_timeout_seconds']}")
    return cla


def probe_request(cfg: dict) -> dict:
    """Capture the exact kwargs the canonical client would send (no network)."""
    cla = register_model(cfg)
    captured: dict = {}

    class _Msg:
        content = "<python>print(1)</python>"
        reasoning_content = None

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Completion:
        choices = [_Choice()]
        usage = None

    class _Completions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return _Completion()

    class _Chat:
        completions = _Completions()

    class RecorderOpenAI:
        def __init__(self, **kwargs):
            import os
            captured["client"] = {k: (v if k != "api_key" else bool(v)) for k, v in kwargs.items()}
            captured["client"]["base_url_effective"] = (
                kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")
                or "https://api.openai.com/v1")
            self.chat = _Chat()

    cla.OpenAI = RecorderOpenAI
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    cla.call_llm_api(msgs, model_name=cfg["model_alias"], trial_info={"trial_id": "probe"})
    captured["resolved"] = list(cla.resolve_model_and_source(cfg["model_alias"]))
    captured["is_reasoning"] = cfg["model_alias"] in cla._REASONING_MODELS
    return captured


def record_observations(out_dir: Path) -> None:
    """Log every <experiment> call with ALL returned rows to observations.jsonl.

    The canonical trajectory keeps only a 10-row preview per batch; the full
    batch lives in the in-memory train_df. This wrapper calls the canonical
    method unchanged and only appends a log line, so the model-visible
    behaviour is identical."""
    import task as task_mod

    orig = task_mod.SimulatorTask.run_experiment
    log_path = Path(out_dir) / "observations.jsonl"

    def run_experiment(self, *args, **kwargs):
        result = orig(self, *args, **kwargs)
        rows = []
        idx = result.get("row_indices_returned") if isinstance(result, dict) else None
        if idx:
            rows = self._train_df.iloc[idx].to_dict(orient="records")
        with log_path.open("a") as fh:
            fh.write(json.dumps({"request": {**kwargs, **({"_args": list(args)} if args else {})},
                                 "result_shown_to_model": result, "returned_rows": rows},
                                default=str) + "\n")
        return result

    task_mod.SimulatorTask.run_experiment = run_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--task-dir", type=Path)
    ap.add_argument("--out", type=Path, help="attempt directory")
    ap.add_argument("--probe-request", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.probe_request:
        print(json.dumps(probe_request(cfg), default=str))
        return

    if cfg["provider"] == "fake":
        from runner import fake_llm
        fake_llm.install(cfg, args.task_dir)
    else:
        register_model(cfg)

    record_observations(args.out)
    import run_baseline  # canonical entry point, unmodified
    sys.argv = [
        "run_baseline.py", str(args.task_dir), cfg["model_alias"],
        "--max-turns", str(cfg["max_turns"]),
        "--out", str(args.out), "--traj-out", str(args.out),
        "--simulator",
    ]
    run_baseline.main()


if __name__ == "__main__":
    main()
