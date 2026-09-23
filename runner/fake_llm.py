"""Scripted stand-in LLM for the zero-API self-test (`provider: fake`).

It drives the canonical agent loop through a real <experiment> against the real
simulator, one <python> call and a contract-shaped <final_formula>, so launch /
retry / resume / packaging can be exercised end to end without an API key.

Per-task behaviour comes from env FAKE_LLM_BEHAVIOR (JSON {task_id: mode}):
  ok              experiment -> python -> final_formula
  no_submit       never submits (ends as max_turns_reached)
  crash_first     transport error on attempt 1, ok afterwards
  rate_limit_first  HTTP-429-style error on attempt 1, ok afterwards
  crash_always    transport error on every attempt
  hang            blocks forever (exercises the task timeout)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _experiment(task) -> str:
    st = task.sim_state
    payload: dict = {}
    if task.has_group_id:
        groups = st.get("groups") or {}
        gid = sorted(groups, key=str)[0]
        sup = groups[gid].get("support", {})
        payload["group_id"] = [int(gid)] * 3
    else:
        sup = st.get("support") or st.get("train_support") or {}
    for col in task.input_cols:
        r = sup.get(col) or {"min": 0.0, "max": 1.0}
        lo, hi = float(r["min"]), float(r["max"])
        payload[col] = [lo, (lo + hi) / 2.0, hi]
    payload.update({"n_samples": 1, "seed": 0})
    return "<experiment>" + json.dumps(payload) + "</experiment>"


def _final(task) -> str:
    used = json.dumps(list(task.input_cols))
    if task.has_group_id:
        body = ("import numpy as np\nUSED_INPUTS = %s\nLAW_CONSTANTS = {}\nOTHER_CONSTANTS = {}\n"
                "LOCAL_FITTABLE = {'a': {'init': None}}\n\n"
                "def fit(X_fit, y_fit):\n    return {'a': float(np.mean(y_fit))}\n\n"
                "def predict(X, a):\n    return np.full(len(X), float(a))\n") % used
    else:
        body = ("import numpy as np\nUSED_INPUTS = %s\nLAW_CONSTANTS = {}\nOTHER_CONSTANTS = {}\n"
                "LOCAL_FITTABLE = {}\n\n"
                "def predict(X):\n    return np.zeros(len(X))\n") % used
    return '<final_formula>\n"""self-test stub"""\n' + body + "</final_formula>"


def install(cfg: dict, task_dir: Path) -> None:
    import agent
    from task import load_task

    task = load_task(task_dir, simulator="simulator")
    modes = json.loads(os.environ.get("FAKE_LLM_BEHAVIOR") or "{}")
    mode = modes.get(task.task_id, "ok")
    attempt = int(os.environ.get("RUNNER_ATTEMPT") or "1")
    state = {"turn": 0}

    def fake_call(messages, model_name, trial_info=None, **_kw):
        state["turn"] += 1
        tid = (trial_info or {}).get("trial_id", "fake")
        if mode == "hang":
            time.sleep(10 ** 6)
        if mode == "crash_always" or (mode == "crash_first" and attempt == 1):
            print(f"[Trial {tid}] OpenAI API error on fake: Connection error.", flush=True)
            raise ConnectionError("Connection error.")
        if mode == "rate_limit_first" and attempt == 1:
            print(f"[Trial {tid}] OpenAI API error on fake: Error code: 429 - rate_limit_exceeded",
                  flush=True)
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")
        if mode == "no_submit":
            text = "<python>print(len(train_df))</python>"
        elif state["turn"] == 1:
            text = _experiment(task)
        elif state["turn"] == 2:
            text = "<python>print(train_df.describe().T.round(3).to_string())</python>"
        else:
            text = _final(task)
        usage = {"prompt_tokens": 100, "prompt_cached_tokens": 0, "completion_tokens": 10,
                 "reasoning_tokens": 0, "total_tokens": 110, "finish_reason": "stop",
                 "model": "fake", "api_source": "fake"}
        return text, None, usage

    agent.call_llm_api = fake_call
