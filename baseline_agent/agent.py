"""Baseline LLM-as-agent loop for RealSR v3.

This is just ONE agent's orchestration: drive a chat model turn-by-turn. The
FIXED, reusable parts live in the harness and are imported here, so any other
agent (e.g. an evolving / search agent with its own loop) can reuse exactly the
same interface:

  - `prompts.load_system_prompt` / `task.get_task_prompt`  — the instruction text
  - `agent_protocol.step(response, sandbox, run_experiment)` — parse the model's
    tool tag, run it (<python> sandbox / <experiment>), and return the submission
    or the feedback to append. (see harness/AGENT_INTERFACE.md)

Swap `call_llm_api` for any client; reuse the rest.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# The fixed interface (prompts + tool-call protocol) lives in the sibling harness.
_HARNESS = Path(__file__).resolve().parent.parent / "harness"
if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

from call_llm_api import call_llm_api          # noqa: E402  (baseline-specific client)
from prompts import load_system_prompt          # noqa: E402  (harness)
import agent_protocol as proto                   # noqa: E402  (harness)


_NUMERIC_USAGE_KEYS = ("prompt_tokens", "prompt_cached_tokens",
                       "completion_tokens", "reasoning_tokens", "total_tokens")

FINAL_ACTION_MSG = (
    "Only one action remains. You must submit the final answer now using exactly "
    "one `<final_formula>...</final_formula>` block. Do not call `<python>` or "
    "`<experiment>`, and do not include prose outside the XML block."
)

FINAL_RETRY_MSG = (
    "Your previous response did not submit a final formula. Output exactly one "
    "complete `<final_formula>...</final_formula>` block now. Do not call "
    "`<python>` or `<experiment>`, and do not include prose outside the XML block."
)

SIMULATOR_EXPERIMENT_REQUIRED_MSG = (
    "This is a simulator-backed task and you have not run any successful "
    "`<experiment>` yet. You must probe the simulator at least once before "
    "submitting. Output exactly one `<experiment>{...}</experiment>` block now."
)


def _call_llm_and_record(messages: List[Dict[str, str]], model_name: str,
                         trial_info: Dict[str, Any]) -> Tuple[List[Dict[str, str]], dict, str]:
    response_text, reasoning_response, usage = call_llm_api(
        messages, model_name=model_name, trial_info=trial_info)
    if response_text is None:
        response_text = ""
    if isinstance(usage, int):
        usage = {"completion_tokens": usage, "total_tokens": usage,
                 "prompt_tokens": 0, "reasoning_tokens": 0, "prompt_cached_tokens": 0}
    elif not isinstance(usage, dict):
        usage = {"completion_tokens": 0, "total_tokens": 0,
                 "prompt_tokens": 0, "reasoning_tokens": 0, "prompt_cached_tokens": 0}
    else:
        usage = dict(usage)
    reasoning_content = str(reasoning_response or "")
    usage["reasoning_content"] = reasoning_content
    usage["reasoning_content_chars"] = len(reasoning_content)
    # Save provider reasoning in usage_per_turn, but never feed it back to the model.
    messages.append({"role": "assistant", "content": response_text})
    return messages, usage, response_text


def _accumulate_usage(usage_total: dict, usage_per_turn: List[dict], usage: dict) -> None:
    for k in _NUMERIC_USAGE_KEYS:
        usage_total[k] += int(usage.get(k, 0) or 0)
    reasoning_content = str(usage.get("reasoning_content") or "")
    usage_total["reasoning_content_chars"] = (
        int(usage_total.get("reasoning_content_chars", 0) or 0)
        + int(usage.get("reasoning_content_chars") or len(reasoning_content))
    )
    per_turn = {k: int(usage.get(k, 0) or 0) for k in _NUMERIC_USAGE_KEYS}
    per_turn["finish_reason"] = usage.get("finish_reason")
    per_turn["reasoning_content"] = reasoning_content
    per_turn["reasoning_content_chars"] = int(
        usage.get("reasoning_content_chars") or len(reasoning_content)
    )
    usage_per_turn.append(per_turn)


def _append_user_nudge(messages: List[Dict[str, str]], text: str) -> None:
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n\n" + text
    else:
        messages.append({"role": "user", "content": text})


def _build_result(status: str, submitted: str, rounds: int, usage_total: dict,
                  usage_per_turn: List[dict], messages: list, n_experiments: int,
                  n_python_calls: int) -> Dict[str, Any]:
    return {
        "status": status,
        "submitted_equation": submitted,
        "rounds": rounds,
        "total_tokens": usage_total["total_tokens"],
        "usage_total": usage_total,
        "usage_per_turn": usage_per_turn,
        "n_experiments": n_experiments,
        "n_python_calls": n_python_calls,
        "chat_history": messages,
    }


def _build_python_sandbox(task: Any) -> Dict[str, Any]:
    """Build the current <python> sandbox from task state.

    Simulator tasks mutate `task.train` after each <experiment>; rebuilding here
    makes the full cumulative experiment log visible to later <python> turns.
    """
    X, y, g = task.train_arrays()
    sandbox = proto.build_sandbox(
        train_df=task.train.copy(),
        X_train=X,
        y_train=y,
        group_ids=g,
        input_cols=task.input_cols,
        target_col=task.target_col,
    )
    experiment_log = getattr(task, "experiment_log", None)
    if experiment_log is not None:
        sandbox["experiment_log"] = list(experiment_log)
    experiment_caps = getattr(task, "experiment_caps", None)
    if callable(experiment_caps):
        sandbox["experiment_caps"] = experiment_caps()
    return sandbox


def conduct_exploration(task: Any, model_name: str, max_turns: int = 30,
                        trial_info: Dict[str, Any] | None = None,
                        checkpoint_fn: Callable[[Dict[str, Any]], None] | None = None,
                        resume_messages: List[Dict[str, str]] | None = None,
                        resume_state: Dict[str, Any] | None = None,
                        ) -> Dict[str, Any]:
    """Run the multi-turn baseline agent and return a trial dict (with
    `submitted_equation`).

    Warm-resume: pass `resume_messages` (a saved `chat_history`) plus
    `resume_state` ({n_experiments, n_python_calls, usage_total,
    usage_per_turn, start_turn}) to continue an interrupted trajectory from its
    checkpoint instead of starting from turn 0. The <python> sandbox is stateless
    across turns (each call re-seeds from the preloaded task data), so a resumed
    run has an execution environment identical to an uninterrupted one.
    """
    sys_prompt = load_system_prompt(is_simulator=hasattr(task, "run_experiment"),
                                    has_group_id=getattr(task, "has_group_id", False))
    if resume_messages is not None:
        messages: List[Dict[str, str]] = list(resume_messages)
    else:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task.get_task_prompt(max_turns=max_turns)},
        ]

    # Preloaded <python> sandbox. For simulator tasks this is refreshed after
    # every successful <experiment> so Python sees the cumulative lab notebook.
    sandbox = _build_python_sandbox(task)
    run_experiment = getattr(task, "run_experiment", None)
    requires_experiment = run_experiment is not None

    usage_total = {k: 0 for k in _NUMERIC_USAGE_KEYS}
    usage_per_turn: List[dict] = []
    n_experiments = n_python_calls = 0
    start_turn = 0
    if resume_state is not None:
        n_experiments = int(resume_state.get("n_experiments", 0) or 0)
        n_python_calls = int(resume_state.get("n_python_calls", 0) or 0)
        start_turn = int(resume_state.get("start_turn", 0) or 0)
        usage_per_turn = list(resume_state.get("usage_per_turn") or [])
        prev_total = resume_state.get("usage_total") or {}
        for k in _NUMERIC_USAGE_KEYS:
            usage_total[k] = int(prev_total.get(k, 0) or 0)
        usage_total["reasoning_content_chars"] = int(
            prev_total.get("reasoning_content_chars", 0) or 0)

    def _checkpoint_result(result: Dict[str, Any]) -> None:
        if checkpoint_fn is None:
            return
        experiment_log = getattr(task, "experiment_log", None)
        if experiment_log is not None:
            result["experiment_log"] = list(experiment_log)
        experiment_caps = getattr(task, "experiment_caps", None)
        if callable(experiment_caps):
            result["experiment_caps"] = experiment_caps()
        try:
            result["train_rows_current"] = len(task.train)
        except Exception:
            pass
        checkpoint_fn(result)

    def _checkpoint(status: str, submitted: str, rounds: int) -> Dict[str, Any]:
        result = _build_result(status, submitted, rounds, usage_total,
                               usage_per_turn, messages, n_experiments,
                               n_python_calls)
        _checkpoint_result(result)
        return result

    # Warm-resume: if the checkpoint stopped right after recording an assistant
    # reply that was never processed (status == "running_llm_response"), step it
    # now before continuing — it may itself carry a submission.
    if (resume_messages is not None and messages
            and messages[-1]["role"] == "assistant" and start_turn < max_turns):
        pending = messages[-1]["content"]
        res = proto.step(pending, sandbox, run_experiment=run_experiment)
        if res["action"] == "submit" and not (requires_experiment and n_experiments == 0):
            return _checkpoint("completed", res["submission"], start_turn)
        if res["action"] == "submit":
            messages.append({"role": "user", "content": SIMULATOR_EXPERIMENT_REQUIRED_MSG})
        else:
            if res["action"] == "python":
                n_python_calls += 1
            elif res["action"] == "experiment" and res.get("ok"):
                n_experiments += 1
                sandbox = _build_python_sandbox(task)
            messages.append({"role": "user", "content": res["feedback"]})
        _checkpoint("running", "", start_turn)

    for turn in range(start_turn, max_turns):
        is_final_action = turn == max_turns - 1
        if is_final_action and requires_experiment and n_experiments == 0:
            _append_user_nudge(messages, SIMULATOR_EXPERIMENT_REQUIRED_MSG)
        elif is_final_action:
            _append_user_nudge(messages, FINAL_ACTION_MSG)

        messages, usage, response_text = _call_llm_and_record(messages, model_name, trial_info or {})
        _accumulate_usage(usage_total, usage_per_turn, usage)
        if "model" in usage and "model" not in usage_total:
            usage_total["model"] = usage["model"]
            usage_total["api_source"] = usage.get("api_source")
        _checkpoint("running_llm_response", "", turn + 1)

        if is_final_action:
            if requires_experiment and n_experiments == 0:
                res = proto.step(response_text, sandbox, run_experiment=run_experiment)
                if res["action"] == "experiment" and res.get("ok"):
                    n_experiments += 1
                    sandbox = _build_python_sandbox(task)
                messages.append({
                    "role": "user",
                    "content": res.get("feedback", SIMULATOR_EXPERIMENT_REQUIRED_MSG),
                })
                _checkpoint("running", "", turn + 1)
                continue
            ok, submitted = proto.parse_final_formula(response_text)
            if ok:
                return _checkpoint("completed", submitted, turn + 1)
            _append_user_nudge(messages, FINAL_RETRY_MSG)
            messages, usage, response_text = _call_llm_and_record(
                messages, model_name, trial_info or {})
            _accumulate_usage(usage_total, usage_per_turn, usage)
            ok, submitted = proto.parse_final_formula(response_text)
            return _checkpoint("completed_forced_final" if ok else "max_turns_reached",
                               submitted, turn + 2)

        res = proto.step(response_text, sandbox, run_experiment=run_experiment)
        if res["action"] == "submit":
            if requires_experiment and n_experiments == 0:
                messages.append({"role": "user", "content": SIMULATOR_EXPERIMENT_REQUIRED_MSG})
                _checkpoint("running", "", turn + 1)
                continue
            return _checkpoint("completed", res["submission"], turn + 1)
        if res["action"] == "python":
            n_python_calls += 1
        elif res["action"] == "experiment" and res.get("ok"):
            n_experiments += 1
            sandbox = _build_python_sandbox(task)
        messages.append({"role": "user", "content": res["feedback"]})
        _checkpoint("running", "", turn + 1)

    _append_user_nudge(messages, FINAL_RETRY_MSG)
    messages, usage, response_text = _call_llm_and_record(messages, model_name, trial_info or {})
    _accumulate_usage(usage_total, usage_per_turn, usage)
    ok, submitted = proto.parse_final_formula(response_text)
    return _checkpoint("completed_forced_final" if ok else "max_turns_reached", submitted,
                       1 if max_turns <= 0 else max_turns + 1)
