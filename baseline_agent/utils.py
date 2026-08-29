"""Shared helpers for working with saved trials under `results/`.

Consolidates the reusable pieces that used to live in the internal
`_count_tokens.py` / `_estimate_cost.py` / `_view_trial.py` scratch scripts:

  - `iter_trials`              walk results/<model>/<task>/<mode>/<exp>/trial_*.json
  - `tokenize_chat`           re-tokenise a chat_history with tiktoken
  - `estimate_tokens_from_chat`  prompt/completion split when usage_total is absent
  - `PRICES` / `cost_from_usage` / `estimate_cost_from_chat`  USD cost estimation
  - `alias_to_full`           model alias → exact provider model id
  - `print_trial`             pretty-print one saved trial JSON

Trial JSON layout (written by run_trial.py):
    results/<model>/<task_id>/<mode>/<experiment>/trial_<seed:04d>_<ts>.json
    {"meta": {...}, "trial": {... "usage_total": {...}}, "chat_history": [...],
     "test_eval": {...}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---- trial iteration --------------------------------------------------------


def iter_trials(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    model: Optional[str] = None,
) -> Iterator[Tuple[Path, Dict[str, Any], str]]:
    """Yield (path, parsed_json, model_alias) for every saved trial.

    Walks `results/<model>/<task>/<mode>/<experiment>/trial_*.json`. When
    `model` is given, restricts to that model's subtree. Files that fail to
    parse, or that don't sit at the expected 5-part depth, are skipped.
    """
    root = Path(results_dir)
    glob_pat = (f"{model}/" if model else "*/") + "*/*/*/trial_*.json"
    for trial_path in root.glob(glob_pat):
        try:
            data = json.loads(trial_path.read_text())
        except Exception:
            continue
        parts = trial_path.relative_to(root).parts
        if len(parts) < 5:
            continue
        yield trial_path, data, parts[0]


# ---- token counting ---------------------------------------------------------


def _get_encoder():
    """tiktoken encoder (o200k_base, falling back to cl100k_base), or None."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def tokenize_chat(chat: List[Dict[str, Any]]) -> Optional[int]:
    """Total token count of every message's content, or None if tiktoken
    is unavailable."""
    enc = _get_encoder()
    if enc is None:
        return None
    return sum(len(enc.encode(m.get("content") or "")) for m in chat)


def estimate_tokens_from_chat(chat: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Approximate prompt/completion split when usage_total is missing.

    Each assistant turn is one LLM call: prompt_tokens at that turn = tokens
    of all prior messages; completion_tokens = tokens of the assistant
    message. Returns None if tiktoken is unavailable.
    """
    enc = _get_encoder()
    if enc is None:
        return None

    def n(s: str) -> int:
        return len(enc.encode(s or ""))

    prompt_total = 0
    completion_total = 0
    running_prompt = 0
    for msg in chat:
        toks = n(msg.get("content", ""))
        if msg.get("role") == "assistant":
            prompt_total += running_prompt
            completion_total += toks
            running_prompt += toks  # this turn becomes part of the next prompt
        else:
            running_prompt += toks
    return {
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "prompt_cached_tokens": 0,
        "reasoning_tokens": 0,
    }


# ---- cost estimation --------------------------------------------------------

# USD per 1M tokens (regular tier; off-peak DeepSeek = 50% off but we use full
# price as an upper bound). Update if rates change. Keyed by exact model id.
PRICES: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4.1-mini-2025-04-14": {"in": 0.40, "in_cached": 0.10, "out": 1.60},
    "gpt-4.1-2025-04-14":      {"in": 2.00, "in_cached": 0.50, "out": 8.00},
    "o4-mini-2025-04-16":      {"in": 1.10, "in_cached": 0.275, "out": 4.40},
    "gpt-5-nano":              {"in": 0.05, "in_cached": 0.005, "out": 0.40},
    "gpt-5-mini-2025-08-07":   {"in": 0.25, "in_cached": 0.025, "out": 2.00},
    "gpt-5":                   {"in": 1.25, "in_cached": 0.125, "out": 10.00},
    "gpt-5-pro":               {"in": 15.0, "in_cached": 1.50,  "out": 120.0},
    "gpt-5-codex":             {"in": 1.25, "in_cached": 0.125, "out": 10.00},
    "gpt-5.1":                 {"in": 1.25, "in_cached": 0.125, "out": 10.00},
    "gpt-5.4":                 {"in": 1.25, "in_cached": 0.125, "out": 10.00},
    "gpt-5.5":                 {"in": 1.25, "in_cached": 0.125, "out": 10.00},
    "gpt-5.5-pro":             {"in": 15.0, "in_cached": 1.50,  "out": 120.0},
    # DeepSeek
    "deepseek-chat":           {"in": 0.27, "in_cached": 0.07, "out": 1.10},
    "deepseek-reasoner":       {"in": 0.55, "in_cached": 0.14, "out": 2.19},
    "deepseek-v4-flash":       {"in": 0.27, "in_cached": 0.07, "out": 1.10},
    "deepseek-v4-pro":         {"in": 0.55, "in_cached": 0.14, "out": 2.19},
}


def cost_from_usage(usage: Dict[str, Any], model: str) -> Optional[float]:
    """USD cost from an exact `usage_total` breakdown, or None if `model`
    isn't in PRICES. `completion_tokens` already includes reasoning tokens
    for OpenAI-style usage."""
    rates = PRICES.get(model)
    if rates is None:
        return None
    p_total = int(usage.get("prompt_tokens", 0))
    p_cached = int(usage.get("prompt_cached_tokens", 0))
    p_uncached = max(p_total - p_cached, 0)
    c = int(usage.get("completion_tokens", 0))
    return (p_uncached * rates["in"] + p_cached * rates["in_cached"]
            + c * rates["out"]) / 1e6


def estimate_cost_from_chat(
    chat: List[Dict[str, Any]], model: str
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Approximate (usd, token_breakdown) by re-tokenising chat_history when
    usage_total is missing. Returns (None, None) if tiktoken is unavailable
    or `model` isn't priced."""
    rates = PRICES.get(model)
    if rates is None:
        return None, None
    est = estimate_tokens_from_chat(chat)
    if est is None:
        return None, None
    cost = (est["prompt_tokens"] * rates["in"]
            + est["completion_tokens"] * rates["out"]) / 1e6
    return cost, {**est, "estimated": True}


# ---- model aliases ----------------------------------------------------------


def alias_to_full(alias: str) -> str:
    """Resolve a model alias to its exact provider model id via
    call_llm_api.api_source_mapping. Returns `alias` unchanged if unknown."""
    try:
        from call_llm_api import api_source_mapping
        return api_source_mapping.get(alias, ("?", alias))[1]
    except Exception:
        return alias


# ---- trial viewer -----------------------------------------------------------


def print_trial(data: Dict[str, Any]) -> None:
    """Pretty-print one saved trial JSON (meta, test eval, full chat history)."""
    trial = data["trial"]
    print("=== TRIAL META ===")
    print(f"  status={trial['status']}  rounds={trial['rounds']}  "
          f"tokens={trial.get('total_tokens')}  elapsed={data.get('meta', {}).get('elapsed_s', data.get('elapsed_s', 0)):.1f}s")
    print(f"  python_calls={trial.get('n_python_calls', 0)}  "
          f"experiments={trial.get('n_experiments', 0)}  "
          f"data_requests={trial.get('n_data_requests', 0)}  "
          f"rows_seen={trial.get('n_unique_rows_seen', 0)}")
    e = data.get("test_eval")
    if e:
        print("\n=== TEST EVAL ===")
        if e["status"] == "ok":
            print(f"  smape={e['smape']:.4f}  mae={e['mae']:.4f}  rmse={e['rmse']:.4f}")
            bid, bval, ds = e.get("best_baseline_id"), e.get("best_baseline_value"), e.get("discovery_score")
            if bid is not None and bval is not None and ds is not None and ds == ds:
                print(f"  discovery_score={ds:.4f}  vs baseline {bid}={bval:.4f}")
            else:
                print("  discovery_score=N/A (no baseline reference)")
        else:
            print(f"  status={e['status']}  error={e.get('error')}")

    chat = data.get("chat_history") or []
    print(f"\n=== CHAT HISTORY ({len(chat)} messages) ===")
    for i, msg in enumerate(chat):
        bar = "═" * 78
        print(f"\n{bar}")
        print(f"[{i}] role={msg['role']}  len={len(msg['content'])}")
        print(bar)
        print(msg["content"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print_trial(json.loads(Path(sys.argv[1]).read_text()))
    else:
        print("usage: python utils.py <trial.json>   # pretty-print a saved trial")
