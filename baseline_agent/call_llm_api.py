"""LLM API client.

Supported providers, selected by alias prefix in `api_source_mapping`:
- 'oa'  : OpenAI direct (gpt-4*, gpt-5*, o-series).
- 'ds'  : DeepSeek (OpenAI-compatible API at https://api.deepseek.com).
- 'an'  : Anthropic (claude-* via the `anthropic` SDK).
- 'go'  : Google (gemini-* via the `google-genai` SDK).
- 'dm'  : DMXAPI aggregator gateway (OpenAI-compatible; serves GLM, Kimi,
          Qwen, DeepSeek, Doubao, etc. via one key). Base URL from
          `DMXAPI_BASE` (defaults to https://www.dmxapi.cn/v1).
- 'lo'  : local OpenAI-compatible server (vLLM/SGLang/Ollama/etc.). Base URL
          from `LOCAL_OPENAI_BASE` (defaults to http://localhost:30000/v1).
- 'or'  : OpenRouter aggregator (OpenAI-compatible; serves Anthropic,
          Google, Meta, DeepSeek, Qwen, Mistral, etc. via one key).
          Base URL from `OPENROUTER_BASE` (defaults to
          https://openrouter.ai/api/v1).

Each provider has its own response-shape quirk (reasoning content, token
budgets, system-prompt placement) — the per-source branch in `call_llm_api`
normalises them all to the same `(content, reasoning_content, breakdown)`
return tuple.
"""
import os

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is a convenience; keys can be exported directly

# API keys. DeepSeek key lives in `method/key` (bare key or `DEEPSEEK=sk-...`).
# Anthropic / Google use env vars (or .env).
_DS_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key")
try:
    with open(_DS_KEY_PATH) as _fh:
        _ds_raw = _fh.read().strip()
    if "=" in _ds_raw and not _ds_raw.startswith("sk-"):
        _ds_key = _ds_raw.split("=", 1)[1].strip()
    else:
        _ds_key = _ds_raw
except (OSError, FileNotFoundError):
    _ds_key = None

keys = {
    'oa': os.getenv("OPENAI_API_KEY"),
    'ds': _ds_key,
    'an': os.getenv("ANTHROPIC_API_KEY"),
    'go': os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    'dm': os.getenv("DMXAPI_KEY"),
    'lo': os.getenv("LOCAL_OPENAI_API_KEY", "EMPTY"),
    'or': os.getenv("OPENROUTER_API_KEY"),
}

_DMXAPI_BASE = os.getenv("DMXAPI_BASE", "https://www.dmxapi.cn/v1")
_LOCAL_OPENAI_BASE = os.getenv("LOCAL_OPENAI_BASE", "http://localhost:30000/v1")
_LOCAL_GLM52_MODEL = os.getenv("LOCAL_GLM52_MODEL", "glm-5.2")
_OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")


# Alias → exact OpenAI model id. NOTE: OpenAI's `/v1/models` listing endpoint
# is INCOMPLETE — some callable models (e.g. gpt-5.5) do not appear in the
# listing but route correctly when used directly. The list below is the
# audited callable set as of 2026-05; cross-referenced against successful
# `memorization/runs/*` audit logs.
api_source_mapping = {
    "gpt41mini":  ("oa", "gpt-4.1-mini-2025-04-14"),   # LLM
    "gpt41":      ("oa", "gpt-4.1-2025-04-14"),        # LLM
    "gpt4omini":  ("oa", "gpt-4o-mini-2024-07-18"),    # LLM
    "gpt-4o-mini": ("oa", "gpt-4o-mini-2024-07-18"),   # LLM
    "o4mini":     ("oa", "o4-mini-2025-04-16"),        # reasoning
    "gpt5nano":   ("oa", "gpt-5-nano"),                # reasoning (cheapest)
    "gpt5mini":   ("oa", "gpt-5-mini-2025-08-07"),     # reasoning
    "gpt-5-mini": ("oa", "gpt-5-mini-2025-08-07"),     # reasoning
    "gpt5":       ("oa", "gpt-5"),                     # reasoning
    "gpt5pro":    ("oa", "gpt-5-pro"),                 # reasoning (strongest gpt-5 series)
    "gpt5codex":  ("oa", "gpt-5-codex"),               # code-specialized
    "gpt51":      ("oa", "gpt-5.1"),                   # reasoning
    "gpt54":      ("oa", "gpt-5.4"),                   # reasoning
    "gpt54mini":  ("oa", "gpt-5.4-mini"),              # reasoning (mini tier)
    "gpt5.4":     ("oa", "gpt-5.4"),                   # reasoning
    "gpt5.4-mini": ("oa", "gpt-5.4-mini"),             # reasoning (mini tier)
    "gpt55":      ("oa", "gpt-5.5"),                   # reasoning (not in /v1/models listing but callable)
    "gpt5.5":     ("oa", "gpt-5.5"),                   # reasoning
    "gpt-5.5":    ("oa", "gpt-5.5"),                   # reasoning
    "gpt55pro":   ("oa", "gpt-5.5-pro"),               # reasoning
    # DeepSeek (OpenAI-compatible). `deepseek-chat` is V3.x non-reasoning;
    # `deepseek-reasoner` is R1-style reasoning with content in
    # `message.reasoning_content`. v4-flash / v4-pro are the V4-generation
    # models reported by /v1/models as of 2026-05.
    "deepseek-chat":      ("ds", "deepseek-chat"),
    "deepseekchat":       ("ds", "deepseek-chat"),
    "deepseek-reasoner":  ("ds", "deepseek-reasoner"),
    "deepseekreasoner":   ("ds", "deepseek-reasoner"),
    "deepseek-v4-flash":  ("ds", "deepseek-v4-flash"),
    "deepseekv4flash":    ("ds", "deepseek-v4-flash"),
    "deepseek-v4-pro":    ("ds", "deepseek-v4-pro"),
    "deepseekv4pro":      ("ds", "deepseek-v4-pro"),

    # Anthropic — extended-thinking reasoning models. The exact model id is
    # `claude-<family>-<version>` (e.g. `claude-opus-4-7`). System prompt is
    # passed via the `system` field, not as a chat message.
    "claude-opus-4-7":     ("an", "claude-opus-4-7"),
    "claude-opus-4":       ("an", "claude-opus-4"),
    "claude-sonnet-4-6":   ("an", "claude-sonnet-4-6"),
    "claude-haiku-4-5":    ("an", "claude-haiku-4-5"),

    # Google Gemini — long-context reasoning models via google-genai SDK.
    # Aliases match the documented model IDs at
    # https://ai.google.dev/gemini-api/docs/models.
    "gemini-3-1-pro":     ("go", "gemini-3.1-pro"),
    "gemini-3.1-pro":     ("go", "gemini-3.1-pro"),
    "gemini-3-pro":       ("go", "gemini-3-pro"),
    "gemini-2-5-pro":     ("go", "gemini-2.5-pro"),

    # DMXAPI aggregator. GLM-5.1, Kimi-K2.6 and Qwen3.6-Plus all emit a
    # reasoning trace inside `completion_tokens` (and expose it via
    # `reasoning_content` on the message), so the dm branch sets a large
    # `max_tokens` budget — see `_REASONING_MODELS` below. Qwen3-Next is
    # non-thinking by default and accepts `temperature`.
    "glm-5.1":            ("dm", "glm-5.1"),
    "glm51":              ("dm", "glm-5.1"),
    # Same upstream model as glm-5.1, but with thinking disabled via
    # Z.ai-native `thinking={"type":"disabled"}` (DMXAPI passes it
    # through). See `_THINKING_DISABLED` below for the body-injection
    # set. (Briefly routed via OpenRouter; switched back to DMXAPI —
    # similar timeout rate but DMXAPI is faster per round.)
    "glm-5.1-nothink":    ("dm", "glm-5.1"),
    "glm51nothink":       ("dm", "glm-5.1"),
    "kimi-k2.6":          ("dm", "kimi-k2.6"),
    "kimik26":            ("dm", "kimi-k2.6"),
    # Same upstream model as kimi-k2.6, but with thinking disabled via
    # `extra_body={'enable_thinking': False}` (DMXAPI passes it through).
    "kimi-k2.6-nothink":  ("dm", "kimi-k2.6"),
    "kimik26nothink":     ("dm", "kimi-k2.6"),
    # Kimi-K2 non-thinking — answer goes straight to msg.content, no
    # `reasoning_content`. Routed through OpenRouter (DMXAPI quota exhausted).
    "kimi-k2":            ("dm", "kimi-k2"),
    "kimik2":             ("or", "moonshotai/kimi-k2"),
    "qwen3.6-plus":       ("dm", "qwen3.6-plus"),
    "qwen36plus":         ("dm", "qwen3.6-plus"),
    "qwen3-next":         ("dm", "qwen3-next-80b-a3b-instruct"),
    # Small non-thinking Qwen — Alibaba's rotating "flash" alias (currently
    # tracks Qwen3.x weights but without the thinking head). Useful as a
    # cheap/fast lower-bound baseline. Do NOT confuse with `qwen3.6-flash`,
    # which DOES emit a reasoning trace.
    "qwen-flash":         ("dm", "qwen-flash"),
    "qwenflash":          ("dm", "qwen-flash"),
    # Qwen3 Coder family — all non-thinking. `qwen3-coder-flash` is the
    # smallest/fastest tier; well-suited as the small-model baseline for
    # SR agent loops since the model spends most of its tokens writing
    # Python fitting code.
    "qwen3-coder-flash":  ("dm", "qwen3-coder-flash"),
    "qwen3coderflash":    ("dm", "qwen3-coder-flash"),

    # Local OpenAI-compatible GLM-5.2 server.
    # Override the served model name with LOCAL_GLM52_MODEL if needed.
    "glm-5.2-local":           ("lo", _LOCAL_GLM52_MODEL),
    "glm52local":              ("lo", _LOCAL_GLM52_MODEL),
    "glm-5.2-local-nothink":   ("lo", _LOCAL_GLM52_MODEL),
    "glm52localnothink":       ("lo", _LOCAL_GLM52_MODEL),

    # OpenRouter aggregator. Model IDs follow `<provider>/<model>` per
    # OpenRouter docs (https://openrouter.ai/docs#models).
    "or-kimi-k2.6":       ("or", "moonshotai/kimi-k2-thinking"),
    "or-kimi-k2":         ("or", "moonshotai/kimi-k2"),
    # Anthropic via OpenRouter — used to top up the direct Anthropic SDK
    # runs when the Anthropic API quota is exhausted. Reasoning effort
    # is forwarded via OR's `reasoning={"effort":"medium"}` field — see
    # `_OR_REASONING_EFFORT` below.
    "or-claude-opus-4-7": ("or", "anthropic/claude-opus-4-7"),
    "or-claude-opus-4":   ("or", "anthropic/claude-opus-4"),
    "or-claude-sonnet-4-6": ("or", "anthropic/claude-sonnet-4-6"),
    "or-gemini-3-1-pro":  ("or", "google/gemini-3.1-pro-preview"),
    "or-gemini-3.1-pro":  ("or", "google/gemini-3.1-pro-preview"),
    "or-glm52":            ("or", "z-ai/glm-5.2"),
    "or-glm-5.2":          ("or", "z-ai/glm-5.2"),
    "or-glm52-nothink":    ("or", "z-ai/glm-5.2"),
    "or-glm-5.2-nothink":  ("or", "z-ai/glm-5.2"),
    "or-deepseek-v4-pro":  ("or", "deepseek/deepseek-v4-pro"),
    "or-qwen37max":        ("or", "qwen/qwen3.7-max"),
    "or-qwen3.7-max":      ("or", "qwen/qwen3.7-max"),
    # Examples (uncomment / add as needed):
    # "or/claude-opus-4-7":   ("or", "anthropic/claude-opus-4-7"),
    # "or/gemini-3-pro":      ("or", "google/gemini-3-pro"),
    # "or/llama-3.3-70b":     ("or", "meta-llama/llama-3.3-70b-instruct"),
    # "or/deepseek-r1":       ("or", "deepseek/deepseek-r1"),
}


# Reasoning models: use `max_completion_tokens` (not `max_tokens`) and do NOT
# pass `temperature` (locked at default; explicit values may be rejected).
# Includes the o-series and all gpt-5*. See memorization/client.py for
# precedent of this exact handling.
_REASONING_MODELS = {
    "o4mini",
    "gpt5nano", "gpt5mini", "gpt-5-mini", "gpt5", "gpt5pro", "gpt5codex",
    "gpt51", "gpt54", "gpt54mini", "gpt5.4", "gpt5.4-mini", "gpt55", "gpt5.5", "gpt-5.5", "gpt55pro",
    # DeepSeek-reasoner exposes `reasoning_content` and uses `max_tokens`
    # (not `max_completion_tokens`); handled in the per-source branch below.
    "deepseek-reasoner", "deepseekreasoner",
    "deepseek-v4-flash", "deepseekv4flash",
    "deepseek-v4-pro",   "deepseekv4pro",
    # Anthropic + Gemini use extended thinking by default for these tiers.
    "claude-opus-4-7", "claude-opus-4", "claude-sonnet-4-6", "claude-haiku-4-5",
    "gemini-3-1-pro", "gemini-3.1-pro", "gemini-3-pro", "gemini-2-5-pro",
    "or-kimi-k2.6", "or-kimi-k2",
    "or-claude-opus-4-7", "or-claude-opus-4", "or-claude-sonnet-4-6",
    "or-gemini-3-1-pro", "or-gemini-3.1-pro",
    "or-glm52", "or-glm-5.2", "or-deepseek-v4-pro", "or-qwen37max", "or-qwen3.7-max",
    # DMXAPI: GLM-5.1, Kimi-K2.6 and Qwen3.6-Plus are thinking models —
    # reasoning is included in `completion_tokens` and exposed via
    # `reasoning_content`, so the dm branch needs a large max_tokens
    # budget. Qwen3-Next is non-thinking by default and stays out of this
    # set.
    "glm-5.1", "glm51",
    "glm-5.2-local", "glm52local",
    "kimi-k2.6", "kimik26",
    "qwen3.6-plus", "qwen36plus",
}

# Aliases that should explicitly disable upstream thinking. The body field
# is provider-specific (Z.ai uses `thinking={"type":"disabled"}`, Moonshot
# uses `enable_thinking=False`); `_thinking_disable_body` returns the right
# shape per model. Sent through the OpenAI SDK's `extra_body=` escape hatch
# since these aren't standard OpenAI parameters. Both DMXAPI and OpenRouter
# forward `extra_body` to the upstream verbatim.
_THINKING_DISABLED = {
    "glm-5.1-nothink", "glm51nothink",
    "glm-5.2-local-nothink", "glm52localnothink",
    "or-glm52-nothink", "or-glm-5.2-nothink",
    "kimi-k2.6-nothink", "kimik26nothink",
}


def _thinking_disable_body(model_name: str) -> dict:
    if model_name.startswith("kimi"):
        return {"enable_thinking": False}
    if "local" in model_name:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    # Z.ai-family default (GLM)
    return {"thinking": {"type": "disabled"}}

# Default completion-token budget for reasoning models. Must cover BOTH the
# internal reasoning trace AND the visible output (tool-call body, code, etc.).
# Empirically gpt-5 burns ~6-8k on reasoning + the visible output can run to
# 1-3k for non-trivial <python> blocks. 32k was occasionally tight (showed up
# as truncated <python> bodies with no closing tag). 64k is a safer floor for
# longer agent loops that include curve_fit / multi-restart code.
_REASONING_MAX_COMPLETION_TOKENS = 65536
_OPENAI_REASONING_EFFORT = (
    os.getenv("OPENAI_REASONING_EFFORT")
    or os.getenv("REASONING_EFFORT")
    or ""
).strip().lower()
_LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS") or "120")


def resolve_model_and_source(model_name, keys=keys):
    """Return (api_source, exact_model_id) or raise.

    api_source ∈ {'oa', 'ds'}. Raises with a clean message if the relevant
    key is missing for the resolved provider.
    """
    if model_name not in api_source_mapping:
        raise ValueError(
            f"Model alias '{model_name}' not in api_source_mapping. "
            f"Available: {sorted(api_source_mapping)}"
        )
    api_source, full_model_name = api_source_mapping[model_name]
    if api_source == 'oa' and not keys.get('oa'):
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    if api_source == 'ds' and not keys.get('ds'):
        raise ValueError(
            f"DeepSeek API key not found. Expected single-line key file at "
            f"{_DS_KEY_PATH!r}."
        )
    if api_source == 'an' and not keys.get('an'):
        raise ValueError("ANTHROPIC_API_KEY is not set in the environment.")
    if api_source == 'go' and not keys.get('go'):
        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.")
    if api_source == 'dm' and not keys.get('dm'):
        raise ValueError(
            "DMXAPI_KEY is not set. Export it (and optionally DMXAPI_BASE) in "
            "the shell that launches this process."
        )
    if api_source == 'or' and not keys.get('or'):
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Export it (and optionally "
            "OPENROUTER_BASE) in the shell that launches this process."
        )
    return api_source, full_model_name


# Close-tag stop sequences — when set as `stop=`, the API truncates
# generation as soon as the model emits any of these. Forces single-tool-
# per-turn at the API level. The close tag itself is stripped from the
# returned content by OpenAI; `_heal_close_tag` re-appends it so the
# downstream parsers (which look for `<tag>...</tag>`) keep working.
_TOOL_CLOSE_TAGS = ("</python>", "</experiment>", "</final_formula>")

# Master switch for API-level `stop=` (one-tool-per-turn enforcement). Disabled
# for now — the agent loop already picks the first emitted tag, so stop is not
# required; some models reject it. Flip to True to re-enable.
_USE_STOP = False

# Full model ids discovered (during this process) to reject the `stop=` param
# — populated lazily on the first rejection so later turns skip it. Some
# OpenAI gpt-5* tiers reject `stop`; the per-turn retry was wasting one call.
_STOP_UNSUPPORTED: set = set()


def _heal_close_tag(content: str | None) -> str | None:
    """If content has an unclosed tool open-tag (because API stopped at the
    matching close tag), append the missing close tag back."""
    if content is None:
        return None
    last_open_pos = -1
    last_open_tag = None
    for tag in ("python", "experiment", "final_formula"):
        o, c = f"<{tag}>", f"</{tag}>"
        po = content.rfind(o)
        pc = content.rfind(c)
        if po > pc and po > last_open_pos:
            last_open_pos = po
            last_open_tag = tag
    if last_open_tag is not None:
        return content + f"</{last_open_tag}>"
    return content


def _split_system(messages):
    """Anthropic / Gemini take the system prompt as a separate field."""
    sys_text = ""
    rest = []
    for m in messages:
        if m.get("role") == "system" and not sys_text:
            sys_text = m.get("content") or ""
        else:
            rest.append(m)
    return sys_text, rest


def _provider_name(api_source: str) -> str:
    """Human-readable provider label for error/log messages."""
    return {"oa": "OpenAI", "ds": "DeepSeek", "an": "Anthropic",
            "go": "Google", "dm": "DMXAPI", "lo": "local OpenAI",
            "or": "OpenRouter"}.get(api_source, api_source)


def _call_anthropic(messages, model, keys, trial_id):
    """Anthropic Messages API. Uses provider defaults — no extended-thinking
    block, no temperature override, no reasoning effort knob."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("`pip install anthropic` is required for claude-* models.") from e
    if not keys.get('an'):
        raise ValueError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=keys['an'])
    system_text, chat = _split_system(messages)
    kwargs = {
        "model": model,
        "system": system_text,
        "messages": chat,
        "max_tokens": _REASONING_MAX_COMPLETION_TOKENS,
    }
    if _USE_STOP:
        # Enforce one-tool-per-turn at the API level. Anthropic strips the
        # matching close tag from the content; `_heal_close_tag` re-appends it.
        kwargs["stop_sequences"] = list(_TOOL_CLOSE_TAGS)
    try:
        resp = client.messages.create(**kwargs)
    except Exception as e:
        print(f"[Trial {trial_id}] Anthropic API error on {model}: {e}", flush=True)
        raise
    content = ""
    thinking = ""
    for block in resp.content or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            content += getattr(block, "text", "") or ""
        elif btype == "thinking":
            thinking += getattr(block, "thinking", "") or ""
    content = _heal_close_tag(content)
    u = getattr(resp, "usage", None)
    breakdown = {
        "prompt_tokens": int(getattr(u, "input_tokens", 0) or 0),
        "prompt_cached_tokens": int(getattr(u, "cache_read_input_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "output_tokens", 0) or 0),
        "reasoning_tokens": 0,  # Anthropic counts thinking inside output_tokens
        "total_tokens": int(getattr(u, "input_tokens", 0) or 0) + int(getattr(u, "output_tokens", 0) or 0),
        "finish_reason": getattr(resp, "stop_reason", "stop"),
        "model": model, "api_source": "an",
    }
    if breakdown["finish_reason"] in ("max_tokens",):
        print(f"[Trial {trial_id}] WARNING: {model} hit max_tokens "
              f"(output={breakdown['completion_tokens']}). Tool body likely truncated.",
              flush=True)
    return content, (thinking or None), breakdown


def _call_gemini(messages, model, keys, trial_id):
    """Gemini via google-genai SDK. Adapts roles user/model and uses
    `system_instruction` for the system prompt."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise ImportError("`pip install google-genai` is required for gemini-* models.") from e
    if not keys.get('go'):
        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set.")
    client = genai.Client(api_key=keys['go'])
    system_text, chat = _split_system(messages)
    contents = []
    for m in chat:
        role = "user" if m.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m.get("content") or "")]))
    cfg = types.GenerateContentConfig(
        system_instruction=system_text,
        max_output_tokens=_REASONING_MAX_COMPLETION_TOKENS,
        # one-tool-per-turn at the API level (gated by _USE_STOP); Gemini strips
        # the matching close tag from the text — `_heal_close_tag` re-appends it.
        stop_sequences=(list(_TOOL_CLOSE_TAGS) if _USE_STOP else None),
    )
    try:
        resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    except Exception as e:
        print(f"[Trial {trial_id}] Gemini API error on {model}: {e}", flush=True)
        raise
    content = ""
    thinking = ""
    for cand in (getattr(resp, "candidates", None) or []):
        for part in getattr(getattr(cand, "content", None), "parts", []) or []:
            if getattr(part, "thought", False):
                thinking += getattr(part, "text", "") or ""
            else:
                content += getattr(part, "text", "") or ""
    content = _heal_close_tag(content)
    u = getattr(resp, "usage_metadata", None)
    breakdown = {
        "prompt_tokens": int(getattr(u, "prompt_token_count", 0) or 0),
        "prompt_cached_tokens": int(getattr(u, "cached_content_token_count", 0) or 0),
        "completion_tokens": int(getattr(u, "candidates_token_count", 0) or 0),
        "reasoning_tokens": int(getattr(u, "thoughts_token_count", 0) or 0),
        "total_tokens": int(getattr(u, "total_token_count", 0) or 0),
        "finish_reason": str(getattr((resp.candidates[0] if resp.candidates else None), "finish_reason", "stop")),
        "model": model, "api_source": "go",
    }
    if "MAX_TOKENS" in breakdown["finish_reason"].upper():
        print(f"[Trial {trial_id}] WARNING: {model} hit max_output_tokens "
              f"(candidates={breakdown['completion_tokens']}). Tool body likely truncated.",
              flush=True)
    return content, (thinking or None), breakdown


def call_llm_api(messages, model_name, keys=keys, temperature=0.4, trial_info=None):
    """Dispatch a chat-completion request to the right provider.

    Returns:
        (content: str, reasoning_content: str | None, breakdown: dict)
    """
    api_source, full_model_name = resolve_model_and_source(model_name, keys)
    trial_id = trial_info.get('trial_id', "unknown") if trial_info else "unknown"

    is_reasoning = model_name in _REASONING_MODELS

    if api_source == 'an':
        return _call_anthropic(messages, full_model_name, keys, trial_id)
    if api_source == 'go':
        return _call_gemini(messages, full_model_name, keys, trial_id)

    is_deepseek = api_source == 'ds'
    is_dmxapi = api_source == 'dm'
    is_local = api_source == 'lo'
    is_openrouter = api_source == 'or'
    kwargs: dict = {"model": full_model_name, "messages": messages}
    if is_deepseek:
        # DeepSeek (OpenAI-compatible) uses plain `max_tokens`, not
        # `max_completion_tokens`. `deepseek-chat` accepts `temperature` and
        # (per docs) `stop`; `deepseek-reasoner` returns the chain-of-thought
        # in `message.reasoning_content` and does NOT accept `temperature`.
        if is_reasoning:
            # Bumped from 16k to 65k — V4 reasoning traces alone can run 10-20k
            # tokens; the visible tool body needs additional headroom, otherwise
            # `finish_reason=length` truncates `<python>` / `<final_formula>`.
            kwargs["max_tokens"] = 65536
        else:
            kwargs["max_tokens"] = 8192
            kwargs["temperature"] = temperature
        # Always try `stop` first; retry-without on rejection (see below).
        kwargs["stop"] = list(_TOOL_CLOSE_TAGS)
    elif is_dmxapi:
        # DMXAPI gateway is OpenAI-compatible and uses plain `max_tokens`.
        # GLM-5.1 includes its reasoning trace inside `completion_tokens`, so
        # a small budget produces empty visible content (reasoning eats it
        # all). 32k covers reasoning + visible <python> bodies; bump if
        # `finish_reason=length` shows up.
        if is_reasoning:
            kwargs["max_tokens"] = 32768
        else:
            kwargs["max_tokens"] = 8192
            kwargs["temperature"] = temperature
        kwargs["stop"] = list(_TOOL_CLOSE_TAGS)
        if model_name in _THINKING_DISABLED:
            # OpenAI SDK's `extra_body` escape hatch — the body field is
            # forwarded verbatim to the upstream (Z.ai for GLM-5.1).
            kwargs["extra_body"] = _thinking_disable_body(model_name)
    elif is_local:
        # Local vLLM/SGLang OpenAI-compatible servers use plain `max_tokens`.
        # vLLM GLM-5.2 disables thinking through `chat_template_kwargs`.
        if is_reasoning:
            kwargs["max_tokens"] = _REASONING_MAX_COMPLETION_TOKENS
        else:
            kwargs["max_tokens"] = 8192
            kwargs["temperature"] = temperature
        kwargs["stop"] = list(_TOOL_CLOSE_TAGS)
        if model_name in _THINKING_DISABLED:
            kwargs["extra_body"] = _thinking_disable_body(model_name)
    elif is_openrouter:
        # OpenRouter is OpenAI-compatible. `max_tokens` is the unified
        # output-length cap regardless of upstream provider; reasoning
        # models route their reasoning trace into the same budget.
        if is_reasoning:
            kwargs["max_tokens"] = _REASONING_MAX_COMPLETION_TOKENS
        else:
            kwargs["max_tokens"] = 8192
            kwargs["temperature"] = temperature
        kwargs["stop"] = list(_TOOL_CLOSE_TAGS)
        if model_name in _THINKING_DISABLED:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        elif is_reasoning and _OPENAI_REASONING_EFFORT:
            kwargs["extra_body"] = {"reasoning": {"effort": _OPENAI_REASONING_EFFORT}}
        elif model_name.startswith("or-claude") or model_name.startswith("or-gemini"):
            # Explicit reasoning-token budget. OR forwards
            # `reasoning.max_tokens` to Anthropic's `thinking.budget_tokens`
            # and Google's `thinkingBudget`.
            kwargs["extra_body"] = {"reasoning": {"max_tokens": 32768}}
    elif is_reasoning:
        # OpenAI reasoning models: budget covers reasoning + visible answer;
        # do NOT pass temperature (default 1.0; explicit values may be
        # rejected). Always TRY `stop` first to enforce one-tool-per-turn —
        # if the model rejects it (some gpt-5* tiers do), the defensive
        # retry below strips `stop` and retries automatically.
        kwargs["max_completion_tokens"] = _REASONING_MAX_COMPLETION_TOKENS
        if _OPENAI_REASONING_EFFORT:
            kwargs["reasoning_effort"] = _OPENAI_REASONING_EFFORT
        kwargs["stop"] = list(_TOOL_CLOSE_TAGS)
    else:
        kwargs["temperature"] = temperature
        kwargs["stop"] = list(_TOOL_CLOSE_TAGS)

    # Fail fast on a stalled call instead of the SDK default (600s × 2 retries),
    # which can freeze a turn for ~10+ min. A reasoning turn that genuinely needs
    # longer can still finish within this timeout; if not, the agent loop surfaces
    # the error rather than hanging.
    _cl = {"timeout": _LLM_TIMEOUT_SECONDS, "max_retries": 1}
    if is_deepseek:
        client = OpenAI(api_key=keys['ds'], base_url="https://api.deepseek.com", **_cl)
    elif is_dmxapi:
        client = OpenAI(api_key=keys['dm'], base_url=_DMXAPI_BASE, **_cl)
    elif is_local:
        client = OpenAI(api_key=keys.get('lo') or "EMPTY", base_url=_LOCAL_OPENAI_BASE, **_cl)
    elif is_openrouter:
        client = OpenAI(api_key=keys['or'], base_url=_OPENROUTER_BASE, **_cl)
    else:
        client = OpenAI(api_key=keys['oa'], **_cl)

    def _do_call(call_kwargs):
        return client.chat.completions.create(**call_kwargs)

    # Drop `stop=` when globally disabled, or for models already known (this
    # process) to reject it — so we don't re-pay the rejected first call.
    if (not _USE_STOP and not is_local) or full_model_name in _STOP_UNSUPPORTED:
        kwargs.pop("stop", None)

    try:
        completion = _do_call(kwargs)
    except Exception as e:
        # Defensive retry: if the provider rejected `stop` (e.g. some
        # DeepSeek models on some API versions), try once more without it.
        # Narrowly scoped: must be a 400-class error AND mention `stop`
        # specifically. Auth/quota/network errors fall through to the raise.
        msg_lower = str(e).lower()
        looks_like_stop_rejection = (
            "stop" in kwargs
            and ("'stop'" in msg_lower or '"stop"' in msg_lower
                 or "parameter stop" in msg_lower or "stop parameter" in msg_lower
                 or "stop sequences" in msg_lower or "stop is not supported" in msg_lower)
        )
        if looks_like_stop_rejection:
            _STOP_UNSUPPORTED.add(full_model_name)  # skip `stop` on later turns
            retry_kwargs = {k: v for k, v in kwargs.items() if k != "stop"}
            print(f"[Trial {trial_id}] {full_model_name} rejected `stop=`; retrying without it "
                  f"(and skipping it for the rest of this run). ({e})",
                  flush=True)
            try:
                completion = _do_call(retry_kwargs)
            except Exception as e2:
                print(f"[Trial {trial_id}] {_provider_name(api_source)} API error on {full_model_name}: {e2}")
                raise
        else:
            print(f"[Trial {trial_id}] {_provider_name(api_source)} API error on {full_model_name}: {e}")
            raise

    msg = completion.choices[0].message
    content = _heal_close_tag(msg.content)
    reasoning_content = (
        getattr(msg, 'reasoning_content', None)
        or getattr(msg, 'reasoning', None)
    )
    finish_reason = completion.choices[0].finish_reason
    usage = getattr(completion, 'usage', None)
    # Build a breakdown dict — captures input/output/reasoning/cache so the
    # trial JSON can compute cost retroactively. Falls back to 0 for any
    # missing field. Backward-compat: callers that read this as `int(t)`
    # would break, but our agent.py reads keys explicitly; legacy callers
    # can fetch `total["completion_tokens"]`.
    def _g(o, k, default=0):
        v = getattr(o, k, None) if o is not None else None
        try: return int(v) if v is not None else default
        except (TypeError, ValueError): return default

    p_details = getattr(usage, 'prompt_tokens_details', None)
    c_details = getattr(usage, 'completion_tokens_details', None)
    breakdown = {
        "prompt_tokens": _g(usage, 'prompt_tokens'),
        "prompt_cached_tokens": (
            _g(p_details, 'cached_tokens')
            or _g(usage, 'prompt_cache_hit_tokens')  # DeepSeek-specific field
        ),
        "completion_tokens": _g(usage, 'completion_tokens'),
        "reasoning_tokens": _g(c_details, 'reasoning_tokens'),
        "total_tokens": _g(usage, 'total_tokens'),
        "finish_reason": finish_reason,
        "model": full_model_name,
        "api_source": api_source,
    }

    if finish_reason == "length":
        budget = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        print(f"[Trial {trial_id}] WARNING: {full_model_name} hit output token cap "
              f"(finish_reason=length, budget={budget}, completion={breakdown['completion_tokens']}, "
              f"reasoning={breakdown['reasoning_tokens']}). Tool body likely truncated.",
              flush=True)
    elif (content is None or content == "") and is_reasoning:
        print(f"[Trial {trial_id}] WARNING: empty visible content from {full_model_name}; "
              f"finish_reason={finish_reason}  completion={breakdown['completion_tokens']}  "
              f"reasoning={breakdown['reasoning_tokens']}.",
              flush=True)
    return content, reasoning_content, breakdown


# Self-test entry point
if __name__ == '__main__':
    sample = [{"role": "user", "content": "Reply with the single word: OK."}]
    for alias in api_source_mapping:
        try:
            src, full = resolve_model_and_source(alias)
            print(f"--- {alias} ({src}/{full}) ---")
            content, reasoning, tokens = call_llm_api(sample, alias)
            print(f"  OK {content!r}  tokens={tokens}")
        except Exception as e:
            print(f"  ERR {e}")
