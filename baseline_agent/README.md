# Baseline agent — LLM-as-agent symbolic-regression solver

A reference solver for RealSR v3: a multi-turn LLM agent that inspects each
task's training data in a Python sandbox, fits constants, and submits a formula
module. It reads ONLY the public task (`tasks/<type>/<task>/`) — context, inputs,
target, scoring metric, and `data/train.csv` — never the private `scoring/` tree.

The FIXED interface (system prompt, tool-call protocol, submission contract)
lives in the harness and is shared by every solver — see
**`../harness/AGENT_INTERFACE.md`** (read this to understand how to test / how to
plug in your own agent, evolving or otherwise).

- **`run_baseline.py`** — run the agent on one task → writes a submission module.
- `agent.py` — the turn loop (a ~40-line wrapper over
  `harness/agent_protocol.step`); `task.py` (public-task loader);
  `call_llm_api.py` (multi-provider LLM client); `utils.py`.

## Requirements

```
pip install numpy scipy pandas pyyaml openai anthropic google-genai
```

Set the API key for whichever provider your model uses (read from the
environment by `call_llm_api.py`):

```
export OPENAI_API_KEY=...        # gpt5, gpt5mini, …
export ANTHROPIC_API_KEY=...     # claude-opus-4-7, …
export OPENROUTER_API_KEY=...    # or-… aliases
export GOOGLE_API_KEY=...        # gemini-3.1-pro
```

## Run one task

```
cd baseline_agent
python run_baseline.py ../tasks/typeI/cepheid_period_luminosity__M_W gpt5mini
# → writes submissions/cepheid_period_luminosity__M_W.py
```

Add `--score` to immediately score it with the sibling harness (needs the
private `scoring/` tree present):

```
python run_baseline.py ../tasks/typeI/cepheid_period_luminosity__M_W gpt5mini --score
```

Options: `--max-turns N` (default 30), `--out DIR` (default `submissions`).

## Run the whole benchmark

```
cd baseline_agent
for d in ../tasks/typeI/*/ ../tasks/typeII/*/ ; do
    python run_baseline.py "$d" gpt5mini --out submissions
done
```

Then score every submission (numeric):

```
for d in ../tasks/typeI/*/ ../tasks/typeII/*/ ; do
    t=$(basename "$d")
    python ../harness/evaluate_numeric.py score "$d" "submissions/$t.py" > "numeric_out/$t.json"
done
```

and run validity (`../harness/VALIDITY_JUDGE.md`, cc subagent). Report the two
score columns side by side — there is no weighted total (see the benchmark
README → *How scores are defined*).

## Models

Aliases are resolved in `call_llm_api.py` (OpenAI / Anthropic / Google /
DeepSeek / OpenRouter). Use any alias listed there, e.g. `gpt5`, `gpt5mini`,
`claude-opus-4-7`, `gemini-3.1-pro`, `deepseek-reasoner`.
