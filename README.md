<div align="center">

# SciLaws-Bench

**Can LLMs Discover Scientific Laws in Real and Parallel Worlds?**

118 law-discovery problems curated from 381 scientific papers · 291 candidate laws · ~8.2M real data points · 6 disciplines

[Project page](https://yiyihum.github.io/SciLaws-Bench/) · [Paper (PDF)](docs/assets/scilaws_bench_paper.pdf) · [Dataset on Hugging Face](https://huggingface.co/datasets/RealSR/SciLaws-Bench)

</div>

---

## What this is

Scientific law discovery asks for a closed-form expression that both predicts held-out
measurements and survives the scientific constraints of its field. SciLaws-Bench builds
that task out of published research rather than textbook equations: every problem comes
from active, data-driven literature where the published law still leaves room for
improvement, and ships with the real dataset it was fitted to.

Each problem is instantiated in two settings:

| | **SciLaws-Real** | **SciLaws-Parallel** |
|---|---|---|
| Evidence | fixed published records | a queryable, residual-calibrated simulator |
| Target | no ground truth; published formulas are reference baselines to beat | a synthesized hidden law, absent from the literature |
| Scores | `S_N` numeric fit (best published formula = 0.5, perfect = 1.0) and `S_V` scientific validity (source-grounded rubric behind an anti-hacking gate) | `S_S` structure recovery on five levels, 0 → 1 |

66 problems are **Type I / single-group** (one global law and parameter set). 52 are
**Type II / multi-group**: related groups share one functional form while a few parameter
values differ, so a law has to transfer across groups rather than be refit per group.

## Repository layout

```
SCILAWS-BENCH/
├── docs/                 # project homepage (GitHub Pages)
├── harness/              # scorers, simulator runtime, agent protocol, prompts
├── baseline_agent/       # the reference ReAct agent behind every reported number
├── dataset/
│   ├── README.md         # Hugging Face dataset card
│   ├── task_index.csv    # 118 tasks: discipline, target, row counts, license
│   └── LICENSES.md       # per-task upstream data licenses
├── scripts/              # dataset export, local site preview
├── requirements.txt      # core; requirements-agent.txt adds the agent's client
└── tasks/                # ← downloaded from HF (RealSR/SciLaws-Bench); not tracked here
```

`harness/` and `baseline_agent/` resolve paths relative to each other and to a sibling
`tasks/`, so keep this layout and download the data into `tasks/` at the repository root.

## Quick start

Every command below is tested as written.

```bash
# 1. Code (this repo). Python 3.10+.
git clone https://github.com/yiyihum/SciLaws-Bench.git
cd SciLaws-Bench
pip install -r requirements.txt          # add -r requirements-agent.txt to run the agent

# 2. Data — 118 tasks, ~1.1 GB, from https://huggingface.co/datasets/RealSR/SciLaws-Bench
hf download RealSR/SciLaws-Bench --repo-type dataset --local-dir . --include 'tasks/*'

# 3. Score a submission on one task (writes JSON to stdout)
python harness/evaluate_numeric.py score \
    tasks/typeI/hea_hardness_lattice_distortion_couzinie__HV \
    my_submission.py
```

A minimal `my_submission.py` (the full contract is one section down):

```python
import numpy as np
USED_INPUTS = ["VEC", "dHmix"]
LAW_CONSTANTS = {}
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}
def predict(X):
    return 120.0 + 80.0 * (X[:, 0] - 5.0)
```

The scorer replies with, among other fields:

```json
{"status": "ok", "contract_ok": true, "numeric_score": 0.0, "raw_metric": 424.0789795633554,
 "score": {"metric": "rmse", "best_reference_id": "temesi_2023_hardness_bonding"}}
```

`numeric_score` is reference-relative: the strongest published formula anchors 0.5, a
perfect predictor 1.0. The stub above is a naive linear guess, so it scores 0.0;
GPT-5.5's actual panel submission for this task scores 0.508.

To open a task's **Parallel world** and query it directly:

```python
import sys; sys.path.insert(0, "harness")
import sim_runtime
sim = sim_runtime.load("tasks/typeI/hea_hardness_lattice_distortion_couzinie__HV/simulator")
r = sim.fetch_where(query="VEC > 4.5 and VEC < 6", limit=5, seed=1)
print(r["n_returned"], r["HV"], r["budget"])   # noisy observations + remaining query budget
```

Each task directory is self-describing:

```
tasks/typeI/<task>/                    # single-group
├── metadata.yaml                      # solver-facing: context, target, inputs, units, ranges
├── data/{train,test}.csv              # column 0 = target, columns 1..N = inputs
├── eval/                              # grader-facing: reference metrics, validity rubrics,
│                                      #   metadata_full.yaml (which published law each
│                                      #   baseline id comes from)
└── simulator/                         # the Parallel world: state.joblib, sample.csv, formula.py

tasks/typeII/<task>/                   # multi-group — data/ is {train,test_fit,test_test}.csv
```

For Type II, the harness re-fits per-group free parameters on each held-out group's
`test_fit.csv` using your `fit()`, then scores `predict()` on `test_test.csv`.
`predict()` never receives `group_id`.

> **`eval/` and `simulator/` are answer keys.** `eval/` holds the reference anchors and
> the frozen validity rubric; `simulator/state.joblib` embeds the Parallel hidden law as
> `formula_source`, which `sim_runtime` executes to generate `y` — so it cannot be
> stripped without breaking the simulator. When evaluating a system, withhold both
> directories from it and mediate every query through `harness/sim_runtime.py`.

## Submission contract

One Python module per task:

```python
USED_INPUTS = ["col_a", "col_b"]   # data columns used, in X-column order
LAW_CONSTANTS = {}                 # global constants
OTHER_CONSTANTS = {}
LOCAL_FITTABLE = {}                # Type II: per-group free params; Type I: {}

def predict(X, **constants):
    ...

def fit(X, y, **LAW_CONSTANTS):    # Type II only, when LOCAL_FITTABLE is non-empty
    ...
    return {"param": value}
```

Type I submissions must not define `fit()`. The full contract is in
[`harness/AGENT_INTERFACE.md`](harness/AGENT_INTERFACE.md).

## Scoring

**Numeric fit** — deterministic, one task at a time. Type II is averaged over three fixed
seeds; Type I runs once.

```bash
python harness/evaluate_numeric.py score tasks/typeI/<task> submissions/<task>.py
```

**Scientific validity** — stages each submission with its metadata, data and frozen
rubric, then dispatches a code-enabled judge.

```bash
python harness/evaluate_validity.py \
  --tasks-dir tasks --submissions submissions \
  --stage-root validity_stage --output-root validity_out \
  --method-name my_method --chunk-size 3 \
  --dispatch codex --max-workers 4 --overwrite
```

Use `--aggregate-only --stage-dir validity_stage/<run_id>` if your own judge already wrote
the per-task JSONs. If the staged anti-hacking rubric returns `N`, aggregation forces that
task's validity score to `0.0`.

**Structure recovery (Parallel)** — compares the submitted formula against the hidden
simulator law.

```bash
python harness/evaluate_parallel.py \
  --tasks-dir tasks --submissions submissions \
  --stage-root parallel_stage --output-root parallel_out \
  --dispatch codex
```

See [`harness/VALIDITY_JUDGE.md`](harness/VALIDITY_JUDGE.md) for the judging protocol.

## Running the reference agent

The agent talks to a model provider, so it needs the client package and a key:

```bash
pip install -r requirements-agent.txt   # adds `openai`; scoring does not need it
export OPENAI_API_KEY=...               # or the key matching your alias: see below
```

```bash
# SciLaws-Real: fixed records
python baseline_agent/run_baseline.py tasks/typeI/<task> <model_alias> --score

# SciLaws-Parallel: active querying against the simulator
python baseline_agent/run_baseline.py tasks/typeI/<task> <model_alias> --simulator
```

Model aliases are resolved in `baseline_agent/call_llm_api.py`, which picks the provider
from the alias and reads its key from the environment (a `.env` file also works if
`python-dotenv` is installed): `OPENAI_API_KEY` for the OpenAI aliases (`gpt4omini`,
`gpt5mini`, `gpt55`, …), `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` / `GEMINI_API_KEY`,
`OPENROUTER_API_KEY`, `DMXAPI_KEY`. The DeepSeek aliases are the exception — they read
`baseline_agent/key`. Default budget is 30 interaction turns (`--max-turns`). Simulator runs are scored by
`harness/evaluate_parallel.py`, not by `--score`.

## Results

Full leaderboard with single/multi-group splits, task explorer and example
trajectories: **https://yiyihum.github.io/SciLaws-Bench/**

Nine frontier models under one fixed harness — a ReAct-style agent with a Python sandbox,
up to 30 turns, and a fixed experimentation budget in the Parallel setting. Only the base
model varies. Values are percentages.

| Base model | Real `S_N` | Real `S_V` | Parallel `S_S` |
|---|---:|---:|---:|
| GPT-5.5 | **50.77** | **81.84** | **58.26** |
| Gemini 3.5 Flash | 46.08 | 74.63 | 50.42 |
| GLM-5.2 | 45.84 | 74.19 | 52.12 |
| Claude Opus 4.8 | 44.94 | 80.24 | 50.00 |
| DeepSeek-V4 Pro | 43.96 | 81.30 | 49.79 |
| Qwen3.7-Max | 43.80 | 72.66 | 50.42 |
| GPT-5-mini | 40.69 | 77.03 | 44.70 |
| GPT-5.4-mini | 37.17 | 72.45 | 42.58 |
| GPT-4o-mini | 20.59 | 56.02 | 33.47 |

Single/multi-group breakdowns are on the project page and in the paper.
`baseline_agent/all_results.md` has the per-split and per-metric breakdown.

## Licensing

The MIT License in `LICENSE` covers the code in `harness/`, `baseline_agent/` and
`scripts/`. Task **data** is not ours to relicense: each task carries the license of its
upstream source, listed per task in [`dataset/LICENSES.md`](dataset/LICENSES.md) and in the
task's own `metadata.yaml`. Terms vary, several are share-alike or non-commercial, so check
the task you use.

## Citation

```bibtex
@article{huang2026scilaws,
  title   = {Can LLMs Discover Scientific Laws in Real and Parallel Worlds?},
  author  = {Huang, Yiming and Liu, Ziche and Wu, Zhuohang and Wang, Yiqian and
             Cui, Junxia and Zou, Xinkai and Mao, Lingjun and Huang, Nan and
             Yu, Naicheng and Zhu, Kaijie and Ma, Yue and Zhou, Kun and
             Peng, Letian and Shang, Jingbo},
  journal = {arXiv preprint},
  year    = {2026}
}
```
