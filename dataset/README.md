---
license: other
license_name: mixed-per-task
license_link: https://github.com/TODO/SciLaws-Bench/blob/main/dataset/LICENSES.md
pretty_name: SciLaws-Bench
task_categories:
  - tabular-regression
language:
  - en
tags:
  - symbolic-regression
  - scientific-discovery
  - equation-discovery
  - benchmark
  - llm-agents
size_categories:
  - 1M<n<10M
configs:
  - config_name: default
    data_files:
      - split: train
        path: tasks/*/*/data/train.csv
---

# SciLaws-Bench

Data for **Can LLMs Discover Scientific Laws in Real and Parallel Worlds?**

118 scientific law-discovery problems curated from 381 papers, covering 291 candidate laws
and ~8.2M real measurements across six disciplines. Every problem comes from active,
data-driven literature where the published law still leaves room for improvement — not from
textbook equations.

- Code and harness: https://github.com/TODO/SciLaws-Bench
- Project page: https://TODO
- Paper: https://TODO

## Two settings per problem

**SciLaws-Real** — propose a closed-form law from fixed published records. Published
formulas are reference baselines to beat, not recovery targets. Scored on numeric fit
`S_N` (best published formula = 0.5, perfect predictor = 1.0) and scientific validity
`S_V` (a frozen, source-grounded rubric behind an anti-hacking gate).

**SciLaws-Parallel** — the same problem as a queryable simulator whose generating
mechanism is a newly synthesized structural variant of the published form, absent from the
literature. Scored on structure recovery `S_S` at five levels: 0 unrelated, 0.25 right
variables or trends, 0.5 published base form, 0.75 base plus most added terms, 1.0 the
complete hidden structure.

## Composition

| Discipline | Tasks |
|---|---:|
| Earth & Physics | 24 |
| Ecology & Hydrology | 22 |
| Astronomy | 20 |
| Materials & Engineering | 19 |
| Social Sciences | 17 |
| Biology | 16 |

66 tasks are Type I (single-group: one global law), 52 are Type II (multi-group: one shared
functional form, a few per-group parameters). All 118 ship a calibrated simulator.

`task_index.csv` in this repository lists every task with its discipline, target variable,
input count, row counts and license.

## Layout

```
tasks/typeI/<task>/
├── metadata.yaml            # solver-facing: context, target, inputs, units, ranges
├── data/
│   ├── train.csv            # column 0 = target, columns 1..N = inputs
│   └── test.csv
├── eval/
│   ├── reference_metrics.json   # published-baseline anchors used to normalize S_N
│   ├── validity_rubrics.json    # frozen source-grounded validity rubric
│   └── metadata_full.yaml
└── simulator/
    ├── state.joblib         # simulator state, loaded by harness/sim_runtime.py
    ├── sample.csv           # fixed free sample
    └── formula.py           # the hidden law — grader-only

tasks/typeII/<task>/
└── data/{train,test_fit,test_test}.csv    # otherwise identical
```

For Type II, `test_fit` calibrates per-group parameters on held-out groups and `test_test`
scores the shared functional form on those same groups. `predict()` never receives
`group_id`.

> **`eval/` and `simulator/` are grader-facing — and this release publishes them.**
> `eval/` holds the reference anchors and the frozen validity rubric. `simulator/`
> holds the Parallel setting's hidden law: not only in `formula.py`, but inside
> `state.joblib` itself, whose `formula_source` field is the law's source text that
> `sim_runtime` executes to generate `y`. Removing it does not hide the law, it breaks
> the simulator. So a locally runnable Parallel world necessarily ships its own answer.
>
> If you evaluate a system on SciLaws-Parallel, withhold `eval/` and `simulator/` from
> it and mediate every query through `harness/sim_runtime.py`. Treat scores obtained by
> a system that had filesystem access to these directories as invalid.

## Usage

```bash
hf download <ORG>/SciLaws-Bench --repo-type dataset --local-dir . --include 'tasks/*'
```

Load one task directly:

```python
import pandas as pd, yaml

task = "tasks/typeI/mauna_loa_co2_keeling_curve_noaa__co2_ppm"
meta = yaml.safe_load(open(f"{task}/metadata.yaml"))
train = pd.read_csv(f"{task}/data/train.csv")
y, X = train.iloc[:, 0], train.iloc[:, 1:]
```

Scoring is done by the harness in the code repository, not by a metric in this dataset.

## Licensing

Each task carries the license of its upstream dataset — 53 CC-BY-4.0, 22 CC0-1.0, and a
long tail of public-domain, MIT, GPL, ODbL and agency-specific terms. Datasets that
disallow redistribution (NDA, paywalled, research-use-only) were excluded during curation.
Per-task terms are in `LICENSES.md` and in each task's `metadata.yaml`. Cite the upstream
source when you use a task's data.

## Citation

```bibtex
@article{huang2026scilaws,
  title   = {Can LLMs Discover Scientific Laws in Real and Parallel Worlds?},
  author  = {Huang, Yiming and Liu, Ziche and Wu, Zhuohang and Wang, Yiqian and
             Cui, Junxia and Zou, Xinkai and Mao, Linjun and Huang, Nan and
             Yu, Naicheng and Zhu, Kaijie and Ma, Yue and Zhou, Kun and
             Peng, Letian and Shang, Jingbo},
  journal = {arXiv preprint},
  year    = {2026}
}
```
