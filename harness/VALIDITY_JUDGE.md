# Validity judge — staging, dispatch and aggregation

`validity_score` for a SciLaws-Bench submission. A code-enabled judge
**executes** the submitted formula on the task data and scores the task's frozen
task `validity_rubrics` plus one staged global anti-hacking rubric, returning
`M/N`.

This is the official validity method; it replaces `eval_consistency`.
The deterministic `numeric_score` is the separate channel (`evaluate_numeric.py`); this
scores physical / functional validity.

The judge never touches the live task tree. First run the fixed staging script
(`harness/evaluate_validity.py`). It creates self-contained task dirs and writes one
subagent prompt per chunk. Then run those prompts with a code-enabled judge.

```
{{STAGE_DIR}}/<task>/
    metadata.yaml          # public task spec (target, inputs, type) — no rubrics
    validity_rubrics.json  # frozen task rubrics plus staged anti-hacking rubric
    submission.py          # the solver's module (predict(X), USED_INPUTS, [fit])
    data/                  # the whole data dir, copied verbatim:
                           #   Type I : train.csv, test.csv
                           #   Type II: train.csv, test_fit.csv, test_test.csv
```

---

## 1. Stage (fixed script)

Provide a submissions directory. It can be either:

- `submissions/<task>.py` for one method.
- `submissions/<method>/<task>.py` for multiple methods. The staged id becomes
  `<method>__<task>` so outputs do not collide.

Stage only:

```bash
python harness/evaluate_validity.py \
  --tasks-dir tasks \
  --submissions submissions \
  --stage-root validity_stage \
  --output-root validity_out \
  --chunk-size 3
```

Stage and immediately dispatch Codex codeagent chunks:

```bash
python harness/evaluate_validity.py \
  --tasks-dir tasks \
  --submissions submissions \
  --stage-root validity_stage \
  --output-root validity_out \
  --chunk-size 3 \
  --dispatch codex \
  --max-workers 4
```

The script:

- copies `metadata.yaml`, `data/`, `submission.py`, and
  `validity_rubrics.json` into each staged task dir;
- appends one global constant-discipline / no-cap-evasion rubric to the staged
  rubric list, without rewriting the source task `eval/` files;
- reads rubrics from `tasks/<type>/<task>/eval/validity_rubrics.json`, with
  fallback to `scoring/<type>/<task>/validity_rubrics.json`;
- writes `manifest.json`;
- writes prompt files under `{{STAGE_DIR}}/prompts/chunk_###.md`.
- with `--dispatch codex`, calls `codex exec` once per prompt chunk and writes
  logs under `{{OUTPUT_DIR}}/agent_logs/`.

Example output:

```text
STAGE_DIR: /abs/path/validity_stage/20260627_031500
OUTPUT_DIR: /abs/path/validity_out/20260627_031500
PROMPTS_DIR: /abs/path/validity_stage/20260627_031500/prompts
MANIFEST: /abs/path/validity_stage/20260627_031500/manifest.json
staged: 36
skipped: 1
chunks: 12
prompt[001]: /abs/path/validity_stage/20260627_031500/prompts/chunk_001.md
```

---

## 2. Dispatch the judge subagents

- If `--dispatch codex` was used in step 1, this is already done by
  `evaluate_validity.py`.
- Otherwise, spawn one general-purpose Claude-Code/Codex subagent per generated
  prompt file.
- Use the prompt files produced by `evaluate_validity.py` directly. Each prompt
  contains the judging instructions plus the staged `validity_rubrics.json`
  path for every task in that chunk.
- Keep **2–3 tasks per subagent**. Larger chunks stall / hit the ~600s
  stream-idle watchdog. Small chunks finish in ~1–3 min.
- Each subagent writes one JSON per staged id to `{{OUTPUT_DIR}}`; aggregate by
  reading that dir. Do not rely on the agent's return text.

For example, assign this file to one subagent:

```text
{{STAGE_DIR}}/prompts/chunk_001.md
```

The script's Codex dispatch is equivalent to:

```bash
codex exec \
  -C /path/to/SciLaws-Bench \
  -s workspace-write \
  --json \
  -o {{OUTPUT_DIR}}/agent_logs/chunk_001.last.txt \
  - < {{STAGE_DIR}}/prompts/chunk_001.md \
  > {{OUTPUT_DIR}}/agent_logs/chunk_001.jsonl 2>&1
```

If `{{STAGE_DIR}}` or `{{OUTPUT_DIR}}` is outside the repo, the script
automatically adds `codex exec --add-dir` for those paths. The default
`validity_stage/` and `validity_out/` roots are inside the repo.

The generated prompt tells the subagent to write:

```json
{
  "task": "<stage_id>",
  "n_satisfied": 4,
  "n_total": 5,
  "validity_score": 0.8,
  "error": null,
  "rubrics": [
    {
      "i": 1,
      "verdict": "Y",
      "kind": "behavioral",
      "evidence": "one-line computed or source evidence"
    }
  ]
}
```

---

## 3. Aggregate

```python
import json, glob
scores = {json.load(open(f))["task"]: json.load(open(f))["validity_score"]
          for f in glob.glob("{{OUTPUT_DIR}}/*.json")}
vals = [(v if isinstance(v, (int, float)) else 0.0) for v in scores.values()]
print(len(scores), "tasks;  mean =", round(sum(vals) / max(1, len(vals)), 3))
```

---

## 4. Cleanup (delete the temp dirs)

```python
import shutil
shutil.rmtree("{{STAGE_DIR}}")        # the printed STAGE_DIR
```

---

## Scope note

Best for **behavioral / numerically-checkable** rubrics (monotonicity, sign,
bound, limit, separability) — those become deterministic. **Structural /
semantic** rubrics ("captures mechanism X") still carry a judgment component.
The constant-discipline rubric is also a judgment rubric: the codeagent should
inspect the final submitted source and metadata caps, not apply a mechanical
numeric-literal rule. It should mark that rubric unsatisfied for training/test
aggregates, lookup tables, profiles, large literal arrays, or obvious attempts
to hide fitted degrees of freedom outside the stated caps. The judge still
writes the direct rubric fraction; `evaluate_validity.py` treats this rubric as
a hard gate during aggregation, setting final `validity_score` to 0 when the
constant-discipline verdict is `N`.
`validity_score` is reported alongside `numeric_score`; there is no weighted
total (see the benchmark README → *How scores are defined*).
