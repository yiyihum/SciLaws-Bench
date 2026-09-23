# SCILAWS-PARALLEL external runner

Runs one base model over all **118 SCILAWS-PARALLEL tasks** (66 single-group + 52
multi-group) with the exact harness behind the paper's Table 2, and packages the
results so we can score them with the official structure judge. **You do not run
the judge.**

This branch (`scilaws-parallel-external-runner`) is upstream `main` @ `9239f66`
plus a thin runner layer. The benchmark itself (prompts, agent loop, `<experiment>`
/ `<python>` / `<final_formula>` protocol, sandbox, query caps, forced final turn,
submission contracts) is the canonical code in `harness/` and `baseline_agent/`,
unmodified; `scripts/preflight.py` verifies that byte-for-byte.

Frozen configuration for this run: **`gpt-5.6-luna`, reasoning effort `medium`,
`max_completion_tokens=65536`, provider-default sampling, no streaming, 30 turns**
(`configs/gpt56_luna.yaml`).

**Shortcut:** `export OPENAI_API_KEY=... && JOBS=40 bash scripts/run_parallel.sh` runs
steps 1–3 and 5 in one command (reusing whatever is already set up) and launches only if
preflight prints `READY`. Re-run it to resume. `DRY_RUN=1` stops before launching;
`TASKS_ARCHIVE=<file>` installs tasks from an archive instead of Hugging Face. The
sections below are the manual equivalent.

## 1. Setup (Python 3.11-3.13; 3.13 recommended)

```bash
git clone -b scilaws-parallel-external-runner https://github.com/yiyihum/SciLaws-Bench.git scilaws-runner
cd scilaws-runner
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements-runner.txt
export OPENAI_API_KEY=sk-...          # only ever from the environment
```

## 2. Task assets (~0.85 GB, solver-side files only)

```bash
python scripts/fetch_tasks.py                                   # pinned Hugging Face revision
# or, if we sent you an archive:
python scripts/fetch_tasks.py --from-archive tasks_scilaws_parallel_118.tar.gz
```

Every file is checked against `manifests/task_hashes.json`. Only `metadata.yaml`,
`data/*.csv` and `simulator/state.joblib` are installed. The hidden-law
`formula.py`, `sample.csv` and `eval/` are never placed in the task tree. Use
`--out DIR` and then `--tasks-dir DIR` on every command below to keep tasks elsewhere.

## 3. Preflight (no paid API calls, ~1 min)

```bash
python scripts/preflight.py --config configs/gpt56_luna.yaml
```

It must end with `READY`. It checks the checkout, pinned versions, canonical file
hashes, the 66/52 task grid and hashes, that every simulator loads, that all 118
prompts equal the Table-2 prompts, the exact API request the client would send
(model string, `reasoning_effort=medium`, 65536 tokens, no sampling overrides,
120 s timeout, 1 retry), and a scripted fake-model run covering retry, resume and
packaging.

## 4. Smoke test (2 tasks, paid; wait for our go-ahead)

```bash
JOBS=2 bash scripts/launch.sh --config configs/gpt56_luna.yaml --run-name smoke_gpt56_luna \
  --tasks mauna_loa_co2_keeling_curve_noaa__co2_ppm optical_dispersion_sellmeier__refractive_index
bash scripts/monitor.sh --run-name smoke_gpt56_luna
```

## 5. Full run

```bash
tmux new -s scilaws                       # the launcher runs in the foreground
JOBS=40 bash scripts/launch.sh --config configs/gpt56_luna.yaml
```

`JOBS` is the number of tasks running at once. Pick what your API quota sustains
(e.g. 40-60). A task makes up to ~31 sequential model calls. Results go to
`runs/gpt56_luna_parallel/`.

**Monitor** (from another shell):

```bash
bash scripts/monitor.sh                 # snapshot
bash scripts/monitor.sh --watch 30      # refresh every 30 s
```

It shows complete/118, running, submitted, model-failed, infra-failed, retries,
429 / timeout counts, token usage and the slowest running tasks.

**Resume**: run the **same launch command** again (after a crash, Ctrl-C, reboot,
or to change `JOBS`). Finished tasks are skipped. Unfinished tasks restart from
turn 0. Parallel tasks are never warm-resumed, because their experiment data lived
in the killed process. Earlier attempts are kept, never overwritten.

**Retries**: only infrastructure failures are retried: API transport errors,
HTTP 429 / 5xx, API timeouts, process crashes, and the per-task wall-clock cap
(`task_timeout_seconds`, 2 h). The limit is 3 counted attempts per task (initial +
2 cold restarts), with back-off. A model outcome is final and never retried: any
submitted formula (even an invalid one), or no formula after 30 turns. A bad key,
unknown model or exhausted quota aborts the launcher without burning attempts.
Fix it and re-run the same command.

## 6. Send results back

```bash
bash scripts/package_results.sh
```

This produces `dist/gpt56_luna_parallel_<UTC>.tar.gz` (prints its sha256). Send us
that file. It contains, for **all 118 tasks** (failed and unsubmitted included):
`MANIFEST.json`, per-task `records/` (status, attempt history, model/endpoint/effort,
commits, task hashes, usage), `trajectories/` (full transcript: messages,
experiment requests + outputs, Python calls + outputs, `experiment_log`, per-turn
usage), `submissions/typeI|typeII/<task>.py`, every attempt directory (logs,
`observations.jsonl` with every returned simulator row), `run_meta.json`,
`audit_report.json` and `SHA256SUMS`. If some tasks are stuck and you must send
early, add `--allow-incomplete`.

## Layout

```
configs/gpt56_luna.yaml      frozen handoff config          configs/selftest_fake.yaml  preflight only
runner/                      launcher, per-task shim, monitor, packager, audit, fake LLM (self-test)
scripts/                     fetch_tasks.py preflight.py launch.sh monitor.sh package_results.sh audit_transcripts.py
manifests/                   118-task grid, task file hashes, canonical source hashes, Table-2 golden prompt hashes, handoff
harness/ baseline_agent/     canonical upstream code (unmodified)
tasks/  runs/  dist/         local data and outputs (git-ignored)
```

## Integrity notes (for us)

The `<python>` sandbox is the canonical in-process AST/lexical blocklist used for
every Table-2 row. It is kept unchanged for protocol parity. It is a lexical
blocklist, not OS isolation, so it cannot guarantee that no route exists. `runner/audit.py`
scans every transcript for filesystem / simulator-state / grader-file access and
for hidden-law text in tool outputs. On the 354 Table-2 Parallel transcripts it
reports 0 leaks and no real access attempts. The packager runs the audit
automatically.
