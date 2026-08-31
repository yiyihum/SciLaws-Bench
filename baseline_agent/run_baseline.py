"""Baseline LLM-as-agent solver for SciLaws-Bench.

Runs the multi-turn equation-discovery agent on ONE public task and writes a
submission module (the agent's `<final_formula>`). In fixed-data mode, it can
optionally call the fixed-data numeric evaluator. In simulator/parallel mode,
evaluation is structure-only and is handled separately by
`harness/evaluate_parallel.py`. The agent sees ONLY the public task context; in
simulator mode it must collect observations through `<experiment>`.

Usage:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / OPENROUTER_API_KEY …
    python run_baseline.py <task_dir> <model> [options]

    <task_dir>   path to a public task, e.g.
                 ../tasks/typeI/cepheid_period_luminosity__M_W
    <model>      model alias (see call_llm_api.py: gpt5, gpt5mini,
                 claude-opus-4-7, gemini-3.1-pro, deepseek-reasoner, …)

Options:
    --max-turns N   agent turn budget (default 30)
    --out DIR       where to write <task_id>.py (default: ./submissions)
    --simulator     run in simulator-backed mode and enable <experiment>
    --score         fixed-data mode only: score it with the sibling numeric
                    harness (requires the private scoring/ tree to be present)

Batch all tasks:
    for d in ../tasks/typeI/*/ ../tasks/typeII/*/ ; do
        python run_baseline.py "$d" gpt5mini --out submissions
    done
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task import load_task                     # noqa: E402
from agent import conduct_exploration          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", help="path to a public task dir (tasks/<type>/<task>)")
    ap.add_argument("model", help="model alias (see call_llm_api.py)")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--out", default="submissions", help="output dir for <task_id>.py")
    ap.add_argument("--traj-out", default=None,
                    help=("directory for per-turn trajectory checkpoints "
                          "(default: same as --out)"))
    ap.add_argument("--simulator", nargs="?", const="simulator", default=None,
                    help=("enable simulator-backed mode. With current tasks, pass "
                          "`--simulator`; old named layouts may pass a simulator name."))
    ap.add_argument("--score", action="store_true",
                    help="score the submission with the sibling harness (needs scoring/)")
    ap.add_argument("--include-test-range", dest="include_test_range",
                    action="store_true", default=None,
                    help="include public metadata input train->test ranges in the task prompt (fixed real-data mode default)")
    ap.add_argument("--no-include-test-range", dest="include_test_range",
                    action="store_false",
                    help="omit public metadata input train->test ranges from the task prompt")
    args = ap.parse_args()

    include_test_range = (
        (args.simulator is None)
        if args.include_test_range is None
        else bool(args.include_test_range)
    )
    if args.simulator is not None:
        include_test_range = False

    task = load_task(
        args.task_dir,
        simulator=args.simulator,
        show_test_range=include_test_range,
    )
    task_type = "typeII" if task.has_group_id else "typeI"
    mode = "simulator" if args.simulator is not None else "fix"
    objective = (
        "objective=structure"
        if args.simulator is not None
        else f"metric={task.headline_metric}"
    )
    print(f"Task: {task.task_id}  type={task_type}  model={args.model}  "
          f"mode={mode}  {objective}  "
          f"train_rows={len(task.train)}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(args.traj_out) if args.traj_out else out_dir
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{task.task_id}.traj.json"

    def write_checkpoint(trial: dict) -> None:
        payload = {
            "meta": {
                "task_id": task.task_id,
                "task_dir": str(task.task_dir),
                "task_type": task_type,
                "mode": mode,
                "model": args.model,
                "max_turns": args.max_turns,
                "include_test_range": include_test_range,
                "checkpoint_path": str(traj_path),
                "updated_at_unix": time.time(),
            },
            "trial": trial,
        }
        tmp_path = traj_path.with_suffix(traj_path.suffix + ".tmp")
        with tmp_path.open("w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, traj_path)

    t0 = time.time()
    initial_train_rows = len(task.train)
    trial = conduct_exploration(task, model_name=args.model, max_turns=args.max_turns,
                                trial_info={"trial_id": f"{args.model}_{task.task_id}"},
                                checkpoint_fn=write_checkpoint)
    eq = trial.get("submitted_equation") or ""
    print(f"\n=== agent done ({time.time()-t0:.0f}s, status={trial.get('status')}, "
          f"rounds={trial.get('rounds')}, tokens={trial.get('total_tokens')}, "
          f"experiments={trial.get('n_experiments', 0)}, "
          f"python_calls={trial.get('n_python_calls', 0)}, "
          f"active_rows={max(0, len(task.train) - initial_train_rows)}) ===")

    if not eq.strip():
        print("agent produced no <final_formula>; nothing written.")
        sys.exit(1)

    out_path = out_dir / f"{task.task_id}.py"
    out_path.write_text(eq)
    print(f"submission written: {out_path}")
    print(f"trajectory checkpoint: {traj_path}")

    if args.score and args.simulator is not None:
        raise SystemExit(
            "--score is fixed-data only. For simulator/parallel runs, use "
            "harness/evaluate_parallel.py to produce structure_score."
        )

    if args.score:
        harness = Path(__file__).resolve().parent.parent / "harness"
        sys.path.insert(0, str(harness.parent))
        from harness import evaluate_on_test     # noqa: PLC0415
        res = evaluate_on_test(eq, task)
        ns = res.get("numeric_score")
        print(f"\nnumeric_score = {ns if ns is None else round(ns, 4)}  "
              f"(metric={res.get('metric')}, contract_ok={res.get('contract_ok')}, "
              f"status={res.get('status')})")


if __name__ == "__main__":
    main()
