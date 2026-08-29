"""Warm-resume an interrupted baseline trajectory (status=running /
running_llm_response) from its saved checkpoint, continuing the same
conversation until it submits or hits the turn budget.

Unlike run_baseline.py (which always restarts from turn 0), this seeds the
agent with the checkpoint's chat_history and per-turn counters, so the spent
turns are preserved. Only meaningful for fixed-data tasks (the <python> sandbox
is stateless across turns, so the resumed execution environment is identical to
an uninterrupted run).

Usage:
    python resume_baseline.py <traj.json> --model or-glm52 \
        --out submissions/typeI --traj-out trajectories/typeI [--max-turns 30]
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

RESUMABLE = {"running", "running_llm_response"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", help="path to an interrupted <task>.traj.json checkpoint")
    ap.add_argument("--model", default=None, help="model alias (default: meta.model)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="turn budget (default: meta.max_turns)")
    ap.add_argument("--out", default="submissions", help="dir for <task_id>.py")
    ap.add_argument("--traj-out", default=None, help="dir for updated <task_id>.traj.json")
    ap.add_argument("--force", action="store_true",
                    help="resume even if status is not running*")
    ap.add_argument("--drop-pending", action="store_true",
                    help="discard a trailing un-stepped assistant reply before "
                         "resuming (use when its <python> hangs in native code); "
                         "the model regenerates that turn instead of re-executing it")
    args = ap.parse_args()

    payload = json.load(open(args.traj))
    meta = payload["meta"]
    trial = payload["trial"]
    status = trial.get("status") or ""
    task_id = meta["task_id"]

    if (trial.get("submitted_equation") or "").strip():
        print(f"[skip] {task_id}: already has a submission")
        sys.exit(0)
    if status not in RESUMABLE and not args.force:
        print(f"[skip] {task_id}: status={status!r} not resumable (use --force)")
        sys.exit(0)

    model = args.model or meta["model"]
    max_turns = args.max_turns if args.max_turns is not None else int(meta["max_turns"])
    include_test_range = bool(meta.get("include_test_range", True))
    rounds = int(trial.get("rounds") or 0)
    chat_history = trial.get("chat_history") or []

    if args.drop_pending and chat_history and chat_history[-1]["role"] == "assistant":
        chat_history = chat_history[:-1]
        rounds = max(0, rounds - 1)
        print(f"[drop-pending] discarded trailing assistant turn; "
              f"regenerating from rounds={rounds}")

    if rounds >= max_turns:
        print(f"[skip] {task_id}: rounds={rounds} >= max_turns={max_turns}; nothing to resume")
        sys.exit(0)

    task = load_task(meta["task_dir"], simulator=None, show_test_range=include_test_range)
    task_type = "typeII" if task.has_group_id else "typeI"

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(args.traj_out) if args.traj_out else Path(args.traj).parent
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"{task_id}.traj.json"

    def write_checkpoint(t: dict) -> None:
        p = {"meta": {**meta, "mode": "resume", "model": model,
                      "max_turns": max_turns, "resumed_from_rounds": rounds,
                      "checkpoint_path": str(traj_path), "updated_at_unix": time.time()},
             "trial": t}
        tmp = traj_path.with_suffix(traj_path.suffix + ".tmp")
        with tmp.open("w") as fh:
            json.dump(p, fh, indent=2, sort_keys=True); fh.write("\n")
        os.replace(tmp, traj_path)

    resume_state = {
        "n_experiments": int(trial.get("n_experiments") or 0),
        "n_python_calls": int(trial.get("n_python_calls") or 0),
        "usage_total": trial.get("usage_total") or {},
        "usage_per_turn": trial.get("usage_per_turn") or [],
        "start_turn": rounds,
    }

    print(f"Resuming {task_id} type={task_type} model={model} "
          f"from rounds={rounds} -> max_turns={max_turns} (status={status})", flush=True)
    t0 = time.time()
    result = conduct_exploration(
        task, model_name=model, max_turns=max_turns,
        trial_info={"trial_id": f"{model}_{task_id}"},
        checkpoint_fn=write_checkpoint,
        resume_messages=chat_history, resume_state=resume_state)

    eq = (result.get("submitted_equation") or "").strip()
    print(f"=== resume done ({time.time()-t0:.0f}s, status={result.get('status')}, "
          f"rounds={result.get('rounds')}, +turns={result.get('rounds', 0) - rounds}) ===")
    if not eq:
        print(f"[fail] {task_id}: still no <final_formula> after resume")
        sys.exit(1)
    out_path = out_dir / f"{task_id}.py"
    out_path.write_text(eq)
    print(f"submission written: {out_path}")


if __name__ == "__main__":
    main()
