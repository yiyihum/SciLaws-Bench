"""Read-only progress view of a run directory.

  python -m runner.monitor --config configs/gpt56_luna.yaml [--watch 30]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import DEFAULT_RUNS_DIR, load_config, read_json, task_list  # noqa: E402

LOG_429 = re.compile(r"Error code: 429|RateLimitError")
LOG_TIMEOUT = re.compile(r"APITimeoutError|Request timed out")


def snapshot(run_dir: Path, subset=None) -> str:
    meta = read_json(run_dir / "run_meta.json") or {}
    grid = task_list()
    subset = subset or meta.get("task_subset")
    if subset:
        grid = [t for t in grid if t["task"] in set(subset)]
    counts = {k: 0 for k in ("submitted", "no_submission", "infra_failed", "running",
                             "pending", "not_started")}
    by_type = {"typeI": {}, "typeII": {}}
    retries = n429 = ntimeout = 0
    infra_kinds: dict = {}
    tokens = prompt_tok = compl_tok = 0
    running = []
    now = time.time()
    for t in grid:
        root = run_dir / "tasks" / t["type"] / t["task"]
        st = read_json(root / "state.json") or {}
        s = st.get("status", "not_started")
        counts[s] = counts.get(s, 0) + 1
        by_type[t["type"]][s] = by_type[t["type"]].get(s, 0) + 1
        atts = st.get("attempts", [])
        retries += max(0, len(atts) - 1)
        for a in atts:
            if a.get("outcome") == "infra_error":
                infra_kinds[a.get("infra_kind")] = infra_kinds.get(a.get("infra_kind"), 0) + 1
            log = run_dir / a["dir"] / "attempt.log" if a.get("dir") else None
            if log and log.exists():
                txt = log.read_text(errors="replace")
                n429 += len(LOG_429.findall(txt))
                ntimeout += len(LOG_TIMEOUT.findall(txt))
            traj = read_json(run_dir / a["dir"] / f"{t['task']}.traj.json") if a.get("dir") else None
            u = ((traj or {}).get("trial") or {}).get("usage_total") or {}
            tokens += int(u.get("total_tokens") or 0)
            prompt_tok += int(u.get("prompt_tokens") or 0)
            compl_tok += int(u.get("completion_tokens") or 0)
        if s == "running" and atts:
            a = atts[-1]
            started = a.get("started_unix")
            traj = read_json(run_dir / a["dir"] / f"{t['task']}.traj.json") or {}
            turn = (traj.get("trial") or {}).get("rounds")
            running.append((now - started if started else 0, t, a.get("attempt"), turn))
    total = len(grid)
    terminal = counts["submitted"] + counts["no_submission"] + counts["infra_failed"]
    lines = [
        f"run: {run_dir}",
        f"model={((meta.get('config') or {}).get('model_id'))} "
        f"effort={((meta.get('config') or {}).get('reasoning_effort'))}",
        f"complete {terminal}/{total}   running {counts['running']}   "
        f"pending {counts['pending'] + counts['not_started']}",
        f"  submitted {counts['submitted']}   model-failed (no submission) "
        f"{counts['no_submission']}   infra-failed {counts['infra_failed']}",
        f"  typeI  {dict(sorted(by_type['typeI'].items()))}",
        f"  typeII {dict(sorted(by_type['typeII'].items()))}",
        f"retries (extra attempts) {retries}   infra errors by kind {infra_kinds or '{}'}",
        f"API 429 messages {n429}   API timeout messages {ntimeout}",
        f"tokens (all attempts) total={tokens:,} prompt={prompt_tok:,} completion={compl_tok:,}",
    ]
    if running:
        lines.append("slowest running:")
        for el, t, att, turn in sorted(running, key=lambda r: -r[0])[:10]:
            lines.append(f"  {el / 60:6.1f} min  {t['type']}/{t['task']}  attempt={att} turn={turn}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    args = ap.parse_args()
    name = args.run_name or (load_config(args.config)["run_name"] if args.config else None)
    if not name:
        raise SystemExit("pass --config or --run-name")
    run_dir = args.runs_dir / name
    if not run_dir.exists():
        raise SystemExit(f"no run directory {run_dir}")
    while True:
        out = snapshot(run_dir)
        if args.watch:
            print("\033[2J\033[H" + time.strftime("%H:%M:%S") + "\n" + out, flush=True)
            time.sleep(args.watch)
        else:
            print(out)
            return


if __name__ == "__main__":
    main()
