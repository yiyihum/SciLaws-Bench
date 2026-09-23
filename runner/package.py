"""Bundle a run directory into one archive to send back.

  python -m runner.package --config configs/gpt56_luna.yaml [--dry-run] [--allow-incomplete]

Writes runs/<run>/MANIFEST.json (one entry for EVERY task of the run's grid,
including not-started / failed ones), runs the transcript audit, writes
SHA256SUMS, and tars the whole run directory (all attempts included) to
dist/<run>_<UTC>.tar.gz.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import (  # noqa: E402
    DEFAULT_RUNS_DIR, DEFAULT_TASKS_DIR, REPO, TERMINAL_STATUSES, git_info, load_config,
    read_json, sha256_file, task_list, utc_now, write_json,
)


def build_manifest(run_dir: Path) -> dict:
    meta = read_json(run_dir / "run_meta.json") or {}
    grid = task_list()
    if meta.get("task_subset"):
        grid = [t for t in grid if t["task"] in set(meta["task_subset"])]
    entries = []
    for t in grid:
        st = read_json(run_dir / "tasks" / t["type"] / t["task"] / "state.json") or {}
        sub = run_dir / "submissions" / t["type"] / f"{t['task']}.py"
        atts = st.get("attempts", [])
        last = atts[-1] if atts else {}
        entries.append({
            "task": t["task"], "type": t["type"],
            "setting": "single" if t["type"] == "typeI" else "multi",
            "status": st.get("status", "not_started"),
            "n_attempts": len(atts),
            "attempt_outcomes": [a.get("outcome") for a in atts],
            "final_traj_status": last.get("traj_status"),
            "final_infra_kind": last.get("infra_kind"),
            "submission": str(sub.relative_to(run_dir)) if sub.exists() else None,
            "submission_sha256": sha256_file(sub) if sub.exists() else None,
            "record": (f"records/{t['type']}/{t['task']}.json"
                       if (run_dir / "records" / t["type"] / f"{t['task']}.json").exists() else None),
        })
    counts: dict = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    cfg = meta.get("config") or {}
    return {
        "generated_at": utc_now(), "run_name": meta.get("run_name"),
        "model": cfg.get("model_id"), "provider": cfg.get("provider"),
        "endpoint": cfg.get("base_url"), "reasoning_effort": cfg.get("reasoning_effort"),
        "n_tasks": len(entries),
        "n_typeI": sum(e["type"] == "typeI" for e in entries),
        "n_typeII": sum(e["type"] == "typeII" for e in entries),
        "status_counts": counts,
        "all_terminal": all(e["status"] in TERMINAL_STATUSES for e in entries),
        "packager_git": git_info(),
        "judge_input": "evaluate_parallel.py --submissions <run>/submissions (typeI/, typeII/)",
        "tasks": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR,
                    help="used only for the audit's hidden-law leak check")
    ap.add_argument("--out-dir", type=Path, default=REPO / "dist")
    ap.add_argument("--dry-run", action="store_true", help="manifest summary only, no archive")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="package even if some tasks are not terminal")
    args = ap.parse_args()
    name = args.run_name or (load_config(args.config)["run_name"] if args.config else None)
    if not name:
        raise SystemExit("pass --config or --run-name")
    run_dir = args.runs_dir / name
    if args.dry_run and not run_dir.exists():
        # Packaging plan for a run that has not started: every task still listed.
        grid = task_list()
        print(f"[dry-run] {run_dir} not started; manifest would list {len(grid)} tasks "
              f"(typeI={sum(t['type'] == 'typeI' for t in grid)}, "
              f"typeII={sum(t['type'] == 'typeII' for t in grid)}), all status=not_started")
        return 0
    if not run_dir.exists():
        raise SystemExit(f"no run directory {run_dir}")

    man = build_manifest(run_dir)
    print(f"{run_dir}: {man['n_tasks']} tasks (typeI={man['n_typeI']} typeII={man['n_typeII']}) "
          f"status={man['status_counts']} all_terminal={man['all_terminal']}")
    if args.dry_run:
        return 0
    if not man["all_terminal"] and not args.allow_incomplete:
        raise SystemExit("some tasks are not terminal; finish/resume the run first, or pass "
                         "--allow-incomplete to send a partial bundle")
    write_json(run_dir / "MANIFEST.json", man)

    audit_cmd = [sys.executable, "-m", "runner.audit", "--run-dir", str(run_dir)]
    if (args.tasks_dir / "typeI").is_dir():
        audit_cmd += ["--tasks-dir", str(args.tasks_dir)]
    subprocess.run(audit_cmd, cwd=REPO, check=False)

    files = sorted(p for p in run_dir.rglob("*")
                   if p.is_file() and p.name not in (".launch.lock", "SHA256SUMS"))
    (run_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(run_dir)}\n" for p in files))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(run_dir, arcname=name,
                filter=lambda ti: None if ti.name.endswith(".launch.lock") else ti)
    print(f"\nbundle: {out}\nsha256: {sha256_file(out)}\nsize:   {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
