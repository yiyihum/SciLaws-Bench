"""Launch or resume the 118-task SCILAWS-PARALLEL run for one model config.

  python -m runner.launch --config configs/gpt56_luna.yaml --jobs 40

Each task attempt runs the canonical `run_baseline.py --simulator` in its own
process (via runner/run_task.py). Re-running the same command resumes: tasks in
a terminal state are skipped, everything else is cold-restarted from turn 0.
Parallel tasks are never warm-resumed (the experiment observations live only
in the dead process).

Terminal statuses
  submitted       the agent produced a <final_formula> (valid or not -- judged later)
  no_submission   the agent used its whole turn budget without a parseable formula
  infra_failed    max_attempts attempts each ended in an infrastructure failure

Only infrastructure failures are retried (transport / HTTP 429 / 5xx / API timeout
/ process crash / task timeout), at most `max_attempts` counted attempts in total.
A model outcome -- any submission, or no submission after the turn budget -- is
final. Attempts killed by the operator (Ctrl-C) or by a fatal configuration error
(bad key, unknown model, exhausted quota) are archived but not counted. Every
attempt directory is kept; nothing is overwritten.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import (  # noqa: E402
    CANONICAL_COMMIT, DEFAULT_RUNS_DIR, DEFAULT_TASKS_DIR, HF_REVISION, MANIFEST_DIR, REPO,
    TERMINAL_STATUSES, append_jsonl, child_env, config_fingerprint, git_info, load_config,
    package_versions, read_json, sha256_file, task_list, utc_now, write_json,
)

FATAL_PATTERNS = [
    r"AuthenticationError", r"Error code: 401", r"Incorrect API key", r"invalid_api_key",
    r"is not set in the environment", r"Error code: 403", r"PermissionDenied",
    r"insufficient_quota", r"model_not_found", r"does not exist or you do not have access",
    r"^FATAL:", r"not in api_source_mapping",
]
INFRA_KINDS = [
    ("rate_limit", r"Error code: 429|RateLimitError|rate_limit|Too Many Requests"),
    ("api_timeout", r"APITimeoutError|Request timed out|ReadTimeout|timed out"),
    ("api_server", r"Error code: 5\d\d|InternalServerError|ServiceUnavailable|Bad Gateway|overloaded"),
    ("transport", r"APIConnectionError|Connection error|ConnectionError|RemoteDisconnected|"
                  r"Connection reset"),
    ("bad_request", r"Error code: 400|BadRequestError"),
]


class Campaign:
    def __init__(self, cfg: Dict[str, Any], cfg_path: Path, run_dir: Path, tasks_dir: Path,
                 jobs: int):
        self.cfg, self.cfg_path, self.run_dir = cfg, cfg_path, run_dir
        self.tasks_dir, self.jobs = tasks_dir, jobs
        self.abort = threading.Event()
        self.fatal_reason: Optional[str] = None
        self.procs: Dict[int, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.meta: Dict[str, Any] = {}

    # ---- paths ------------------------------------------------------------
    def task_root(self, job) -> Path:
        return self.run_dir / "tasks" / job["type"] / job["task"]

    def state_path(self, job) -> Path:
        return self.task_root(job) / "state.json"

    def load_state(self, job) -> Dict[str, Any]:
        st = read_json(self.state_path(job))
        if st is None:
            st = {"task": job["task"], "type": job["type"], "status": "pending", "attempts": []}
        return st

    def save_state(self, job, st) -> None:
        st["updated_at"] = utc_now()
        write_json(self.state_path(job), st)

    # ---- classification ---------------------------------------------------
    def classify(self, job, adir: Path, rc: Optional[int], timed_out: bool,
                 interrupted: bool) -> Dict[str, Any]:
        traj = read_json(adir / f"{job['task']}.traj.json") or {}
        trial = traj.get("trial") or {}
        sub = adir / f"{job['task']}.py"
        submitted = sub.exists() and sub.read_text(errors="replace").strip() != ""
        log = (adir / "attempt.log").read_text(errors="replace") if (adir / "attempt.log").exists() else ""
        tail = "\n".join(log.strip().splitlines()[-25:])
        out: Dict[str, Any] = {
            "returncode": rc, "traj_status": trial.get("status"),
            "rounds": trial.get("rounds"), "n_experiments": trial.get("n_experiments"),
            "n_python_calls": trial.get("n_python_calls"),
            "usage_total": {k: v for k, v in (trial.get("usage_total") or {}).items()
                            if k != "reasoning_content"},
            "n_rate_limit_msgs": len(re.findall(r"Error code: 429|RateLimitError", log)),
            "n_timeout_msgs": len(re.findall(r"APITimeoutError|Request timed out", log)),
        }
        if interrupted:
            out.update(outcome="interrupted", counts_toward_cap=False)
        elif timed_out:
            out.update(outcome="infra_error", infra_kind="task_timeout", counts_toward_cap=True)
        elif trial.get("status") in ("completed", "completed_forced_final") and submitted:
            out.update(outcome="submitted", counts_toward_cap=True,
                       submission_sha256=sha256_file(sub))
        elif trial.get("status") == "max_turns_reached":
            out.update(outcome="no_submission", counts_toward_cap=True)
        elif any(re.search(p, log, re.M) for p in FATAL_PATTERNS):
            out.update(outcome="infra_error", infra_kind="fatal_config", fatal=True,
                       counts_toward_cap=False)
        else:
            kind = next((k for k, p in INFRA_KINDS if re.search(p, log)), None)
            if kind is None:
                kind = "killed_by_signal" if (rc is not None and rc < 0) else "process_crash"
            out.update(outcome="infra_error", infra_kind=kind, counts_toward_cap=True)
        if out["outcome"] == "infra_error":
            out["error_excerpt"] = tail
        return out

    # ---- one attempt ------------------------------------------------------
    def run_attempt(self, job, st) -> Dict[str, Any]:
        n = len(st["attempts"]) + 1
        adir = self.task_root(job) / f"attempt_{n:02d}"
        while adir.exists():  # never reuse an attempt directory
            n += 1
            adir = self.task_root(job) / f"attempt_{n:02d}"
        adir.mkdir(parents=True)
        rec = {"attempt": n, "dir": str(adir.relative_to(self.run_dir)), "started_at": utc_now(),
               "started_unix": time.time(), "pid": None}
        st["attempts"].append(rec)
        st["status"] = "running"
        self.save_state(job, st)

        cmd = [sys.executable, "-m", "runner.run_task", "--config", str(self.cfg_path),
               "--task-dir", str(Path(job["task_dir"]).resolve()), "--out", str(adir)]
        timed_out = False
        rc: Optional[int] = None
        with (adir / "attempt.log").open("w") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    env=child_env(self.cfg, n), start_new_session=True)
            rec["pid"] = proc.pid
            self.save_state(job, st)
            with self.lock:
                self.procs[proc.pid] = proc
            deadline = time.time() + float(self.cfg["task_timeout_seconds"])
            try:
                while True:
                    try:
                        rc = proc.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        if time.time() > deadline:
                            timed_out = True
                            _kill_group(proc)
                            rc = proc.wait()
                            log.write(f"\nRUNNER: task timeout after "
                                      f"{self.cfg['task_timeout_seconds']}s; process group killed\n")
                            break
            finally:
                with self.lock:
                    self.procs.pop(proc.pid, None)
        interrupted = self.abort.is_set() and self.fatal_reason is None and not timed_out \
            and rc is not None and rc != 0
        rec.update(self.classify(job, adir, rc, timed_out, interrupted))
        rec["ended_at"] = utc_now()
        rec["elapsed_seconds"] = round(time.time() - rec.pop("started_unix"), 1)
        self.save_state(job, st)
        return rec

    # ---- one task, all attempts --------------------------------------------
    def run_task(self, job) -> None:
        st = self.load_state(job)
        if st["status"] in TERMINAL_STATUSES:
            return
        while not self.abort.is_set():
            counted = [a for a in st["attempts"] if a.get("counts_toward_cap")]
            if len(counted) >= int(self.cfg["max_attempts"]):
                st["status"] = "infra_failed"
                self.finalize(job, st)
                return
            if counted and counted[-1].get("outcome") == "infra_error":
                kind = counted[-1].get("infra_kind")
                delay = 60 * len(counted) if kind == "rate_limit" else 15 * len(counted)
                delay *= float(os.environ.get("RUNNER_BACKOFF_SCALE") or 1.0)  # self-test only
                if self.abort.wait(delay):
                    break
            rec = self.run_attempt(job, st)
            _progress(self, job, rec)
            if rec["outcome"] in ("submitted", "no_submission"):
                st["status"] = rec["outcome"]
                self.finalize(job, st)
                return
            if rec.get("fatal"):
                self.fatal_reason = f"{job['task']}: fatal configuration/auth error"
                self.abort.set()
                break
            if rec["outcome"] == "interrupted":
                break
        st["status"] = "pending"
        self.save_state(job, st)

    # ---- outputs ----------------------------------------------------------
    def finalize(self, job, st) -> None:
        final = st["attempts"][-1] if st["attempts"] else {}
        adir = self.run_dir / final["dir"] if final else None
        traj_dst = self.run_dir / "trajectories" / job["type"] / f"{job['task']}.traj.json"
        sub_dst = self.run_dir / "submissions" / job["type"] / f"{job['task']}.py"
        if adir and (adir / f"{job['task']}.traj.json").exists():
            traj_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(adir / f"{job['task']}.traj.json", traj_dst)
        if st["status"] == "submitted":
            sub_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(adir / f"{job['task']}.py", sub_dst)
        st["finalized_at"] = utc_now()
        self.save_state(job, st)
        write_json(self.run_dir / "records" / job["type"] / f"{job['task']}.json",
                   build_record(self, job, st))
        append_jsonl(self.run_dir / "events.jsonl",
                     {"at": utc_now(), "task": job["task"], "type": job["type"],
                      "status": st["status"], "attempts": len(st["attempts"])})

    def kill_all(self) -> None:
        with self.lock:
            procs = list(self.procs.values())
        for p in procs:
            _kill_group(p)


def build_record(c: Campaign, job, st) -> Dict[str, Any]:
    final = st["attempts"][-1] if st["attempts"] else {}
    return {
        "task_id": job["task"],
        "task_type": job["type"],
        "setting": "single" if job["type"] == "typeI" else "multi",
        "terminal_status": st["status"],
        "final_attempt": final.get("attempt"),
        "model_outcome": final.get("traj_status"),
        "infra_error": (final.get("infra_kind") if final.get("outcome") == "infra_error" else None),
        "n_attempts": len(st["attempts"]),
        "attempts": st["attempts"],
        "submission": (f"submissions/{job['type']}/{job['task']}.py"
                       if st["status"] == "submitted" else None),
        "submission_sha256": final.get("submission_sha256"),
        "trajectory": (f"trajectories/{job['type']}/{job['task']}.traj.json"
                       if final.get("traj_status") else None),
        "usage_total_final_attempt": final.get("usage_total"),
        "model": c.cfg["model_id"],
        "model_alias": c.cfg["model_alias"],
        "provider": c.cfg["provider"],
        "endpoint": c.cfg["base_url"],
        "reasoning_effort": c.cfg["reasoning_effort"],
        "max_completion_tokens": 65536,
        "max_turns": c.cfg["max_turns"],
        "runner_commit": c.meta.get("git", {}).get("runner_commit"),
        "runner_dirty": c.meta.get("git", {}).get("runner_dirty"),
        "canonical_upstream_commit": CANONICAL_COMMIT,
        "hf_dataset_revision": HF_REVISION,
        "task_file_sha256": c.meta.get("task_hashes", {}).get(f"{job['type']}/{job['task']}"),
        "finalized_at": st.get("finalized_at"),
    }


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


_done_lock = threading.Lock()


def _progress(c: Campaign, job, rec) -> None:
    with _done_lock:
        line = (f"[{utc_now()}] {job['type']}/{job['task']} attempt={rec['attempt']} "
                f"outcome={rec['outcome']}"
                + (f"/{rec.get('infra_kind')}" if rec.get("infra_kind") else "")
                + f" traj={rec.get('traj_status')} rounds={rec.get('rounds')} "
                f"exps={rec.get('n_experiments')} tok={rec.get('usage_total', {}).get('total_tokens')} "
                f"{rec['elapsed_seconds']}s")
        print(line, flush=True)
        with (c.run_dir / "launcher.log").open("a") as fh:
            fh.write(line + "\n")


def plan_jobs(tasks_dir: Path, subset: List[str]) -> List[Dict[str, str]]:
    grid = task_list()
    if subset:
        names = {t["task"] for t in grid}
        unknown = [s for s in subset if s not in names]
        if unknown:
            raise SystemExit(f"unknown task(s): {unknown}")
        grid = [t for t in grid if t["task"] in set(subset)]
    jobs = []
    for t in grid:
        td = tasks_dir / t["type"] / t["task"]
        jobs.append({**t, "task_dir": str(td)})
    return jobs


def verify_tasks(tasks_dir: Path, jobs) -> Dict[str, Dict[str, str]]:
    """Hash-check the task files a run will use; return {type/task: {file: sha}}."""
    manifest = read_json(MANIFEST_DIR / "task_hashes.json")["files"]
    out, bad = {}, []
    for j in jobs:
        key = f"{j['type']}/{j['task']}"
        want = manifest[key]
        root = tasks_dir / j["type"] / j["task"]
        for rel, sha in want.items():
            p = root / rel
            if not p.exists() or sha256_file(p) != sha:
                bad.append(f"{key}/{rel}")
        out[key] = want
    if bad:
        raise SystemExit(f"task files missing or hash mismatch ({len(bad)}), e.g. {bad[:5]}. "
                         f"Run scripts/fetch_tasks.py / scripts/preflight.py first.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=None,
                    help="concurrent tasks (default: $JOBS or config `jobs`)")
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--run-name", default=None, help="default: config run_name")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--tasks", nargs="+", default=[], help="optional subset (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--skip-verify", action="store_true", help="skip task hash check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    jobs_n = args.jobs or int(os.environ.get("JOBS") or 0) or int(cfg["jobs"])
    run_name = args.run_name or cfg["run_name"]
    run_dir = (args.runs_dir / run_name).resolve()
    tasks_dir = args.tasks_dir.resolve()
    jobs = plan_jobs(tasks_dir, args.tasks)

    def _status(j):
        st = read_json(run_dir / "tasks" / j["type"] / j["task"] / "state.json") or {}
        return st.get("status", "not_started")
    statuses = [_status(j) for j in jobs]
    todo = [j for j, s in zip(jobs, statuses) if s not in TERMINAL_STATUSES]
    n1 = sum(j["type"] == "typeI" for j in jobs)
    print(f"run_dir={run_dir}\nmodel={cfg['model_id']} effort={cfg['reasoning_effort']} "
          f"provider={cfg['provider']} endpoint={cfg['base_url']}\n"
          f"tasks={len(jobs)} (typeI={n1} typeII={len(jobs) - n1})  "
          f"terminal={len(jobs) - len(todo)}  to_run={len(todo)}  jobs={jobs_n}  "
          f"max_attempts={cfg['max_attempts']}  task_timeout={cfg['task_timeout_seconds']}s",
          flush=True)
    if args.dry_run:
        for j in todo:
            print(f"  would run {j['type']}/{j['task']}")
        return 0

    missing_dirs = [j["task_dir"] for j in jobs if not Path(j["task_dir"]).is_dir()]
    if missing_dirs:
        raise SystemExit(f"{len(missing_dirs)} task dirs missing under {tasks_dir}; "
                         f"run scripts/fetch_tasks.py")
    if cfg["provider"] != "fake" and not os.environ.get(cfg["api_key_env"]):
        raise SystemExit(f"{cfg['api_key_env']} is not set")

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = (run_dir / ".launch.lock").open("w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another launcher is already running on {run_dir}")

    # Freeze the config for this run directory; refuse silent changes.
    snap = run_dir / "config.yaml"
    meta_path = run_dir / "run_meta.json"
    meta = read_json(meta_path)
    if meta is None:
        shutil.copy2(args.config, snap)
        meta = {"run_name": run_name, "created_at": utc_now(),
                "config_fingerprint": config_fingerprint(cfg), "config": cfg,
                "task_subset": sorted(args.tasks) or None,
                "canonical_upstream": {"repo": "yiyihum/SciLaws-Bench", "commit": CANONICAL_COMMIT},
                "hf_dataset": {"repo": "RealSR/SciLaws-Bench", "revision": HF_REVISION},
                "manifests_sha256": {p.name: sha256_file(p) for p in sorted(MANIFEST_DIR.glob("*.json"))},
                "launches": []}
    else:
        if meta["config_fingerprint"] != config_fingerprint(cfg):
            raise SystemExit(f"config differs from the one this run was started with:\n"
                             f"  run:    {meta['config_fingerprint']}\n"
                             f"  config: {config_fingerprint(cfg)}\n"
                             f"use a different --run-name for a different config")
        if (meta.get("task_subset") or None) != (sorted(args.tasks) or None):
            raise SystemExit("task subset differs from the one this run was started with")
    gi = git_info()
    if gi["runner_dirty"]:
        print("WARNING: runner repo has uncommitted changes (recorded in run_meta)", flush=True)
    meta["git"] = gi
    meta["launches"].append({"at": utc_now(), "jobs": jobs_n, "git": gi,
                             "versions": package_versions(), "host": os.uname().nodename,
                             "to_run": len(todo)})
    meta["task_hashes"] = ({} if args.skip_verify else verify_tasks(tasks_dir, jobs))
    write_json(meta_path, meta)

    camp = Campaign(cfg, snap.resolve(), run_dir, tasks_dir, jobs_n)
    camp.meta = meta

    # Attempts left "running" by a killed launcher are archived as interrupted.
    for j in jobs:
        st = read_json(camp.state_path(j))
        if st and st.get("status") == "running":
            last = st["attempts"][-1]
            last.setdefault("outcome", "interrupted")
            last.setdefault("counts_toward_cap", False)
            last.setdefault("ended_at", None)
            last.pop("started_unix", None)
            st["status"] = "pending"
            camp.save_state(j, st)

    def _on_signal(signum, frame):  # noqa: ARG001
        print(f"\nsignal {signum}: stopping, killing running attempts ...", flush=True)
        camp.abort.set()
        camp.kill_all()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        futs = [ex.submit(camp.run_task, j) for j in todo]
        while any(not f.done() for f in futs):
            time.sleep(1)
        for f in futs:
            f.result()

    final = [_status(j) for j in jobs]
    counts = {s: final.count(s) for s in sorted(set(final))}
    print(f"\nDONE {sum(s in TERMINAL_STATUSES for s in final)}/{len(jobs)} terminal  {counts}",
          flush=True)
    if camp.fatal_reason:
        print(f"ABORTED: {camp.fatal_reason}. Fix the key/model/quota and re-run the same "
              f"command; the affected attempt is archived and not counted.", flush=True)
        return 3
    if camp.abort.is_set():
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
