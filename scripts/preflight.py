#!/usr/bin/env python3
"""Zero-API preflight for the SCILAWS-PARALLEL external runner.

  python scripts/preflight.py --config configs/gpt56_luna.yaml [--tasks-dir tasks]

Makes NO paid API call. Checks, in order:
  git        runner checkout is clean and descends from the canonical commit
  deps       Python + pinned package versions
  canonical  harness/ + baseline_agent/ runtime files byte-identical to upstream
  tasks      exactly 66 typeI + 52 typeII, every file hash matches, no grader files
  simulators every simulator/state.joblib loads
  prompts    regenerated system+user prompts == Table-2 golden hashes, 118/118
  config     config matches the frozen handoff (manifests/handoff.json)
  request    the request the canonical client WOULD send (captured, not sent):
             exact model string, reasoning_effort, max_completion_tokens, no
             sampling overrides, 120 s timeout, 1 retry
  launcher   dry-run plans exactly 118 tasks
  selftest   scripted fake-LLM run: submit / retry / no-retry / cap / interrupt+
             resume / skip-on-relaunch / packaging (zero API)
  package    packaging dry-run lists all 118 task ids
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import (  # noqa: E402
    BASELINE_DIR, CANONICAL_COMMIT, DEFAULT_TASKS_DIR, HARNESS_DIR, MANIFEST_DIR, REPO, child_env,
    git_info, load_config, load_manifest, package_versions, read_json, sha256_file, sha256_text,
    task_list,
)

PINS = {"numpy": "2.4.4", "scipy": "1.17.1", "pandas": "3.0.2", "scikit-learn": "1.8.0",
        "joblib": "1.5.3", "PyYAML": "6.0.3", "openai": "2.34.0"}
RESULTS: list = []


def check(name: str, ok: bool, detail: str = "", warn: bool = False) -> bool:
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((name, tag, detail))
    print(f"[{tag}] {name}: {detail}", flush=True)
    return ok


def _fetch_mod():
    spec = importlib.util.spec_from_file_location("fetch_tasks", REPO / "scripts" / "fetch_tasks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_selftest(tasks_dir: Path) -> None:
    grid = task_list()
    t1 = [t["task"] for t in grid if t["type"] == "typeI"]
    t2 = [t["task"] for t in grid if t["type"] == "typeII"]
    plan = {t1[0]: ("ok", "submitted", 1), t2[0]: ("crash_first", "submitted", 2),
            t1[1]: ("no_submit", "no_submission", 1), t2[1]: ("rate_limit_first", "submitted", 2),
            t1[2]: ("crash_always", "infra_failed", 3)}
    interrupt_task = t2[2]
    cfg = REPO / "configs" / "selftest_fake.yaml"
    runs = Path(tempfile.mkdtemp(prefix="scilaws_selftest_"))
    base = [sys.executable, "-m", "runner.launch", "--config", str(cfg), "--tasks-dir",
            str(tasks_dir), "--runs-dir", str(runs), "--jobs", "5"]
    env = dict(os.environ, RUNNER_BACKOFF_SCALE="0.01",
               FAKE_LLM_BEHAVIOR=json.dumps({k: v[0] for k, v in plan.items()}))
    try:
        # 1) five scripted tasks
        p = subprocess.run(base + ["--run-name", "st", "--tasks", *plan], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=900)
        run = runs / "st"
        ok_all, details = p.returncode == 0, []
        for task, (mode, want, n_att) in plan.items():
            tt = "typeI" if task in t1 else "typeII"
            st = read_json(run / "tasks" / tt / task / "state.json") or {}
            got, n = st.get("status"), len(st.get("attempts", []))
            has_sub = (run / "submissions" / tt / f"{task}.py").exists()
            has_rec = (run / "records" / tt / f"{task}.json").exists()
            good = got == want and n == n_att and has_sub == (want == "submitted") and has_rec
            ok_all &= good
            details.append(f"{mode}->{got}/{n}att{'' if good else ' (WANT ' + want + f'/{n_att})'}")
        check("selftest: submit / infra-retry / no-retry / attempt cap", ok_all,
              "; ".join(details) + ("" if p.returncode == 0 else f"; rc={p.returncode} {p.stderr[-300:]}"))
        # 2) relaunch the same command -> nothing re-runs, attempts unchanged
        before = sorted(str(x) for x in run.rglob("attempt_*"))
        p2 = subprocess.run(base + ["--run-name", "st", "--tasks", *plan], cwd=REPO, env=env,
                            capture_output=True, text=True, timeout=300)
        after = sorted(str(x) for x in run.rglob("attempt_*"))
        check("selftest: relaunch skips terminal tasks", p2.returncode == 0 and before == after
              and "to_run=0" in p2.stdout, re.search(r"terminal=\d+\s+to_run=\d+", p2.stdout).group(0)
              if re.search(r"terminal=\d+\s+to_run=\d+", p2.stdout) else p2.stdout[-200:])
        # 3) kill the launcher mid-attempt, then resume -> cold restart, attempt 1 archived
        env_h = dict(env, FAKE_LLM_BEHAVIOR=json.dumps({interrupt_task: "hang"}))
        cmd_i = base + ["--run-name", "st_int", "--tasks", interrupt_task]
        proc = subprocess.Popen(cmd_i, cwd=REPO, env=env_h, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        st_path = runs / "st_int" / "tasks" / "typeII" / interrupt_task / "state.json"
        for _ in range(120):
            if (read_json(st_path) or {}).get("status") == "running":
                break
            time.sleep(0.5)
        time.sleep(3)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=60)
        env_ok = dict(env, FAKE_LLM_BEHAVIOR=json.dumps({interrupt_task: "ok"}))
        p3 = subprocess.run(cmd_i, cwd=REPO, env=env_ok, capture_output=True, text=True, timeout=300)
        st = read_json(st_path) or {}
        outs = [(a.get("outcome"), a.get("counts_toward_cap")) for a in st.get("attempts", [])]
        check("selftest: interrupt + resume = cold restart, old attempt archived",
              p3.returncode == 0 and st.get("status") == "submitted"
              and outs == [("interrupted", False), ("submitted", True)]
              and (runs / "st_int" / "tasks" / "typeII" / interrupt_task / "attempt_01").is_dir(),
              f"attempts={outs} status={st.get('status')}")
        # 4) package the scripted run
        p4 = subprocess.run([sys.executable, "-m", "runner.package", "--run-name", "st",
                             "--runs-dir", str(runs), "--tasks-dir", str(tasks_dir),
                             "--out-dir", str(runs / "dist")], cwd=REPO, capture_output=True,
                            text=True, timeout=600)
        man = read_json(run / "MANIFEST.json") or {}
        bundles = list((runs / "dist").glob("*.tar.gz"))
        check("selftest: package bundle + manifest + audit", p4.returncode == 0 and len(bundles) == 1
              and man.get("n_tasks") == len(plan) and (run / "audit_report.json").exists()
              and (run / "SHA256SUMS").exists(),
              f"manifest n_tasks={man.get('n_tasks')} status={man.get('status_counts')}")
    finally:
        shutil.rmtree(runs, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "gpt56_luna.yaml")
    ap.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-version-drift", action="store_true")
    ap.add_argument("--allow-non-handoff-model", action="store_true")
    args = ap.parse_args()
    tasks_dir = args.tasks_dir.resolve()
    cfg = load_config(args.config)
    handoff = load_manifest("handoff.json")
    t0 = time.time()

    # git
    gi = git_info()
    check("git: canonical commit is an ancestor", gi["canonical_is_ancestor"],
          f"branch={gi['runner_branch']} commit={gi['runner_commit']} canonical={CANONICAL_COMMIT[:7]}")
    check("git: working tree clean (tracked files)", not gi["runner_dirty"],
          "clean" if not gi["runner_dirty"] else "uncommitted changes to tracked files",
          warn=args.allow_dirty)

    # deps
    vers = package_versions()
    py_ok = sys.version_info[:2] >= (3, 11)
    check("deps: python >= 3.11", py_ok, vers["python"])
    drift = {k: f"{vers.get(k)}!={v}" for k, v in PINS.items() if vers.get(k) != v}
    check("deps: pinned versions", not drift, "all pinned" if not drift else str(drift),
          warn=args.allow_version_drift)

    # canonical sources
    cs = load_manifest("canonical_sources.json")
    bad = [p for p, sha in {**cs["runtime_files"], **cs["grader_files"]}.items()
           if sha256_file(REPO / p) != sha]
    check("canonical: runtime files byte-identical to upstream", not bad,
          f"{len(cs['runtime_files'])} runtime files @ {cs['upstream_commit'][:7]}"
          + (f"; MODIFIED: {bad}" if bad else ""))

    # tasks
    files = load_manifest("task_hashes.json")["files"]
    grid = task_list()
    n1 = sum(t["type"] == "typeI" for t in grid)
    check("tasks: grid is 66 typeI + 52 typeII = 118",
          (n1, len(grid) - n1) == (66, 52) and len(files) == 118, f"{n1}+{len(grid) - n1}")
    problems = _fetch_mod().verify(tasks_dir, files) if tasks_dir.is_dir() else ["tasks dir missing"]
    check("tasks: all hashes match, no grader/GT files, no extras", not problems,
          f"{tasks_dir}" + (f"; {len(problems)} problems e.g. {problems[:3]}" if problems else ""))
    forbidden = [p for p in tasks_dir.rglob("*") if p.name in ("formula.py", "sample.csv")
                 or p.parent.name == "eval"] if tasks_dir.is_dir() else []
    check("tasks: no formula.py / sample.csv / eval/ in solver task tree", not forbidden,
          f"{len(forbidden)} found")
    if problems:
        print("\ncannot continue without a verified task tree (run scripts/fetch_tasks.py)")
        return 1

    # simulators + prompts (in-process, canonical code)
    for p in (str(BASELINE_DIR), str(HARNESS_DIR)):
        sys.path.insert(0, p)
    import sim_runtime
    from prompts import load_system_prompt
    from task import load_task
    golden = load_manifest("golden_prompts.json")["prompts"]
    sim_bad, mism = [], []
    for t in grid:
        td = tasks_dir / t["type"] / t["task"]
        try:
            sim = sim_runtime.load(td / "simulator")
            assert hasattr(sim, "fetch_data")
        except Exception as e:  # noqa: BLE001
            sim_bad.append(f"{t['task']}: {type(e).__name__}: {e}")
            continue
        task = load_task(td, simulator="simulator")
        sys_p = load_system_prompt(is_simulator=True, has_group_id=task.has_group_id)
        usr_p = task.get_task_prompt(max_turns=int(cfg["max_turns"]))
        g = golden[t["task"]]
        if (sha256_text(sys_p), sha256_text(usr_p)) != (g["system_sha256"], g["user_sha256"]):
            mism.append(t["task"])
    check("simulators: every state.joblib loads", not sim_bad,
          f"{len(grid) - len(sim_bad)}/{len(grid)}" + (f"; {sim_bad[:3]}" if sim_bad else ""))
    check("prompts: regenerated == Table-2 golden hashes", not mism and not sim_bad,
          f"{len(grid) - len(mism) - len(sim_bad)}/{len(grid)} match"
          + (f"; mismatches {mism[:5]}" if mism else ""))

    # config vs frozen handoff
    diffs = {k: (cfg.get(k), handoff[k]) for k in
             ("model_id", "provider", "base_url", "reasoning_effort", "max_turns",
              "llm_timeout_seconds", "max_attempts") if cfg.get(k) != handoff[k]}
    check("config: matches frozen handoff (gpt-5.6-luna / medium / 30 turns)", not diffs,
          f"{args.config.name}" + (f"; DIFF {diffs}" if diffs else ""),
          warn=args.allow_non_handoff_model)
    key_set = bool(os.environ.get(cfg["api_key_env"]))
    check(f"config: {cfg['api_key_env']} is set", key_set,
          "set" if key_set else "export it before launching (never put it in the config)")

    # request wiring probe (same process shape as a real task attempt; no network)
    if key_set:
        pr = subprocess.run([sys.executable, "-m", "runner.run_task", "--config", str(args.config),
                             "--probe-request"], cwd=REPO, env=child_env(cfg),
                            capture_output=True, text=True, timeout=120)
        try:
            cap = json.loads(pr.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            cap = {}
        req, cli = cap.get("request") or {}, cap.get("client") or {}
        exp = handoff
        checks = {
            "model": req.get("model") == cfg["model_id"] == exp["model_id"],
            "resolved": cap.get("resolved") == ["oa", cfg["model_id"]],
            "reasoning_branch": cap.get("is_reasoning") is True,
            "reasoning_effort": req.get("reasoning_effort") == exp["reasoning_effort"],
            "max_completion_tokens": req.get("max_completion_tokens") == exp["max_completion_tokens"],
            "no_sampling_overrides": not any(k in req for k in exp["forbidden_request_keys"]),
            "timeout": float(cli.get("timeout") or 0) == float(exp["llm_timeout_seconds"]),
            "max_retries": cli.get("max_retries") == exp["client_max_retries"],
            "endpoint": (cli.get("base_url_effective") or "").rstrip("/") == exp["base_url"].rstrip("/"),
        }
        shown = {k: req.get(k) for k in ("model", "reasoning_effort", "max_completion_tokens")}
        check("request: canonical client would send the frozen request", all(checks.values()),
              f"{shown} timeout={cli.get('timeout')} retries={cli.get('max_retries')} "
              f"endpoint={cli.get('base_url_effective')} keys={sorted(req)}"
              + ("" if all(checks.values()) else f"; FAILED {[k for k, v in checks.items() if not v]}"
                 + f"; stderr={pr.stderr[-400:]}"))
    else:
        check("request: canonical client would send the frozen request", False,
              "skipped: API key env var not set")

    # launcher dry-run on a throwaway runs dir
    with tempfile.TemporaryDirectory() as tmp:
        pl = subprocess.run([sys.executable, "-m", "runner.launch", "--config", str(args.config),
                             "--tasks-dir", str(tasks_dir), "--runs-dir", tmp, "--dry-run"],
                            cwd=REPO, capture_output=True, text=True, timeout=120)
    n_would = pl.stdout.count("would run ")
    check("launcher: dry-run plans the full grid", pl.returncode == 0 and n_would == 118
          and "tasks=118 (typeI=66 typeII=52)" in pl.stdout, f"{n_would} tasks planned")

    # zero-API scripted self-test of launch / retry / resume / package
    if args.skip_selftest:
        check("selftest", True, "skipped (--skip-selftest)", warn=True)
    else:
        run_selftest(tasks_dir)

    pk = subprocess.run([sys.executable, "-m", "runner.package", "--config", str(args.config),
                         "--dry-run", "--runs-dir", tempfile.gettempdir() + "/scilaws_nonexistent"],
                        cwd=REPO, capture_output=True, text=True, timeout=120)
    check("package: dry-run lists all 118 task ids", pk.returncode == 0
          and "manifest would list 118 tasks (typeI=66, typeII=52)" in pk.stdout, pk.stdout.strip()[-160:])

    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    n_warn = sum(r[1] == "WARN" for r in RESULTS)
    print(f"\npreflight: {len(RESULTS) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail "
          f"({time.time() - t0:.0f}s)")
    report = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": str(args.config),
              "git": gi, "versions": vers, "results": RESULTS}
    (REPO / "runs").mkdir(exist_ok=True)
    (REPO / "runs" / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("READY" if n_fail == 0 else "NOT READY")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
