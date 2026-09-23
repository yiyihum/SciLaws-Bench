#!/usr/bin/env python3
"""MAINTAINER ONLY: (re)build manifests/ from a full task tree and the Table-2
GPT-5.5 Parallel trajectories. Teammates never run this.

  python scripts/build_manifests.py --source-tasks <full tasks tree> \
      --golden-trajectories <Table-2 .../parallel/trajectories/gpt-5.5>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import (  # noqa: E402
    CANONICAL_COMMIT, HF_DATASET, HF_REVISION, MANIFEST_DIR, REPO, is_task_file, sha256_file,
    sha256_text, write_json,
)

# Canonical files the runner executes (must stay byte-identical to upstream).
RUNTIME_FILES = [
    "harness/agent_protocol.py", "harness/prompts.py", "harness/sim_runtime.py",
    "baseline_agent/agent.py", "baseline_agent/task.py", "baseline_agent/run_baseline.py",
    "baseline_agent/call_llm_api.py",
]
# Grader-side files kept in the repo for our own scoring; hashed for provenance.
GRADER_FILES = ["harness/evaluate_parallel.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tasks", required=True, type=Path)
    ap.add_argument("--golden-trajectories", required=True, type=Path)
    args = ap.parse_args()

    tasks, files, excluded = [], {}, set()
    for tt in ("typeI", "typeII"):
        for td in sorted(p for p in (args.source_tasks / tt).iterdir() if p.is_dir()):
            tasks.append({"task": td.name, "type": tt})
            entry = {}
            for f in sorted(p for p in td.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
                rel = f.relative_to(td).as_posix()
                if is_task_file(rel):
                    entry[rel] = sha256_file(f)
                else:
                    excluded.add(rel if "/" not in rel else rel.split("/")[0] + "/" + (
                        "*" if rel.startswith("eval/") else rel.split("/", 1)[1]))
            if "simulator/state.joblib" not in entry:
                raise SystemExit(f"{td}: no simulator/state.joblib")
            files[f"{tt}/{td.name}"] = entry
    n1 = sum(t["type"] == "typeI" for t in tasks)
    if (n1, len(tasks) - n1) != (66, 52):
        raise SystemExit(f"expected 66/52 tasks, got {n1}/{len(tasks) - n1}")
    write_json(MANIFEST_DIR / "tasks_118.json", {
        "benchmark": "SCILAWS-PARALLEL (SciLaws-Bench, arXiv:2609.01552)",
        "n_tasks": len(tasks), "n_typeI_single": n1, "n_typeII_multi": len(tasks) - n1,
        "tasks": tasks})
    write_json(MANIFEST_DIR / "task_hashes.json", {
        "hf_dataset": HF_DATASET, "hf_revision": HF_REVISION,
        "shipped": ["metadata.yaml", "data/*.csv", "simulator/state.joblib"],
        "excluded_grader_files": sorted(excluded),
        "files": files})

    def blob_sha(path: str) -> str:
        txt = subprocess.run(["git", "-C", str(REPO), "show", f"{CANONICAL_COMMIT}:{path}"],
                             capture_output=True, check=True).stdout
        import hashlib
        return hashlib.sha256(txt).hexdigest()
    write_json(MANIFEST_DIR / "canonical_sources.json", {
        "upstream_repo": "https://github.com/yiyihum/SciLaws-Bench",
        "upstream_commit": CANONICAL_COMMIT,
        "runtime_files": {p: blob_sha(p) for p in RUNTIME_FILES},
        "grader_files": {p: blob_sha(p) for p in GRADER_FILES}})

    golden = {}
    for t in tasks:
        p = args.golden_trajectories / t["type"] / f"{t['task']}.traj.json"
        hist = json.loads(p.read_text())["trial"]["chat_history"]
        golden[t["task"]] = {"type": t["type"], "system_sha256": sha256_text(hist[0]["content"]),
                             "user_sha256": sha256_text(hist[1]["content"])}
    write_json(MANIFEST_DIR / "golden_prompts.json", {
        "source": "Table-2 GPT-5.5 SCILAWS-PARALLEL run (2026-06-29), messages[0:2] of each "
                  "trajectory; max_turns=30",
        "prompts": golden})
    print(f"manifests written: {len(tasks)} tasks ({n1} typeI / {len(tasks) - n1} typeII); "
          f"excluded grader files: {sorted(excluded)}")


if __name__ == "__main__":
    main()
