#!/usr/bin/env python3
"""Stage the 118 SciLaws-Bench tasks for release as a Hugging Face dataset.

Two views are supported:

  full        everything -- metadata.yaml, data/, eval/ (reference anchors and the
              frozen validity rubrics) and simulator/ (state.joblib, sample.csv,
              formula.py). This is what reproduces the paper end to end, and it
              publishes every answer key the benchmark has.

  real-only   metadata.yaml and data/ only. Drops eval/ and the whole simulator/
              directory, i.e. ships SciLaws-Real without any answer key.

There is deliberately no "solver-safe Parallel" view. Every simulator/state.joblib
carries `formula_source` -- the hidden law's own source text -- because sim_runtime
execs it to generate y. Removing it does not hide the law, it breaks the simulator
(verified: fetch_where then fails). Distributing a runnable Parallel world locally
therefore means distributing its answer. Keeping Parallel hidden requires serving
the oracle remotely instead of shipping it.

Staging only copies files. Pushing is a separate, explicit step (--push).

    python scripts/export_hf_dataset.py --view full --out build/hf_full
    python scripts/export_hf_dataset.py --view full --out build/hf_full \
        --push --repo-id RealSR/SciLaws-Bench
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Top-level entries under a task directory withheld from the real-only view.
REAL_ONLY_EXCLUDE = ("eval", "simulator")


def stage(tasks_root: Path, out: Path, view: str) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "tasks").mkdir(parents=True)

    counts = {"typeI": 0, "typeII": 0, "files": 0, "bytes": 0}
    for kind in ("typeI", "typeII"):
        src_kind = tasks_root / kind
        if not src_kind.is_dir():
            sys.exit(f"missing {src_kind} -- point --tasks-root at a checkout that has the data")
        for task in sorted(p for p in src_kind.iterdir() if p.is_dir()):
            dst = out / "tasks" / kind / task.name
            dst.mkdir(parents=True)
            for item in sorted(task.rglob("*")):
                if item.is_dir() or "__pycache__" in item.parts:
                    continue
                rel = item.relative_to(task)
                if view == "real-only" and rel.parts[0] in REAL_ONLY_EXCLUDE:
                    continue
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
                counts["files"] += 1
                counts["bytes"] += item.stat().st_size
            counts[kind] += 1

    shutil.copy2(REPO / "dataset" / "README.md", out / "README.md")
    shutil.copy2(REPO / "dataset" / "LICENSES.md", out / "LICENSES.md")
    shutil.copy2(REPO / "dataset" / "task_index.csv", out / "task_index.csv")
    (out / "EXPORT.json").write_text(json.dumps({"view": view, **counts}, indent=2) + "\n")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-root", type=Path, default=REPO / "tasks",
                    help="directory containing typeI/ and typeII/ (default: <repo>/tasks)")
    ap.add_argument("--out", type=Path, default=REPO / "build" / "hf",
                    help="staging directory to write")
    ap.add_argument("--view", choices=("full", "real-only"), default="full",
                    help="what to include (default: full)")
    ap.add_argument("--push", action="store_true",
                    help="upload the staged directory to the Hub (requires --repo-id)")
    ap.add_argument("--repo-id", default=None, help="e.g. RealSR/SciLaws-Bench")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    args = ap.parse_args()

    counts = stage(args.tasks_root.resolve(), args.out.resolve(), args.view)
    gb = counts["bytes"] / 1e9
    print(f"staged {counts['typeI']} typeI + {counts['typeII']} typeII tasks "
          f"({counts['files']} files, {gb:.2f} GB) -> {args.out}  [view={args.view}]")
    if args.view == "full":
        print("note: this view publishes eval/ rubrics, reference anchors and every "
              "Parallel hidden law.")

    if not args.push:
        print("\nnot pushed. review the staging directory, then re-run with:\n"
              "  --push --repo-id RealSR/SciLaws-Bench")
        return 0

    if not args.repo_id:
        sys.exit("--push requires --repo-id")
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset",
                            folder_path=str(args.out))
    print(f"pushed to https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
