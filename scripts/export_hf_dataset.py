#!/usr/bin/env python3
"""Stage the 118 SciLaws-Bench tasks for release as a Hugging Face dataset.

Three views are supported:

  eval        what the harness reads -- metadata.yaml, data/,
              eval/{reference_metrics,validity_rubrics}.json and
              simulator/{state.joblib,sample.csv,formula.py} -- plus a sanitised
              eval/metadata_full.yaml, which carries the `references` block
              (which published law each baseline id comes from). Sanitising
              keeps the YAML values and drops curation comments and paths to
              files outside the release. Drops __pycache__. This is the
              release view.

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

    python scripts/export_hf_dataset.py --view eval --out build/hf_eval
    python scripts/export_hf_dataset.py --view eval --out build/hf_eval \
        --push --repo-id RealSR/SciLaws-Bench
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# Top-level entries under a task directory withheld from the real-only view.
REAL_ONLY_EXCLUDE = ("eval", "simulator")

# Exact per-task files the harness reads (harness/*.py, baseline_agent/*.py).
EVAL_KEEP = {
    "metadata.yaml",
    "data/train.csv", "data/test.csv", "data/test_fit.csv", "data/test_test.csv",
    "eval/reference_metrics.json", "eval/validity_rubrics.json",
    "simulator/state.joblib", "simulator/sample.csv", "simulator/formula.py",
}

# Written from the curation tree rather than copied: the source file carries
# curation comments and per-reference paths into trees that are not released.
SANITISED = "eval/metadata_full.yaml"
# Per-reference keys that point at the curation tree, not at anything shipped.
DROP_REFERENCE_KEYS = ("formula_file", "reference_pdf")


def sanitise_metadata_full(src: Path) -> str:
    """Re-emit metadata_full.yaml from its parsed values only.

    Round-tripping through the YAML loader drops every comment, so curation
    notes never reach the release; the dangling per-reference paths are removed
    explicitly. The reference `id` is kept and still matches the keys in
    eval/reference_metrics.json.
    """
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    for ref in doc.get("references") or []:
        if isinstance(ref, dict):
            for key in DROP_REFERENCE_KEYS:
                ref.pop(key, None)
    header = ("# Grader-facing task record: the solver-facing fields of metadata.yaml\n"
              "# plus `references` (the published laws this task is anchored on),\n"
              "# `validity_rubrics` and `best_baseline`. Withhold from a system under\n"
              "# evaluation, like the rest of eval/.\n")
    return header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


def stage(tasks_root: Path, out: Path, view: str) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "tasks").mkdir(parents=True)

    counts = {"typeI": 0, "typeII": 0, "files": 0, "bytes": 0}
    # top-level task metadata (domains.json: the six-discipline grouping)
    for item in sorted(p for p in tasks_root.iterdir() if p.is_file()):
        shutil.copy2(item, out / "tasks" / item.name)
        counts["files"] += 1
        counts["bytes"] += item.stat().st_size
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
                if view == "eval" and str(rel) not in EVAL_KEEP and str(rel) != SANITISED:
                    continue
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if view == "eval" and str(rel) == SANITISED:
                    target.write_text(sanitise_metadata_full(item), encoding="utf-8")
                else:
                    shutil.copy2(item, target)
                counts["files"] += 1
                counts["bytes"] += target.stat().st_size
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
    ap.add_argument("--view", choices=("eval", "full", "real-only"), default="eval",
                    help="what to include (default: eval)")
    ap.add_argument("--push", action="store_true",
                    help="upload the staged directory to the Hub (requires --repo-id)")
    ap.add_argument("--repo-id", default=None, help="e.g. RealSR/SciLaws-Bench")
    ap.add_argument("--private", action="store_true", help="create the Hub repo as private")
    args = ap.parse_args()

    counts = stage(args.tasks_root.resolve(), args.out.resolve(), args.view)
    gb = counts["bytes"] / 1e9
    print(f"staged {counts['typeI']} typeI + {counts['typeII']} typeII tasks "
          f"({counts['files']} files, {gb:.2f} GB) -> {args.out}  [view={args.view}]")
    if args.view in ("eval", "full"):
        print("note: this view publishes eval/ rubrics, reference anchors and every "
              "Parallel hidden law.")
    if args.view == "full":
        print("warning: the full view copies eval/metadata_full.yaml verbatim, including "
              "its curation comments. Use --view eval for anything public.")

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
