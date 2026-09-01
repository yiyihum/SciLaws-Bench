#!/usr/bin/env python3
"""Regenerate the `tasks` array in `docs/assets/data.js` from `dataset/task_index.csv`.

The homepage task explorer filters and sorts on these records, so they carry more than
the id/domain/target the page used to ship. Everything else in `data.js` (stats, domains,
leaderboard) is left byte-identical.

    python scripts/build_task_table.py            # rewrite in place
    python scripts/build_task_table.py --check    # exit 1 if the file is out of date
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "dataset" / "task_index.csv"
DATA_JS = ROOT / "docs" / "assets" / "data.js"

# `"tasks": [ ... ]` up to the closing bracket that sits at the same indent
TASKS_BLOCK = re.compile(r'( "tasks":\s*)\[.*?\n \]', re.DOTALL)


def subfield(domain: str) -> str:
    """`astronomy / pulsar_timing` -> `pulsar timing`; keep the part after the slash."""
    tail = domain.split("/")[-1].strip()
    return tail.replace("_", " ")


def build() -> list[dict]:
    out = []
    for r in csv.DictReader(INDEX.open(encoding="utf-8")):
        tid = r["task_id"]
        out.append({
            "id": tid,
            "domain": r["discipline"],
            "sub": subfield(r["domain"]),
            "group": r["group_structure"],
            "name": tid.split("__")[0].replace("_", " "),
            "target": r["target"],
            "unit": r["unit"].strip(),
            "ni": int(r["n_inputs"]),
            "tr": int(r["rows_train"]),
            "te": int(r["rows_test"]),
        })
    out.sort(key=lambda t: t["id"])
    return out


def render(tasks: list[dict]) -> str:
    body = ",\n".join("  " + json.dumps(t, ensure_ascii=False) for t in tasks)
    return f"[\n{body}\n ]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args()

    src = DATA_JS.read_text(encoding="utf-8")
    if not TASKS_BLOCK.search(src):
        sys.exit(f"no `tasks` array found in {DATA_JS}")
    new = TASKS_BLOCK.sub(lambda m: m.group(1) + render(build()), src, count=1)

    if new == src:
        print(f"up to date ({len(build())} tasks)")
        return 0
    if args.check:
        print(f"{DATA_JS} is out of date; run {Path(__file__).name}")
        return 1
    DATA_JS.write_text(new, encoding="utf-8")
    print(f"wrote {len(build())} tasks to {DATA_JS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
