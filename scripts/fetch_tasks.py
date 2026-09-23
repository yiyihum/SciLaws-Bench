#!/usr/bin/env python3
"""Install the pinned SCILAWS-PARALLEL task assets (118 tasks) and verify them.

Only solver-side files are installed: metadata.yaml, data/*.csv and
simulator/state.joblib. Grader material (simulator/formula.py, sample.csv,
eval/) is never copied into the runner's task tree.

  python scripts/fetch_tasks.py                               # from Hugging Face, pinned revision
  python scripts/fetch_tasks.py --from-archive tasks.tar.gz   # prepacked archive we sent you
  python scripts/fetch_tasks.py --from-dir /path/to/tasks     # an existing task tree
  python scripts/fetch_tasks.py --verify-only                 # just check --out
  ... --out /elsewhere/tasks     (then pass --tasks-dir /elsewhere/tasks to preflight/launch)
  ... --pack tasks_scilaws_parallel_118.tar.gz   (write a stripped archive after verifying)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import (  # noqa: E402
    DEFAULT_TASKS_DIR, HF_DATASET, HF_REVISION, read_json, sha256_file,
)

MANIFEST = Path(__file__).resolve().parent.parent / "manifests" / "task_hashes.json"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/tasks/{path}"


def download_hf(dst_root: Path, files: dict, revision: str, workers: int = 8) -> None:
    """Fetch exactly the manifest's files from the pinned HF revision (stdlib only:
    no huggingface_hub/httpx, whose brotli decoding breaks in some environments)."""
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(f"{k}/{rel}", sha) for k, e in files.items() for rel, sha in e.items()]
    token = os.environ.get("HF_TOKEN")

    def one(item):
        rel, sha = item
        dst = dst_root / rel
        if dst.exists() and sha256_file(dst) == sha:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = HF_URL.format(repo=HF_DATASET, rev=revision, path=urllib.parse.quote(rel))
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Accept-Encoding": "identity",
                                                           **({"Authorization": f"Bearer {token}"}
                                                              if token else {})})
                tmp = dst.with_name(dst.name + ".part")
                with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as fh:
                    shutil.copyfileobj(r, fh, 1 << 20)
                if sha256_file(tmp) != sha:
                    raise IOError(f"sha256 mismatch for {rel}")
                os.replace(tmp, dst)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        raise SystemExit(f"download failed for {rel}: {last}")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(one, jobs):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} files", flush=True)


def _find_root(d: Path) -> Path:
    """Locate the directory that contains typeI/ and typeII/."""
    for cand in [d, *sorted(d.rglob("typeI"))]:
        root = cand if (cand / "typeI").is_dir() else cand.parent
        if (root / "typeI").is_dir() and (root / "typeII").is_dir():
            return root
    raise SystemExit(f"no typeI/ + typeII/ task tree found under {d}")


def install(src_root: Path, out: Path, files: dict) -> None:
    n = 0
    for key, entry in files.items():
        for rel in entry:
            s, d = src_root / key / rel, out / key / rel
            if not s.exists():
                raise SystemExit(f"source is missing {key}/{rel}")
            if d.exists() and sha256_file(d) == entry[rel]:
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            n += 1
    print(f"installed/updated {n} files into {out}")


def verify(out: Path, files: dict) -> list:
    problems = []
    expected = {f"{k}/{rel}" for k, e in files.items() for rel in e}
    for key, entry in files.items():
        for rel, sha in entry.items():
            p = out / key / rel
            if not p.exists():
                problems.append(f"missing {key}/{rel}")
            elif sha256_file(p) != sha:
                problems.append(f"hash mismatch {key}/{rel}")
    for tt in ("typeI", "typeII"):
        for p in (out / tt).rglob("*") if (out / tt).is_dir() else []:
            rel = p.relative_to(out).as_posix()
            if p.is_file() and rel not in expected:
                problems.append(f"unexpected file {rel}")
            if p.is_dir() and p.parent.parent == out and f"{tt}/{p.name}" not in files:
                problems.append(f"unexpected task dir {rel}")
    extra_top = [p.name for p in out.iterdir() if p.name not in ("typeI", "typeII")] if out.is_dir() else []
    problems += [f"unexpected top-level entry {x}" for x in extra_top]
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_TASKS_DIR)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--from-dir", type=Path)
    src.add_argument("--from-archive", type=Path)
    src.add_argument("--verify-only", action="store_true")
    ap.add_argument("--revision", default=HF_REVISION, help="HF dataset revision (pinned)")
    ap.add_argument("--pack", type=Path, default=None, help="write a stripped .tar.gz after verifying")
    args = ap.parse_args()
    files = read_json(MANIFEST)["files"]
    out = args.out.resolve()

    if not args.verify_only:
        with tempfile.TemporaryDirectory(prefix="scilaws_fetch_", dir=out.parent if out.parent.exists() else None) as tmp:
            if args.from_dir:
                src_root = _find_root(args.from_dir.resolve())
            elif args.from_archive:
                with tarfile.open(args.from_archive) as tar:
                    tar.extractall(tmp, filter="data")
                src_root = _find_root(Path(tmp))
            else:
                print(f"downloading {HF_DATASET}@{args.revision} (solver-side files only, "
                      f"~0.85 GB) ...", flush=True)
                download_hf(Path(tmp) / "tasks", files, args.revision)
                src_root = Path(tmp) / "tasks"
            out.mkdir(parents=True, exist_ok=True)
            install(src_root, out, files)

    problems = verify(out, files)
    if problems:
        print(f"FAIL: {len(problems)} problem(s) in {out}:")
        for p in problems[:30]:
            print("  " + p)
        sys.exit(1)
    n1 = sum(k.startswith("typeI/") for k in files)
    print(f"OK: {len(files)} tasks (typeI={n1}, typeII={len(files) - n1}) verified in {out}; "
          f"no grader files present")
    if args.pack:
        with tarfile.open(args.pack, "w:gz") as tar:
            for tt in ("typeI", "typeII"):
                tar.add(out / tt, arcname=f"tasks/{tt}")
        print(f"packed {args.pack} sha256={sha256_file(args.pack)}")


if __name__ == "__main__":
    main()
