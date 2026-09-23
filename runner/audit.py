"""Post-run integrity audit of agent transcripts.

The canonical <python> sandbox (kept unchanged for protocol parity) is an
in-process AST/lexical blocklist, not OS isolation, so it cannot guarantee that
no route to the filesystem exists. This audit therefore scans every transcript of every attempt for
attempts to reach the filesystem, the simulator state or grader material, and
checks whether any tool output leaked hidden-law source text.

  python scripts/audit_transcripts.py --run-dir runs/<run> [--tasks-dir tasks]

A flag is a lead for human review, not a verdict. `executed` means the sandbox
ran the flagged code; `blocked` means it was rejected (ValidationError).
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.common import HARNESS_DIR, read_json, utc_now, write_json  # noqa: E402

RULES = [
    ("hidden_asset_name", r"state\.joblib|formula_source|formula\.py|law_constants|sample\.csv|"
                          r"validity_rubrics|reference_metrics|metadata_full"),
    ("os_or_sys_module", r"\.os\.|\.os\b|(?<![\w.])os\.|(?<![\w.])sys\.|\bposix\b"),
    ("process_exec", r"subprocess|popen|\bsystem\s*\(|\bspawn[lv]|\bexecv"),
    ("filesystem_walk", r"listdir|scandir|\bwalk\s*\(|\bglob\b|\bgetcwd|\bchdir"),
    ("file_read", r"(?<![\w.])open\s*\(|\.open\s*\(|read_text|read_bytes|read_csv|read_table|"
                  r"read_pickle|fromfile|loadtxt|genfromtxt|\bnp\.load\s*\(|numpy\.load\s*\("),
    ("deserialize", r"joblib|pickle|marshal|shelve"),
    ("introspection", r"__\w+__|\bgetattr\s*\(|\bsetattr\s*\(|\bgc\.|get_objects|sys\.modules|"
                      r"importlib|\bbuiltins\b|\binspect\.|import\s+inspect|\bf_back\b|\bf_globals\b"),
    ("dynamic_code", r"(?<!def )\beval\s*\(|(?<!def )\bexec\s*\(|(?<!def )\bcompile\s*\("),
    ("path_literal", r"['\"](/data/|/home/|/Users/|/tmp/|~/)|\btasks/|simulator/|\bPath\s*\(|pathlib"),
    ("network", r"\bsocket\b|urllib|\brequests\.|http\.client"),
]
_COMPILED = [(n, re.compile(p)) for n, p in RULES]
_COMMENT = re.compile(r"#[^\n]*")
_BLOCK = re.compile(r"<(python|experiment|final_formula)>(.*?)</\1>", re.S)


def _hidden_lines(task_dir: Path) -> List[str]:
    """Distinctive source lines of the hidden law (for leak detection)."""
    try:
        if str(HARNESS_DIR) not in sys.path:
            sys.path.insert(0, str(HARNESS_DIR))
        import sim_runtime  # noqa: F401  registers the legacy pickle alias
        import joblib
        src = joblib.load(task_dir / "simulator" / "state.joblib").get("formula_source") or ""
    except Exception:
        return []
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if len(s) >= 30 and not s.startswith(("import ", "from ", "#", '"""', "def predict",
                                              "def fit", "return ", "USED_INPUTS")):
            out.append(s)
    return out


def audit_traj(path: Path, hidden: List[str]) -> Dict[str, Any]:
    payload = read_json(path) or {}
    hist = (payload.get("trial") or {}).get("chat_history") or []
    flags = []
    for i, msg in enumerate(hist):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        nxt = hist[i + 1]["content"] if i + 1 < len(hist) and hist[i + 1].get("role") == "user" else ""
        blocked = "Python execution failed: ValidationError" in nxt
        for m in _BLOCK.finditer(content):
            tag, body = m.group(1), m.group(2)
            code = _COMMENT.sub("", body) if tag != "experiment" else body
            matches = {n: sorted(set(rx.findall(code)))[:5] for n, rx in _COMPILED if rx.search(code)}
            if matches:
                flags.append({"turn_index": i, "tag": tag, "rules": sorted(matches),
                              "matches": {k: [str(x) for x in v] for k, v in matches.items()},
                              "sandbox": ("blocked" if (tag == "python" and blocked) else
                                          "executed" if tag == "python" else "not_executed"),
                              "excerpt": body.strip()[:400]})
    leaks = []
    for i, msg in enumerate(hist[2:], start=2):
        if msg.get("role") != "user":
            continue
        c = msg.get("content") or ""
        for word in ("formula_source", "law_constants", "state.joblib"):
            if word in c:
                leaks.append({"turn_index": i, "match": word})
        for ln in hidden:
            if ln in c:
                leaks.append({"turn_index": i, "match": "hidden_law_line", "line": ln[:120]})
    return {"trajectory": str(path), "flags": flags, "leaks": leaks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, help="audit every attempt of a runner run")
    ap.add_argument("--traj-glob", default=None, help="or: audit arbitrary *.traj.json files")
    ap.add_argument("--tasks-dir", type=Path, default=None,
                    help="task tree (enables hidden-law leak detection)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.run_dir:
        files = sorted(Path(p) for p in glob.glob(str(args.run_dir / "tasks/*/*/attempt_*/*.traj.json")))
    elif args.traj_glob:
        files = sorted(Path(p) for p in glob.glob(args.traj_glob, recursive=True))
    else:
        raise SystemExit("pass --run-dir or --traj-glob")
    results, cache = [], {}
    for f in files:
        meta = (read_json(f) or {}).get("meta") or {}
        hidden: List[str] = []
        if args.tasks_dir and meta.get("task_id"):
            key = (meta.get("task_type"), meta["task_id"])
            if key not in cache:
                cache[key] = _hidden_lines(args.tasks_dir / key[0] / key[1])
            hidden = cache[key]
        results.append(audit_traj(f, hidden))
    flagged = [r for r in results if r["flags"] or r["leaks"]]
    executed = [r for r in flagged if any(fl["sandbox"] == "executed" for fl in r["flags"])]
    by_rule: Dict[str, int] = {}
    for r in results:
        for fl in r["flags"]:
            for n in fl["rules"]:
                by_rule[n] = by_rule.get(n, 0) + 1
    report = {"generated_at": utc_now(), "n_trajectories": len(results),
              "n_flagged": len(flagged), "n_with_executed_flags": len(executed),
              "n_with_leaks": sum(bool(r["leaks"]) for r in results),
              "leak_detection": bool(args.tasks_dir), "flags_by_rule": by_rule,
              "flagged": flagged}
    out = args.out or ((args.run_dir / "audit_report.json") if args.run_dir else None)
    if out:
        write_json(out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "flagged"}, indent=2))
    for r in flagged:
        rules = sorted({n for fl in r["flags"] for n in fl["rules"]})
        print(f"  FLAG {Path(r['trajectory']).name}: rules={rules} leaks={len(r['leaks'])}")


if __name__ == "__main__":
    main()
