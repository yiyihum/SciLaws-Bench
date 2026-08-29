"""Fixed tool-call interface for RealSR v3 agents.

This is the FROZEN protocol any solver shares (the baseline agent and any
plugged-in evolving / search agent alike). It owns three things:

  1. The tool TAGS the model emits and how they are parsed:
       <python>...code...</python>                          inspect data / fit
       <experiment>{...}</experiment>                       probe a simulator (if any)
       <final_formula>...module text...</final_formula>     submit (ends the trial)
  2. The <python> SANDBOX: validate + exec with preloaded variables, capture stdout.
  3. The per-turn DISPATCH (`step`): pick the first emitted tag, run it, and return
     either the submission or the feedback string to append to the conversation.

Pair this with `prompts.load_system_prompt` / `prompts.build_task_prompt` (the
matching instruction text). Your agent only has to: call your LLM -> `step(...)`
-> append feedback / stop on submit. Nothing here depends on a particular LLM
client or task object, so it is reusable across agents.
"""
from __future__ import annotations

import ast
import builtins
import io
import json
import math
import re
import signal
import traceback
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

TOOL_TAGS = ("final_formula", "python", "experiment")
PYTHON_TIMEOUT_SECONDS = 100
PYTHON_STDOUT_MAX_CHARS = 16_000
EXPERIMENT_OUTPUT_MAX_CHARS = 16_000
BRUTE_FORCE_MAX_KNOWN_ITERATIONS = 200_000
BRUTE_FORCE_MAX_NESTED_LOOP_DEPTH = 4


# ---- tag parsers -----------------------------------------------------------

def _last_block(text: str, open_tag: str, close_tag: str) -> Optional[str]:
    start = text.rfind(open_tag)
    if start == -1:
        return None
    end = text.find(close_tag, start)
    if end == -1:
        return None
    return text[start + len(open_tag):end].strip()


def parse_experiment(text: str) -> Optional[Dict[str, Any]]:
    block = _last_block(text, "<experiment>", "</experiment>")
    if block is None:
        return None
    try:
        parsed = json.loads(block)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_final_formula(text: str) -> Tuple[bool, str]:
    """Return (ok, module_text) for the <final_formula>...</final_formula> block.
    `ok` is False unless the block exists and defines `predict`."""
    block = _last_block(text, "<final_formula>", "</final_formula>")
    if block is None or "def predict" not in block:
        return False, ""
    return True, block


def extract_python(text: str) -> Optional[str]:
    return _last_block(text, "<python>", "</python>")


def first_tool_tag(text: str) -> Optional[str]:
    """The tool tag that appears FIRST by source position (the model's intent
    when it emits several in one turn — e.g. explore, then submit)."""
    first_tag, first_pos = None, -1
    for t in TOOL_TAGS:
        p = text.find(f"<{t}>")
        if p >= 0 and (first_pos < 0 or p < first_pos):
            first_pos, first_tag = p, t
    return first_tag


def unclosed_tags(text: str) -> List[str]:
    out = []
    for tag in ("python", "experiment", "final_formula"):
        o, c = f"<{tag}>", f"</{tag}>"
        if o in text and text.rfind(o) > text.rfind(c):
            out.append(tag)
    return out


# ---- python sandbox --------------------------------------------------------

_DANGEROUS_PATTERNS = [
    r"import\s+os", r"import\s+sys", r"import\s+subprocess",
    r"from\s+os\s+import", r"from\s+sys\s+import", r"from\s+subprocess\s+import",
    r"import\s+pathlib", r"from\s+pathlib\s+import",
    r"import\s+pickle", r"from\s+pickle\s+import",
    r"import\s+joblib", r"from\s+joblib\s+import",
    r"import\s+glob", r"from\s+glob\s+import",
    r"import\s+importlib", r"from\s+importlib\s+import",
    r"import\s+inspect", r"from\s+inspect\s+import",
    r"import\s+shutil", r"from\s+shutil\s+import",
    r"__import__", r"\beval\(", r"\bexec\(", r"\bopen\(", r"\bfile\(",
    r"\binput\(", r"\braw_input\(", r"\bcompile\(",
    r"\bglobals\(", r"\blocals\(", r"\bgetattr\(", r"\bsetattr\(",
    r"__dict__", r"__class__", r"__mro__", r"__subclasses__",
    r"\bread_text\(", r"\bread_bytes\(", r"\bread_csv\(", r"\bread_table\(",
    r"\bread_excel\(", r"\bloadtxt\(", r"\bgenfromtxt\(",
    r"\bGridSearchCV\b", r"\bParameterGrid\b",
]

_ALLOWED_IMPORT_ROOTS = {
    "collections",
    "functools",
    "itertools",
    "math",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "statistics",
    "warnings",
}

_BLOCKED_IMPORT_ROOTS = {
    "builtins",
    "glob",
    "importlib",
    "inspect",
    "io",
    "joblib",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "subprocess",
    "sys",
}

_BLOCKED_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "file",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "raw_input",
    "setattr",
}

_BLOCKED_CALL_ATTRS = {
    "dump",
    "dumps",
    "fromfile",
    "genfromtxt",
    "get_handle",
    "load",
    "loadtxt",
    "open",
    "read_bytes",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_hdf",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_stata",
    "read_table",
    "read_text",
    "savetxt",
    "tofile",
}

_SAFE_BUILTIN_NAMES = {
    "ArithmeticError",
    "AssertionError",
    "Exception",
    "FloatingPointError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NameError",
    "OverflowError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "dict",
    "enumerate",
    "filter",
    "float",
    "hasattr",
    "int",
    "isinstance",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
}


class _PythonExecTimeout(BaseException):
    pass


def _python_timeout_handler(signum, frame):  # noqa: ARG001
    raise _PythonExecTimeout()


def _truncate_stdout(stdout: str) -> tuple[str, bool, int]:
    n = len(stdout)
    if n <= PYTHON_STDOUT_MAX_CHARS:
        return stdout, False, n
    omitted = n - PYTHON_STDOUT_MAX_CHARS
    truncated = (
        stdout[:PYTHON_STDOUT_MAX_CHARS]
        + f"\n...[python stdout truncated; omitted {omitted} characters]"
    )
    return truncated, True, n


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    unary = isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
    if unary and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
        v = float(node.operand.value)
        return -v if isinstance(node.op, ast.USub) else v
    return None


def _literal_int(node: ast.AST) -> int | None:
    v = _literal_number(node)
    if v is None or not float(v).is_integer():
        return None
    return int(v)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _range_like_count(call: ast.Call) -> int | None:
    name = _call_name(call.func)
    args = call.args
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}
    if name == "range":
        vals = [_literal_int(a) for a in args]
        if any(v is None for v in vals):
            return None
        if len(vals) == 1:
            start, stop, step = 0, vals[0], 1
        elif len(vals) == 2:
            start, stop, step = vals[0], vals[1], 1
        elif len(vals) == 3:
            start, stop, step = vals
        else:
            return None
        if step == 0:
            return None
        return max(0, math.ceil((stop - start) / step)) if step > 0 else max(0, math.ceil((start - stop) / abs(step)))
    if name.endswith("linspace") or name.endswith("logspace"):
        if "num" in kwargs:
            return _literal_int(kwargs["num"])
        if len(args) >= 3:
            return _literal_int(args[2])
        return 50
    if name.endswith("arange"):
        vals = [_literal_number(a) for a in args]
        if any(v is None for v in vals):
            return None
        if len(vals) == 1:
            start, stop, step = 0.0, vals[0], 1.0
        elif len(vals) == 2:
            start, stop, step = vals[0], vals[1], 1.0
        elif len(vals) == 3:
            start, stop, step = vals
        else:
            return None
        if step == 0:
            return None
        return max(0, math.ceil((stop - start) / step)) if step > 0 else max(0, math.ceil((start - stop) / abs(step)))
    return None


def _iter_count(node: ast.AST) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name.endswith("product"):
            repeat = 1
            for kw in node.keywords:
                if kw.arg == "repeat":
                    repeat = _literal_int(kw.value) or repeat
            counts = [_iter_count(arg) for arg in node.args]
            if not counts or any(c is None for c in counts):
                return None
            prod = 1
            for c in counts:
                prod *= c
            return prod ** repeat
        return _range_like_count(node)
    return None


def _validate_no_large_bruteforce(tree: ast.AST) -> Tuple[bool, Optional[str]]:
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_counts: List[int | None] = []
            self.error: Optional[str] = None

        def visit_For(self, node: ast.For) -> None:
            if self.error:
                return
            count = _iter_count(node.iter)
            self.loop_counts.append(count)
            depth = len(self.loop_counts)
            if depth >= BRUTE_FORCE_MAX_NESTED_LOOP_DEPTH:
                self.error = (
                    f"Code appears to use high-dimensional brute-force search "
                    f"({depth} nested for-loops). Use vectorized fitting or a "
                    "small targeted search instead."
                )
                return
            known = [c for c in self.loop_counts if c is not None]
            if len(known) == depth:
                prod = 1
                for c in known:
                    prod *= c
                if prod > BRUTE_FORCE_MAX_KNOWN_ITERATIONS:
                    self.error = (
                        f"Code appears to use large brute-force grid search "
                        f"({prod} known loop iterations > "
                        f"{BRUTE_FORCE_MAX_KNOWN_ITERATIONS}). Use a smaller "
                        "targeted search or scipy fitting."
                    )
                    return
            self.generic_visit(node)
            self.loop_counts.pop()

    v = Visitor()
    v.visit(tree)
    return (v.error is None, v.error)


def _validate_no_file_or_unsafe_imports(tree: ast.AST) -> Tuple[bool, Optional[str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                    return False, f"Import of module {alias.name!r} is not allowed in the sandbox"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                return False, "Relative imports are not allowed in the sandbox"
            root = node.module.split(".", 1)[0]
            if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
                return False, f"Import from module {node.module!r} is not allowed in the sandbox"
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            attr = name.rsplit(".", 1)[-1]
            if name in _BLOCKED_CALL_NAMES or attr in _BLOCKED_CALL_ATTRS:
                return False, f"Call to {name!r} is not allowed in the sandbox"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, f"Access to dunder attribute {node.attr!r} is not allowed in the sandbox"
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") and node.id.endswith("__"):
                return False, f"Access to dunder name {node.id!r} is not allowed in the sandbox"
    return True, None


def validate_python(code: str) -> Tuple[bool, Optional[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for pat in _DANGEROUS_PATTERNS:
        if re.search(pat, code, re.IGNORECASE):
            return False, f"Code contains blocked pattern: {pat}"
    ok, err = _validate_no_file_or_unsafe_imports(tree)
    if not ok:
        return False, err
    ok, err = _validate_no_large_bruteforce(tree)
    if not ok:
        return False, err
    return True, None


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ARG001
    if level != 0:
        raise ImportError("relative imports are not allowed in the sandbox")
    root = str(name).split(".", 1)[0]
    if root in _BLOCKED_IMPORT_ROOTS or root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_builtins() -> Dict[str, Any]:
    out = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    out["__import__"] = _safe_import
    return out


def build_sandbox(train_df=None, X_train=None, y_train=None, group_ids=None,
                  input_cols=None, target_col=None) -> Dict[str, Any]:
    """Preloaded variables exposed inside <python>. Pass whatever you have;
    `np`/`scipy`/`pd` are added automatically by `run_python`."""
    sb: Dict[str, Any] = {}
    if train_df is not None:
        sb["train_df"] = train_df
    if X_train is not None:
        sb["X_train"] = X_train
    if y_train is not None:
        sb["y_train"] = y_train
    if group_ids is not None:
        sb["group_ids_train"] = group_ids
    if input_cols is not None:
        sb["input_cols"] = list(input_cols)
    if target_col is not None:
        sb["target_col"] = target_col
    return sb


def run_python(code: str, sandbox: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate + exec `code` with `sandbox` preloaded; capture stdout. Only
    print() output is returned to the model."""
    ok, err = validate_python(code)
    if not ok:
        return {"success": False, "error_type": "ValidationError",
                "error_message": err, "code": code}
    ns: Dict[str, Any] = {"np": np, "__builtins__": _safe_builtins()}
    try:
        import scipy  # noqa: F401
        ns["scipy"] = scipy
    except ImportError:
        pass
    try:
        import pandas as pd  # noqa: F401
        ns["pd"] = pd
    except ImportError:
        pass
    ns.update(sandbox or {})
    buf = io.StringIO()
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _python_timeout_handler)
    old_timer = signal.setitimer(signal.ITIMER_REAL, PYTHON_TIMEOUT_SECONDS)
    try:
        with redirect_stdout(buf):
            exec(code, ns)
        stdout, truncated, original_len = _truncate_stdout(buf.getvalue().strip())
        return {
            "success": True,
            "stdout": stdout,
            "stdout_truncated": truncated,
            "stdout_original_chars": original_len,
            "code": code,
        }
    except _PythonExecTimeout:
        stdout, truncated, original_len = _truncate_stdout(buf.getvalue().strip())
        return {
            "success": False,
            "error_type": "TimeoutError",
            "error_message": f"Python execution exceeded {PYTHON_TIMEOUT_SECONDS}s",
            "stdout": stdout,
            "stdout_truncated": truncated,
            "stdout_original_chars": original_len,
            "code": code,
        }
    except Exception as e:
        stdout, truncated, original_len = _truncate_stdout(buf.getvalue().strip())
        return {"success": False, "error_type": type(e).__name__,
                "error_message": str(e), "traceback": traceback.format_exc(),
                "stdout": stdout, "stdout_truncated": truncated,
                "stdout_original_chars": original_len, "code": code}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


# ---- feedback strings ------------------------------------------------------

def format_python_feedback(result: Dict[str, Any]) -> str:
    if not result["success"]:
        stdout = result.get("stdout") or ""
        stdout_block = f"\nPartial stdout before failure:\n{stdout}" if stdout else ""
        return (f"<python_output>\nPython execution failed: "
                f"{result['error_type']}: {result['error_message']}"
                f"{stdout_block}\n</python_output>")
    out = result["stdout"] if result["stdout"] else "(no stdout)"
    return f"<python_output>\n{out}\n</python_output>"


def format_experiment_feedback(result: Dict[str, Any]) -> str:
    text = json.dumps(result, default=str)
    if len(text) > EXPERIMENT_OUTPUT_MAX_CHARS:
        omitted = len(text) - EXPERIMENT_OUTPUT_MAX_CHARS
        text = (
            text[:EXPERIMENT_OUTPUT_MAX_CHARS]
            + f"\n...[experiment output truncated; omitted {omitted} characters]"
        )
    return f"<experiment_output>\n{text}\n</experiment_output>"


INVALID_RESPONSE_MSG = (
    "Invalid response. Output exactly one XML block and no prose:\n"
    "<python>...code...</python>\n"
    "<experiment>{\"<input>\": [vals], \"n_samples\": 3}</experiment>  "
    "(simulator tasks only; JSON literals only, no Python expressions)\n"
    "<final_formula>...complete Python module...</final_formula>"
)


def unclosed_msg(tag: str) -> str:
    return (f"Your `<{tag}>` block was not closed (no `</{tag}>` after the open "
            f"tag). The reply was likely truncated by the token budget — try a "
            f"shorter response or split the work across turns. Re-emit a complete "
            f"primitive.")


# ---- one-call dispatch -----------------------------------------------------

def step(response_text: str, sandbox: Optional[Dict[str, Any]] = None,
         run_experiment: Optional[Callable[..., dict]] = None) -> Dict[str, Any]:
    """Apply the fixed protocol to ONE model turn.

    Picks the first-emitted tool tag and acts on it. Returns one of:
      {"action": "submit",     "submission": <module text>}
      {"action": "python",     "feedback": <str>, "ok": <bool>}
      {"action": "experiment", "feedback": <str>, "ok": <bool>}
      {"action": "invalid",    "feedback": <str>}

    The caller appends `feedback` to the conversation and continues, or stops
    when action == "submit". `run_experiment(**payload)` is only called for
    `<experiment>` tags (simulator tasks); omit it for fix-data tasks.
    """
    tag = first_tool_tag(response_text)

    if tag == "final_formula":
        ok, submitted = parse_final_formula(response_text)
        if ok:
            return {"action": "submit", "submission": submitted}

    if tag == "python":
        code = extract_python(response_text)
        if code is not None:
            res = run_python(code, sandbox)
            return {"action": "python", "ok": res["success"],
                    "feedback": format_python_feedback(res)}

    if tag == "experiment":
        exp = parse_experiment(response_text)
        if exp is not None:
            if run_experiment is None:
                fb = ('<experiment_output>\n{"error": "this task does not support '
                      '`<experiment>` (no simulator backing)."}\n</experiment_output>')
                return {"action": "experiment", "ok": False, "feedback": fb}
            result = run_experiment(**exp)
            return {"action": "experiment", "ok": "error" not in result,
                    "feedback": format_experiment_feedback(result)}

    unc = unclosed_tags(response_text)
    return {"action": "invalid",
            "feedback": unclosed_msg(unc[0]) if unc else INVALID_RESPONSE_MSG}
