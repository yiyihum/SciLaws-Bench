#!/usr/bin/env python3
"""Post-run integrity audit of transcripts (see runner/audit.py).

  python scripts/audit_transcripts.py --run-dir runs/gpt56_luna_parallel --tasks-dir tasks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.audit import main  # noqa: E402

if __name__ == "__main__":
    main()
