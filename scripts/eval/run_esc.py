#!/usr/bin/env python
"""Entry point for the ESC harness.

Adds this directory (``scripts/eval``) to ``sys.path`` so the ``esc`` package
resolves whether the harness is launched by path or as a module, then delegates
to :func:`esc.cli.main`.

Examples:
    python scripts/eval/run_esc.py analyze
    python scripts/eval/run_esc.py score -c 0.7 -e 0.6 -u 0.8 -l 0.7 -t 0.5 -s 0.9
    python scripts/eval/run_esc.py evaluate --agent-cmd "my-agent --prompt-file {prompt_file}"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from esc.cli import main  # noqa: E402  (path set above)

if __name__ == "__main__":
    raise SystemExit(main())
