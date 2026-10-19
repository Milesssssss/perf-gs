# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""``python -m esc`` entry point (requires scripts/eval on sys.path)."""

from __future__ import annotations

from esc.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
