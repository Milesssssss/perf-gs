# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Thin entry point for the v2 trainer; all training logic lives in :mod:`trainer.frontend.app`."""

from trainer.frontend.app import main
from trainer.frontend.cli import build_parser

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
