"""``python -m trainer.contracts`` entry point.

The contract codegen lives in :mod:`trainer.contracts.codegen`; this keeps the
long-standing ``python -m trainer.contracts`` invocation (and the generated
Slang headers' regenerate hint) working now that ``contracts`` is a package.
"""

from __future__ import annotations

from trainer.contracts.codegen import main

if __name__ == "__main__":
    raise SystemExit(main())
