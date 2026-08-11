"""Per-run output-file lifecycle for the training frontend.

Every training run writes a fixed set of JSONL/text streams (plus the optional
SADGS-schema population-trajectory writer). Opening and closing them used to be
spread across ``app.main()`` as nine ``open(...)`` calls and two divergent
teardown blocks (the Nsight capture-exit path closed a different subset than the
normal teardown).

:class:`RunArtifacts` owns that lifecycle: :meth:`create` opens every stream
(gating optional ones by their CLI flag and emitting the same console notices as
before) and :meth:`close_all` closes whatever is open. Optional streams are
``None`` when their flag is off; callers guard with the usual ``is not None``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import argparse


class RunArtifacts:
    """Owns all per-run output streams and their open/close lifecycle."""

    def __init__(
        self,
        *,
        log_txt: TextIO,
        log_json: TextIO,
        densify_json: TextIO,
        eval_curve_json: "TextIO | None",
        psnr_curve_json: "TextIO | None",
        bin_viz_json: "TextIO | None",
        profile_json: "TextIO | None",
    ) -> None:
        self.log_txt = log_txt
        self.log_json = log_json
        self.densify_json = densify_json
        self.eval_curve_json = eval_curve_json
        self.psnr_curve_json = psnr_curve_json
        self.bin_viz_json = bin_viz_json
        self.profile_json = profile_json

    @classmethod
    def create(cls, run_dir: Path, args: "argparse.Namespace") -> "RunArtifacts":
        """Open every run stream, gating optional ones by their CLI flag."""
        def _stream(name: str) -> TextIO:
            return (run_dir / name).open("w", buffering=1)

        # Always-on streams.
        log_txt = _stream("log.txt")
        log_json = _stream("log.jsonl")
        densify_json = _stream("densify_log.jsonl")

        # Per-frame profile records, opened only under --profile-every.
        profile_json = _stream("profile_log.jsonl") if args.profile_every > 0 else None

        # Full-dataset average-PSNR curve, opened only when --eval-every > 0.
        # One JSON line per eval sweep for post-hoc convergence plotting.
        eval_curve_json = _stream("eval_curve.jsonl") if args.eval_every > 0 else None
        # Pre-densify PSNR curve (--psnr-curve): a focused TEST-set PSNR sample
        # taken one iter before each densify step, streamed to psnr_curve.jsonl.
        psnr_curve_json = _stream("psnr_curve.jsonl") if args.psnr_curve else None
        # Binning/overlap debug viz (--debug-bin-viz): one JSONL line per
        # rendered eval view. Only when both the flag and --eval-every are on.
        bin_viz_json = (
            _stream("binviz.jsonl")
            if (args.debug_bin_viz and args.eval_every > 0) else None
        )
        if args.debug_bin_viz and args.eval_every <= 0:
            print("[debug-bin-viz] ignored: requires --eval-every > 0.")

        return cls(
            log_txt=log_txt, log_json=log_json, densify_json=densify_json, eval_curve_json=eval_curve_json,
            psnr_curve_json=psnr_curve_json, bin_viz_json=bin_viz_json,
            profile_json=profile_json,
        )

    def close_all(self) -> None:
        for stream in (
            self.log_txt, self.log_json, self.densify_json, self.eval_curve_json, self.psnr_curve_json,
            self.bin_viz_json, self.profile_json
        ):
            if stream is not None:
                stream.close()
