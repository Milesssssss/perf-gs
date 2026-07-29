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
        densify_diag_json: "TextIO | None",
        eval_curve_json: "TextIO | None",
        psnr_curve_json: "TextIO | None",
        bin_viz_json: "TextIO | None",
        profile_json: "TextIO | None",
    ) -> None:
        self.log_txt = log_txt
        self.log_json = log_json
        self.densify_json = densify_json
        self.densify_diag_json = densify_diag_json
        self.eval_curve_json = eval_curve_json
        self.psnr_curve_json = psnr_curve_json
        self.bin_viz_json = bin_viz_json
        self.profile_json = profile_json

    @classmethod
    def create(cls, run_dir: Path, args: "argparse.Namespace") -> "RunArtifacts":
        """Open every run stream, gating optional ones by their CLI flag."""
        def _stream(name: str) -> TextIO:
            return (run_dir / name).open("w", buffering=1)

        # Always-on streams
        log_txt = _stream("log.txt")
        log_json = _stream("log.jsonl")
        densify_json = _stream("densify_log.jsonl")
        profile_json = _stream("profile_log.jsonl") if args.profile_every > 0 else None
        densify_diag_json = _stream("densify_diag.jsonl") if args.densify_diag else None
        eval_curve_json = _stream("eval_curve.jsonl") if args.eval_every > 0 else None
        psnr_curve_json = _stream("psnr_curve.jsonl") if args.psnr_curve else None

        bin_viz_json = (
            _stream("binviz.jsonl")
            if (args.debug_bin_viz and args.eval_every > 0) else None
        )
        if args.debug_bin_viz and args.eval_every <= 0:
            print("[debug-bin-viz] ignored: requires --eval-every > 0.")

        return cls(
            log_txt=log_txt, log_json=log_json, densify_json=densify_json,
            densify_diag_json=densify_diag_json, eval_curve_json=eval_curve_json,
            psnr_curve_json=psnr_curve_json, bin_viz_json=bin_viz_json,
            profile_json=profile_json
        )

    def close_all(self) -> None:
        """Close every open stream (optional ones only if present).

        Idempotent: safe to call from both the Nsight capture-exit path and the
        normal teardown. Streams are line-buffered, so no data is lost if a
        stream was already flushed.
        """
        for stream in (
            self.log_txt, self.log_json, self.densify_json,
            self.densify_diag_json, self.eval_curve_json, self.psnr_curve_json,
            self.bin_viz_json, self.profile_json, self.popstat,
        ):
            if stream is not None:
                stream.close()
