# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Per-run output-file lifecycle; optional streams are ``None`` when their CLI flag is off."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from trainer.config.resolve import end_of_run_cadence

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
        profile_json: "TextIO | None",
    ) -> None:
        self.log_txt = log_txt
        self.log_json = log_json
        self.densify_json = densify_json
        self.eval_curve_json = eval_curve_json
        self.psnr_curve_json = psnr_curve_json
        self.profile_json = profile_json

    @classmethod
    def create(cls, run_dir: Path, args: "argparse.Namespace") -> "RunArtifacts":
        """Open every run stream, gating optional ones by their CLI flag."""
        def _stream(name: str) -> TextIO:
            return (run_dir / name).open("w", buffering=1)

        log_txt = _stream("log.txt")
        log_json = _stream("log.jsonl")
        densify_json = _stream("densify_log.jsonl")

        profile_json = _stream("profile_log.jsonl") if args.profile_every > 0 else None

        # This stream also gates the loop's eval sweeps: no stream, no sweep.
        eval_every = end_of_run_cadence(args.eval_every, args.iters)
        do_eval = ((args.eval_every is None or eval_every > 0 or bool(getattr(args, "eval_iters", None)))
                   and args.evaluation)
        eval_curve_json = _stream("eval_curve.jsonl") if do_eval else None
        psnr_curve_json = _stream("psnr_curve.jsonl") if args.psnr_curve else None

        return cls(
            log_txt=log_txt, log_json=log_json, densify_json=densify_json, eval_curve_json=eval_curve_json,
            psnr_curve_json=psnr_curve_json, profile_json=profile_json,
        )

    def close_all(self) -> None:
        for stream in (
            self.log_txt, self.log_json, self.densify_json, self.eval_curve_json, self.psnr_curve_json,
            self.profile_json
        ):
            if stream is not None:
                stream.close()
