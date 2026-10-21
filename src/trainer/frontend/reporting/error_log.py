# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Capture Python and native stderr before GPU setup, without a Python callback
on a Vulkan/SlangPy logging thread. Fatal exits keep the bytes already written.
"""
from __future__ import annotations

import codecs
import faulthandler
import io
import os
import sys
import threading
import traceback
from collections import deque
from contextlib import ExitStack
from pathlib import Path

from trainer.frontend.viewer.session import LOG_CAPACITY, Session


def report_exception(context: str, error: BaseException,
                     state: Session | None = None) -> None:
    """Keep caught failures visible as well as recording their full traceback."""
    message = f"[error] {context}: {type(error).__name__}: {error}"
    print(message, file=sys.stderr, flush=True)
    traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    if state is not None:
        state.log(message)
        state.toggle_log(True)


class ErrorLog:
    """Session-scoped stderr tee: durable file, original terminal, and viewer tail.

    A pipe would lose its unread tail on abort. Redirect fd 2 straight to the
    file instead; the forwarding thread is only for the live terminal/panel.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._state: Session | None = None
        self._lines: deque[str] = deque(maxlen=LOG_CAPACITY)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""

    def __enter__(self) -> ErrorLog:  # noqa: PYI034 -- supports Python 3.10, before typing.Self
        self._resources = ExitStack()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._resources.enter_context(self.path.open("ab", buffering=0))
            self._reader = self._resources.enter_context(self.path.open("rb", buffering=0))
            self._reader.seek(0, os.SEEK_END)
            # Windows ConsoleIO cannot write to a redirected file handle.
            self._stderr = self._resources.enter_context(io.TextIOWrapper(
                io.FileIO(os.dup(self._file.fileno()), "wb"), encoding="utf-8",
                errors="backslashreplace", line_buffering=True, write_through=True))
            self._previous_stderr = sys.stderr
            self._previous_stderr.flush()
            self._saved_fd = os.dup(2)
            self._resources.callback(os.close, self._saved_fd)
            self._resources.callback(self._restore_stderr)
            os.dup2(self._file.fileno(), 2)
            sys.stderr = self._stderr
            if not faulthandler.is_enabled():
                self._resources.callback(faulthandler.disable)
                faulthandler.enable(file=2, all_threads=True)
            self._thread = threading.Thread(
                target=self._forward, name="trainer-error-log", daemon=True)
            print(f"[diagnostics] errors and native crash output: {self.path}", flush=True)
            self._thread.start()
            return self
        except BaseException:
            self._resources.close()
            raise

    def _restore_stderr(self) -> None:
        os.dup2(self._saved_fd, 2)
        if sys.stderr is self._stderr:
            sys.stderr = self._previous_stderr

    def attach(self, state: Session) -> None:
        """Replay setup errors when the viewer becomes available."""
        with self._lock:
            self._state = state
            state.log(f"[diagnostics] error log: {self.path}")
            for line in self._lines:
                state.log(line)
            if self._lines:
                state.toggle_log(True)

    def _publish(self, line: str) -> None:
        if not line.strip():
            return
        self._lines.append(line)
        if self._state is not None:
            self._state.log(line)
            self._state.toggle_log(True)

    def flush(self) -> None:
        """Drain available bytes without relying on file-size metadata."""
        with self._lock:
            self._stderr.flush()
            while data := self._reader.read(65536):
                try:
                    os.write(self._saved_fd, data)
                except OSError:
                    pass  # The terminal may close; the on-disk log is intact.
                self._pending += self._decoder.decode(data)
                while "\n" in self._pending:
                    line, self._pending = self._pending.split("\n", 1)
                    self._publish(line.rstrip("\r"))

    def _forward(self) -> None:
        while not self._stop.wait(0.05):
            self.flush()

    def __exit__(self, exc_type, error, tb) -> None:
        self._stop.set()
        self._thread.join()
        try:
            if error is not None and not isinstance(error, (SystemExit, KeyboardInterrupt)):
                traceback.print_exception(exc_type, error, tb, file=self._stderr)
            self.flush()
            with self._lock:
                self._pending += self._decoder.decode(b"", final=True)
                if self._pending:
                    self._publish(self._pending)
        finally:
            self._resources.close()