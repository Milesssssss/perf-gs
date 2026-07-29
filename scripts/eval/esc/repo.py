"""Shared repository scanning.

A tiny, dependency-free file enumerator used by the static-quality and locality
analysers. Honours the configured excludes and source suffixes and reads text
defensively so a single undecodable file never breaks a scan.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    """One scanned source file and its text."""

    path: Path
    rel: str
    text: str

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def line_count(self) -> int:
        return self.text.count("\n") + 1


class RepoScanner:
    """Enumerates source files under ``root`` respecting excludes/suffixes."""

    def __init__(self, root: Path, excludes: tuple[str, ...],
                 suffixes: tuple[str, ...]) -> None:
        self.root = root.resolve()
        self._excludes = frozenset(excludes)
        self._suffixes = frozenset(suffixes)

    def _is_excluded(self, path: Path) -> bool:
        return any(part in self._excludes for part in path.relative_to(self.root).parts)

    def iter_files(self) -> Iterator[SourceFile]:
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix not in self._suffixes:
                continue
            if self._is_excluded(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            yield SourceFile(path=path, rel=str(path.relative_to(self.root)).replace("\\", "/"), text=text)

    def python_files(self) -> list[SourceFile]:
        return [f for f in self.iter_files() if f.suffix == ".py"]

    def all_files(self) -> list[SourceFile]:
        return list(self.iter_files())
