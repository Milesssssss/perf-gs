# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Keyboard bindings for the viewer: hold keys and press keys."""
from __future__ import annotations

from collections.abc import Callable, Iterable


class InputMap:
    """Keyboard bindings, with a self-describing help table."""

    def __init__(self) -> None:
        self._hold: dict[object, str] = {}
        self._press: dict[object, Callable[[], None]] = {}
        self._help: list[tuple[str, str]] = []
        self.held: set[str] = set()

    def hold(self, key, tag: str) -> InputMap:
        self._hold[key] = tag
        return self

    def hold_many(self, pairs: Iterable[tuple[object, str]]) -> InputMap:
        for key, tag in pairs:
            self._hold[key] = tag
        return self

    def press(self, key, on_press: Callable[[], None]) -> InputMap:
        self._press[key] = on_press
        return self

    def document(self, keys: str, action: str) -> InputMap:
        """Record a help row (shown in the panel's CONTROLS section)."""
        self._help.append((keys, action))
        return self

    def describe(self) -> list[tuple[str, str]]:
        return list(self._help)

    def handle(self, event) -> None:
        """Update ``held`` from key-down/up and fire ``press`` callbacks."""
        tag = self._hold.get(event.key)
        if event.is_key_press() or event.is_key_repeat():
            if tag is not None:
                self.held.add(tag)
        elif event.is_key_release():
            if tag is not None:
                self.held.discard(tag)
        if event.is_key_press():
            handler = self._press.get(event.key)
            if handler is not None:
                handler()

    def release_all(self) -> None:
        """Drop all held keys (e.g. on focus loss, so movement can't stick)."""
        self.held.clear()
