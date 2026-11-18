# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from types import SimpleNamespace
import threading

import pytest
from slangpy import platform

from trainer.frontend import hub as hub_module
from trainer.frontend.cli import build_parser
from trainer.frontend.hub import TrainingHub
from trainer.frontend.reconstruction import ReconTools
from trainer.frontend.viewer.session import MODE_HUB, ResultInfo, Session


def hub_for_one_request(monkeypatch):
    state = Session(MODE_HUB)
    alive = iter((True, False))
    viewer = SimpleNamespace(state=state, is_alive=lambda: next(alive))
    monkeypatch.setattr(hub_module.time, "sleep", lambda seconds: None)
    return state, TrainingHub(SimpleNamespace(_viewer=viewer))


@pytest.mark.parametrize("command", ["browse", "new_dataset"])
@pytest.mark.parametrize("selection", ["scene", None, NotImplementedError()])
def test_open_folder_handles_platform_selection(command, selection, monkeypatch):
    state, hub = hub_for_one_request(monkeypatch)
    def choose():
        assert threading.current_thread() is threading.main_thread()
        if isinstance(selection, Exception):
            raise selection
        return selection
    monkeypatch.setattr(platform, "choose_folder_dialog", choose)
    loads = []
    monkeypatch.setattr(hub, "_hub_dropped", loads.append)
    state.request(command)
    hub._hub_choose_scene(None)
    assert loads == (["scene"] if selection == "scene" else [])
    notices = [n.text for n in state.snapshot().notices]
    if isinstance(selection, Exception):
        assert "Drop a dataset folder" in notices[-1]
    else:
        assert not notices


@pytest.mark.parametrize("command,expected", [
    ("confirm", ["clear", "replacement"]), ("cancel", []), ("confirm_alt", ["export"]),
])
def test_replacing_dataset_resolves_pending_path_once(command, expected, monkeypatch):
    state, hub = hub_for_one_request(monkeypatch)
    state.training_completed(ResultInfo(iterations=10))
    events = []
    monkeypatch.setattr(hub, "_hub_clear_session", lambda *, notify: events.append("clear"))
    monkeypatch.setattr(hub, "_hub_load_dataset", events.append)
    monkeypatch.setattr(hub, "_hub_export_ply", lambda: events.append("export"))
    hub._hub_dropped("replacement")
    assert state.snapshot().confirm.command == "load_dropped"
    state.request(command)
    hub._hub_choose_scene(None)
    hub._hub_load_pending_drop()
    assert events == expected


@pytest.mark.parametrize("scene", [None, "explicit-scene"])
def test_startup_opens_an_explicit_scene_once(scene, monkeypatch):
    state = Session(MODE_HUB)
    viewer = SimpleNamespace(state=state, start=lambda: None)
    args = build_parser().parse_args(["--hub"] + (["--scene", scene] if scene else []))
    app = SimpleNamespace(args=args, error_log=None, _create_device=lambda: None,
                          device=None, gpu_lock=None, _viewer=viewer)
    hub = TrainingHub(app)
    monkeypatch.setattr(hub_module.ViewerSession, "create_hub", lambda **kwargs: viewer)
    monkeypatch.setattr(ReconTools, "is_ready", lambda self: True)
    loaded = []
    monkeypatch.setattr(hub, "_hub_dropped", loaded.append)
    monkeypatch.setattr(hub, "_hub_choose_scene", lambda tools: None)
    hub.run()
    assert loaded == ([scene] if scene else [])
