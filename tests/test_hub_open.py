# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from types import SimpleNamespace
import threading

import pytest
from slangpy import platform

from trainer.frontend import hub as hub_module
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from trainer.frontend.hub import TrainingHub
from trainer.frontend.reconstruction import ReconTools
from trainer.frontend.viewer.session import DatasetInfo, MODE_HUB, ResultInfo, Session


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
    ("confirm_alt", ["clear", "replacement"]), ("cancel", []), ("confirm", ["export", "clear", "replacement"]),
])
def test_replacing_dataset_resolves_pending_path_once(command, expected, monkeypatch):
    state, hub = hub_for_one_request(monkeypatch)
    state.training_completed(ResultInfo(iterations=10))
    events = []
    monkeypatch.setattr(hub, "_hub_clear_session", lambda *, notify: events.append("clear"))
    monkeypatch.setattr(hub, "_hub_load_dataset", events.append)
    def export():
        events.append("export")
        return True
    monkeypatch.setattr(hub, "_hub_export_ply", export)
    hub._hub_dropped("replacement")
    assert state.snapshot().confirm.alt_command == "load_dropped"
    state.request(command)
    hub._hub_choose_scene(None)
    hub._hub_load_pending_drop()
    assert events == expected


def test_cancelling_save_keeps_unsaved_model_and_current_dataset(monkeypatch):
    state, hub = hub_for_one_request(monkeypatch)
    state.training_completed(ResultInfo(iterations=10))
    monkeypatch.setattr(hub, "_hub_export_ply", lambda: False)
    hub._hub_dropped("replacement")
    state.request("confirm")
    hub._hub_choose_scene(None)
    assert state.snapshot().unexported and hub._pending_drop is None


@pytest.mark.parametrize("save", [True, False])
def test_export_and_train_waits_for_a_successful_save(monkeypatch, save):
    state, hub = hub_for_one_request(monkeypatch)
    state.dataset_ready(DatasetInfo(path="scene", scene_dir="scene"))
    state.training_completed(ResultInfo(iterations=10))
    assert hub._hub_request_training() is None
    assert state.snapshot().confirm.confirm_label == "Export and train"
    monkeypatch.setattr(hub, "_hub_export_ply", lambda: save)
    state.request("confirm")
    assert hub._hub_choose_scene(None) == ("scene" if save else None)


def test_next_run_setup_previews_original_points_and_back_restores_model(monkeypatch):
    state = Session(MODE_HUB)
    state.dataset_ready(DatasetInfo(path="scene", scene_dir="scene"))
    state.training_completed(ResultInfo(iterations=10))
    events = []
    commands = iter(("configure_next_run", "back_to_result", None))
    def alive():
        command = next(commands)
        if command:
            state.request(command)
        return bool(command)
    viewer = SimpleNamespace(state=state, is_alive=alive,
                             preview_points=events.append, push_snapshot=events.append)
    hub = TrainingHub(SimpleNamespace(viewer=viewer, _viewer=viewer, _pipeline="trained model"))
    monkeypatch.setattr(hub, "_cached_colmap", lambda path: SimpleNamespace(initial_points="point cloud"))
    monkeypatch.setattr(hub_module.time, "sleep", lambda seconds: None)
    hub._hub_choose_scene(None)
    assert events == ["point cloud", "trained model"]
    assert state.snapshot().unexported and not state.snapshot().next_run_setup


@pytest.mark.parametrize("save", [True, False])
def test_native_export_dialog_saves_only_on_selection(tmp_path, monkeypatch, save):
    from trainer.backend import export
    state, hub = hub_for_one_request(monkeypatch)
    state.training_completed(ResultInfo(iterations=10))
    hub.app.pipeline = hub.app._pipeline = object()
    hub.app.viewer = SimpleNamespace(suspend_rendering=lambda: None, resume_rendering=lambda: None)
    dest = tmp_path / "my-model.ply"
    monkeypatch.setattr(platform, "save_file_dialog", lambda _: dest.with_suffix("") if save else None)
    monkeypatch.setattr(export, "save_ply", lambda _, path: path.write_bytes(b"model"))
    assert hub._hub_export_ply() == save
    assert dest.is_file() == save
    assert state.snapshot().unexported != save


def test_new_dataset_after_release_writes_a_separate_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = TrainingApplication(build_parser().parse_args(["--hub", "--run-name", "garden"]))
    original = app.paths.root
    (original / "config.json").write_text("{}")
    app._begin_repeat_run()
    assert app.paths.root != original and app.paths.root.is_dir()
    assert (original / "config.json").read_text() == "{}"


def test_failed_export_preserves_existing_file_and_model(tmp_path, monkeypatch):
    from trainer.backend import export
    state, hub = hub_for_one_request(monkeypatch)
    state.training_completed(ResultInfo(iterations=10))
    hub.app.pipeline = hub.app._pipeline = object()
    hub.app.viewer = SimpleNamespace(suspend_rendering=lambda: None, resume_rendering=lambda: None)
    dest = tmp_path / "model.ply"
    dest.write_bytes(b"previous")
    monkeypatch.setattr(platform, "save_file_dialog", lambda _: dest)
    def fail(_, path):
        path.write_bytes(b"partial")
        raise OSError("out of disk space")
    monkeypatch.setattr(export, "save_ply", fail)
    assert not hub._hub_export_ply()
    assert dest.read_bytes() == b"previous" and state.snapshot().unexported


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
