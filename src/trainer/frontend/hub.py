# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Interactive session lifecycle around a reusable training application."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from trainer.data.colmap import load_colmap
from trainer.frontend.profiling.profiler import apply_profile_session_overrides
from trainer.frontend.reporting.error_log import report_exception
from trainer.frontend.viewer.session import (
    Confirm, DatasetInfo, ERROR, KIND_COLMAP, KIND_IMAGES, OK, ResultInfo,
    Session, ToolsState, WARN, format_duration,
)
from trainer.frontend.viewer.viewer_control import ViewerSession

if TYPE_CHECKING:
    from trainer.frontend.app import TrainingApplication


def _progress_relay(state: Session, parser):
    """Relay subprocess lines to the session log and, via ``parser``, to the progress bar."""
    def relay(line: str) -> None:
        state.log(line)
        parsed = parser.feed(line)
        if parsed is not None:
            state.set_progress(parsed.label, parsed.fraction)
    return relay

class TrainingHub:
    def __init__(self, app: TrainingApplication) -> None:
        self.app = app

    def run(self) -> None:
        """Run the interactive hub until the window closes; a session outlives its runs.
        The viewer and this loop communicate only through ``session.state``: the panel posts
        one-shot commands, this loop consumes them and publishes phase/progress/log back."""
        from trainer.frontend.reconstruction import ReconTools
        from trainer.frontend.settings import TrainingSetup

        self.app.args.no_viewer = False  # the hub is the viewer
        self.app.profiling_enabled = apply_profile_session_overrides(self.app.args)
        self.app._create_device()

        # Seeded from the CLI args; the panel edits it and it is applied back on Train.
        setup = TrainingSetup.from_args(self.app.args)
        self.app.viewer = ViewerSession.create_hub(
            device=self.app.device,
            width=self.app.args.width or 960,
            height=self.app.args.height or 540,
            fovy=getattr(self.app.args, "fovy", 45.0),
            gpu_lock=self.app.gpu_lock,
            sh_degree=self.app.args.sh_degree,
            mip_enabled=self.app.args.mip_splatting,
            mip_kernel_size=(self.app.args.mip_kernel_size if self.app.args.mip_splatting else 0.3),
            setup=setup,
        )
        if self.app.error_log is not None:
            self.app.error_log.attach(self.app.viewer.state)
        self.app.viewer.start()

        state = self.app._viewer.state
        state.set_recon_cuda(self.app.args.recon_cuda)
        tools = ReconTools(colmap_bin=self.app.args.colmap_bin,
                           glomap_bin=self.app.args.glomap_bin,
                           recon_cuda=self.app.args.recon_cuda)
        state.tools(ToolsState.READY if tools.is_ready() else ToolsState.MISSING)
        state.notify("Viewer ready - drop a dataset folder", OK)
        print("[hub] viewer up — drag a dataset folder onto the window.")

        while True:
            scene_dir = self._hub_choose_scene(tools)
            if scene_dir is None:
                print("[hub] window closed; exiting.")
                return
            self._hub_train_scene(scene_dir, setup)

    def _hub_choose_scene(self, tools) -> "str | None":
        """Service panel requests until a scene is ready; ``None`` once the window closes.
        Every branch publishes a state transition so the UI reacts on the same frame."""
        state = self.app._viewer.state
        while self.app._viewer.is_alive():
            dropped = state.take_dropped()
            if dropped:
                self._hub_dropped(dropped)

            command = state.take_command()
            if command in ("confirm", "confirm_alt"):
                command = state.take_confirm(accept=True,
                                             alt=command == "confirm_alt")
                if command == "export" and getattr(self, "_pending_drop", None):
                    # "Export first" must not then destroy the result by continuing the load.
                    self.app._pending_drop = None
                    state.notify("Exporting first - drop the dataset again after",
                                 WARN)
            elif command == "cancel":
                state.take_confirm(accept=False)
                state.notify("Cancelled", OK)
                command = None

            if command == "install_tools":
                self._hub_setup_tools(tools)
            elif command == "reconstruct":
                self._hub_reconstruct(tools)
            elif command == "export":
                self._hub_export_ply()
            elif command == "reveal":
                self._hub_reveal_export()
            elif command == "clear":
                self._hub_request_clear()
            elif command == "clear!":
                self._hub_clear_session()
            elif command == "new_dataset":
                state.notify("Drop the next dataset onto the window", OK)
            elif command == "load_dropped":
                self._hub_load_pending_drop()
            elif command in ("train", "restart"):
                scene = self._hub_request_training(command)
                if scene is not None:
                    return scene
            elif command == "train!":
                return state.scene_dir
            time.sleep(0.05)
        return None

    def _hub_dropped(self, path: str) -> None:
        """Load a dropped folder, asking first if it would replace an existing result."""
        state = self.app._viewer.state
        if state.snapshot().has_result:
            self.app._pending_drop = path
            unexported = state.snapshot().unexported
            state.ask(Confirm(
                key="dataset", title="Load a new dataset?",
                body=("The current reconstruction has not been exported and will "
                      "be discarded." if unexported else
                      "The current reconstruction will be cleared."),
                command="load_dropped", confirm_label="Discard and load",
                alt_label="Export first" if unexported else "",
                alt_command="export" if unexported else ""))
            state.notify("Current reconstruction would be replaced", WARN)
            return
        self._hub_load_dataset(path)

    def _hub_load_pending_drop(self) -> None:
        """The confirmed half of :meth:`_hub_dropped`."""
        path, self.app._pending_drop = getattr(self, "_pending_drop", None), None
        if path is None:
            return
        self._hub_clear_session(notify=False)
        self._hub_load_dataset(path)

    def _hub_load_dataset(self, path: str) -> None:
        """INSPECTING -> detected -> (LOADING) -> READY / ERROR."""
        from trainer.frontend.reconstruction import classify_dataset

        state = self.app._viewer.state
        state.dataset_inspecting(path)
        state.log(f"dropped {path}")
        kind = classify_dataset(path)
        state.dataset_detected(kind, scene_dir=path if kind == KIND_COLMAP else "")
        state.log(f"detected: {kind}")
        if kind == KIND_COLMAP:
            self._hub_preview_scene(path)
            return
        if kind == KIND_IMAGES:
            state.dataset_ready(DatasetInfo(path=path, kind=kind))
            state.notify("Image folder loaded - reconstruct it to train", OK)
            return
        state.dataset_failed("No images and no sparse model here. A dataset is a "
                             "folder of images, or one with sparse/ + images/.")
        state.notify("Could not load dataset", ERROR)

    def _hub_request_training(self, command: str) -> "str | None":
        """Start or restart. A restart discards a finished result, so it asks."""
        state = self.app._viewer.state
        snap = state.snapshot()
        if not snap.trainable:
            state.notify("No trainable scene yet", WARN)
            return None
        if command == "restart" and snap.has_result:
            state.ask(Confirm(
                key="result", title="Restart training?",
                body=("This reconstruction has not been exported. Training again "
                      "discards it and optimizes the same dataset from scratch."
                      if snap.unexported else
                      "The current reconstruction is discarded and training "
                      "restarts from scratch on the same dataset."),
                command="train!", confirm_label="Discard and restart",
                alt_label="Export first" if snap.unexported else "",
                alt_command="export" if snap.unexported else ""))
            return None
        return state.scene_dir

    def _hub_request_clear(self) -> None:
        state = self.app._viewer.state
        if state.snapshot().unexported:
            state.ask(Confirm(
                key="result", title="Clear the session?",
                body="This reconstruction has not been exported and will be lost.",
                command="clear!", confirm_label="Discard and clear",
                alt_label="Export first", alt_command="export"))
            return
        self._hub_clear_session()

    def _hub_train_scene(self, scene_dir: str, setup) -> None:
        """Build and train one run on ``scene_dir``, feeding the running viewer.
        Setup runs with rendering suspended (queue races); a failure leaves the hub usable."""
        state = self.app._viewer.state
        print(f"[hub] training on {scene_dir}")
        # Acknowledge the click first: scene load, pipeline build and uploads precede iteration 1.
        state.reset_for_new_run()
        state.training_initializing(self.app.args.iters)
        state.begin_activity("Loading scene and building the pipeline")
        self.app.viewer.clear_stop_request()
        state.log(f"preparing training on {scene_dir}")
        self.app.viewer.suspend_rendering()
        try:
            self.app._begin_repeat_run()
            # The panel's settings must become the run's args before anything reads them.
            self._apply_hub_setup(setup)
            self.app._prepare_scene(scene_dir)
            self.app._create_pipeline_for_scene(self.app._sceneContext)
            self.app._prepare_textures()
            self.app._start_profiler()
            self.app._create_training_services()
        except Exception as error:                     # noqa: BLE001
            self._hub_run_failed("building the run", error)
            return
        finally:
            self.app.viewer.resume_rendering()

        self.app._attach_scene_to_viewer()
        self.app.viewer.push_snapshot(self.app._pipeline)  # show the initial gaussians
        try:
            self.app._run_training_and_post(hold_viewer=False)
        except Exception as error:                     # noqa: BLE001
            self._hub_run_failed("training", error)
            return
        # The pipeline stays alive: Export writes it and the viewer keeps showing it.
        stats = state.snapshot().stats
        stopped = self.app.viewer.stop_requested()
        state.training_completed(ResultInfo(
            iterations=stats.step, target_iterations=stats.total,
            wall_s=stats.elapsed_s, splats=stats.gaussians,
            run_dir=str(self.app.paths.root)))
        state.set_export_name(f"{self.app.paths.root.name}.ply")
        state.notify(f"Training stopped at iteration {stats.step:,}" if stopped
                     else f"Training complete in "
                          f"{format_duration(stats.elapsed_s)}", OK)
        state.log(f"training finished — .ply export goes to {self.app.paths.root}")

    def _hub_run_failed(self, what: str, error: Exception) -> None:
        """Report a failed hub run and leave the session usable."""
        state = self.app._viewer.state
        state.training_failed(f"{what}: {error}")
        state.notify(f"Training failed while {what}", ERROR)
        report_exception(f"[hub] {what} failed", error, state)

    def _hub_export_ply(self) -> None:
        """Write the trained gaussians to ``<run>/output.ply`` from the panel."""
        from trainer.backend.export import save_ply

        state = self.app._viewer.state
        if self.app.pipeline is None:
            state.result_failed("Nothing to export yet - train a scene first.")
            state.notify("Nothing to export", WARN)
            return
        path = self.app.paths.root / (state.export_name or "output.ply")
        state.begin_activity(f"Exporting {path.name}")
        state.log(f"exporting {path.name} …")
        # save_ply reads back the gaussian buffers, so the render thread must be off the queue.
        self.app.viewer.suspend_rendering()
        try:
            save_ply(self.app._pipeline, path)
        except Exception as error:                     # noqa: BLE001
            state.result_failed(f"Could not write {path}: {error}")
            state.notify("Export failed", ERROR)
            report_exception("[hub] export failed", error, state)
            return
        finally:
            self.app.viewer.resume_rendering()
        size_mb = path.stat().st_size / (1024 * 1024)
        state.result_exported(str(path.resolve()), size_mb)
        state.notify(f"Exported {path.name}", OK)
        state.log(f"exported {size_mb:,.1f} MB -> {path.resolve()}")
        print(f"[hub] exported {path.resolve()} ({size_mb:,.1f} MB)")

    def _hub_reveal_export(self) -> None:
        """Open the exported file's folder in the OS file manager.
        Windows-only on purpose: ``os.startfile`` needs no extra dependency."""
        state = self.app._viewer.state
        exported = state.snapshot().result.exported_path
        if not exported:
            return
        folder = Path(exported).parent
        try:
            os.startfile(str(folder))   # user-initiated, this machine only
        except (AttributeError, OSError) as error:
            state.result_failed(f"Could not open {folder}: {error}")
            state.notify("Could not open the folder", ERROR)
            report_exception("[hub] opening the export folder failed", error, state)

    def _hub_clear_session(self, *, notify: bool = True) -> None:
        """Forget the dataset and the finished run so the hub can start over.
        The window, device, installed tools and log survive."""
        state = self.app._viewer.state
        self.app.viewer.suspend_rendering()
        try:
            self.app._release_run_resources()
        finally:
            self.app.viewer.resume_rendering()
        self.app.viewer.clear_view()
        state.clear_session()
        if notify:
            state.notify("Session cleared", OK)
        state.log("session cleared — drag another dataset onto the window")
        print("[hub] session cleared")

    def _apply_hub_setup(self, setup) -> None:
        """Fold the panel's training setup into this run's args; call with rendering suspended.
        Also rebuilds the viewer's pipeline, whose SH buffers, Mip raster path and compiled
        shader specialization are fixed at construction."""
        state = self.app._viewer.state
        for change in setup.apply_to(self.app.args):
            state.log(f"[setup] {change}")
        self.app.viewer.reconfigure_model(
            sh_degree=self.app.args.sh_degree,
            mip_enabled=self.app.args.mip_splatting,
            mip_kernel_size=(self.app.args.mip_kernel_size
                             if self.app.args.mip_splatting else 0.3),
            slang_session=self.app.slang_session(),
        )

    def _hub_setup_tools(self, tools) -> None:
        """Download the colmap/glomap binaries, with a determinate bar."""
        from trainer.frontend.reconstruction.progress import setup_parser

        state = self.app._viewer.state
        state.tools(ToolsState.INSTALLING)
        # The panel's checkbox is the install-time build choice (see ReconTools).
        tools.recon_cuda = state.recon_cuda
        ready = tools.setup(progress=_progress_relay(state, setup_parser()))
        state.tools(ToolsState.READY if ready else ToolsState.MISSING)
        state.notify("Reconstruction tools ready" if ready
                     else "Tool install failed - see diagnostics",
                     OK if ready else ERROR)
        state.log("reconstruction tools ready" if ready
                  else "tool setup failed — see the log above")

    def _hub_reconstruct(self, tools) -> None:
        """Run images -> COLMAP for the dropped dataset, with a determinate bar."""
        from trainer.frontend.reconstruction import reconstruct
        from trainer.frontend.reconstruction.progress import reconstruction_parser

        state = self.app._viewer.state
        source = state.dataset_path
        if not source:
            state.notify("No dataset to reconstruct", WARN)
            return
        state.dataset_reconstructing()
        out = Path(source) / "reconstruction"
        result = reconstruct(tools, source, out, matcher=state.matcher,
                             progress=_progress_relay(state, reconstruction_parser()))
        if result is None:
            state.dataset_failed("Reconstruction failed - see diagnostics for the "
                                 "COLMAP output.")
            state.notify("Reconstruction failed", ERROR)
            return
        state.dataset_detected(KIND_IMAGES, scene_dir=str(result))
        state.notify("Reconstruction complete", OK)
        state.log("reconstruction complete")
        self._hub_preview_scene(str(result))

    def _hub_preview_scene(self, scene_dir: str) -> None:
        """Show the dataset's SfM points before Train so a bad reconstruction is visible early.
        Best-effort: a scene the loader rejects leaves the hub usable."""
        state = self.app._viewer.state
        state.dataset_loading("Reading the COLMAP model")
        state.log("loading the SfM point cloud …")
        started = time.time()
        # Suspend during the load too: the render thread holds the GIL and starves the parsers.
        self.app.viewer.suspend_rendering()
        try:
            dataset = load_colmap(
                scene_dir,
                width=self.app.args.width,
                height=self.app.args.height,
                downsample=self.app.args.downsample,
                init_opacity=self.app.args.init_opacity,
            )
            shown = self.app.viewer.preview_points(dataset.initial_points)
        except Exception as error:                    # noqa: BLE001
            state.dataset_failed(str(error))
            state.notify("Could not load dataset", ERROR)
            report_exception(f"[hub] cannot preview this dataset: {scene_dir}", error, state)
            return
        finally:
            self.app.viewer.resume_rendering()

        # Every camera, not the eval split, so the whole capture can be inspected before a run.
        self.app.viewer.attach_scene(
            [(entry.image_name, entry.camera) for entry in dataset.images],
            snap=True)
        first = dataset.images[0].camera if dataset.images else None
        info = state.snapshot().dataset
        state.dataset_ready(DatasetInfo(
            path=info.path or scene_dir, kind=info.kind or KIND_COLMAP,
            images=len(dataset.images),
            width=int(getattr(first, "width", 0) or 0),
            height=int(getattr(first, "height", 0) or 0),
            points=int(shown), scene_dir=info.scene_dir or scene_dir))
        state.result_initialized(int(shown))
        state.notify(f"{info.kind_label} loaded - {len(dataset.images)} images", OK)
        state.log(f"previewing {shown:,} SfM points from {len(dataset.images)} "
                  f"cameras — press Train to start")
        print(f"[hub] previewing {shown:,} initial points from {scene_dir} "
              f"({time.time() - started:.1f}s)")
