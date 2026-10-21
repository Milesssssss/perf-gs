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
    """Adapt a subprocess line stream to the session state.

    Every line lands in the log (wrapped, never clipped, by the panel) and is
    ALSO fed to a stage parser that turns it into a determinate fraction — which
    is what the viewer's progress bar shows for setup/reconstruction.
    """
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
        """Interactive hub: bring up a scene-less viewer, let the user drag-drop a
        dataset and choose Reconstruct (images -> GLOMAP/COLMAP) or Train (a
        COLMAP dataset) from the panel, then train the chosen dataset in-process
        feeding the same viewer. CLI training (start_training) is unaffected.

        The session OUTLIVES a run: when training finishes the result stays on
        screen and in memory, and the panel offers Export .ply, Train again (with
        the TRAINING SETUP values it now holds), another dataset, or Clear session.
        Only closing the window ends it.

        The viewer and this method communicate ONLY through ``session.state``: the
        panel produces one-shot commands, this loop consumes them and publishes
        phase/progress/log back (see :mod:`trainer.frontend.viewer.state`)."""
        from trainer.frontend.reconstruction import ReconTools
        from trainer.frontend.settings import (
            TrainingSetup,
            viewer_stream_capacity,
        )

        self.app.args.no_viewer = False  # the hub IS the viewer
        self.app.profiling_enabled = apply_profile_session_overrides(self.app.args)
        self.app._create_device()

        # The panel's TRAINING SETUP edits this store; it is seeded from the CLI
        # args and applied back onto them when the user presses Train.
        setup = TrainingSetup.from_args(self.app.args)
        self.app.viewer = ViewerSession.create_hub(
            device=self.app.device,
            width=self.app.args.width or 960,
            height=self.app.args.height or 540,
            max_splats=viewer_stream_capacity(self.app.args.max_splats,
                                              self.app.args.stream_max_splats),
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

        # ONE session, MANY runs: a finished run no longer ends the hub. Control
        # goes straight back to the panel, where the result can be exported, the
        # same scene re-trained with different parameters, another dataset dropped,
        # or the whole session cleared. Only closing the window ends this.
        while True:
            scene_dir = self._hub_choose_scene(tools)
            if scene_dir is None:
                print("[hub] window closed; exiting.")
                return
            self._hub_train_scene(scene_dir, setup)

    def _hub_choose_scene(self, tools) -> "str | None":
        """Service panel requests until a scene is ready to train.

        Returns the scene dir to train, or ``None`` if the window closed. The panel
        owns *when* (every button's ``enabled`` is a ``can_*`` property); this owns
        *how*, and every branch here publishes a state transition so the UI reacts
        on the same frame the request lands.
        """
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
                    # "Export first" rescues the result; it does NOT then destroy it
                    # by continuing with the load. The drop is forgotten and the
                    # user is told, rather than being applied behind an export that
                    # might itself fail.
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
        """A folder was dropped. Acknowledge it BEFORE reading anything.

        Replacing a live reconstruction is consequential, so an unexported result
        asks first — and offers to export instead of losing it. Anything else loads
        straight away: a confirmation for a harmless action is just a click.
        """
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
        """Build and train ONE run on ``scene_dir``, feeding the running viewer.

        Heavy GPU setup runs with the render thread suspended to avoid queue races.
        The run is also SURVIVABLE: a failure leaves the hub usable (error phase,
        commands served again) instead of taking the process down with it, because
        the end of a run is no longer the end of the session.
        """
        state = self.app._viewer.state
        print(f"[hub] training on {scene_dir}")
        # Acknowledge the click before the slowest part of a short run: the scene
        # load, pipeline build and texture upload all happen before iteration 1.
        state.reset_for_new_run()
        state.training_initializing(self.app.args.iters)
        state.begin_activity("Loading scene and building the pipeline")
        self.app.viewer.clear_stop_request()
        state.log(f"preparing training on {scene_dir}")
        self.app.viewer.suspend_rendering()
        try:
            self.app._begin_repeat_run()
            # The panel's settings become the run's args HERE — before anything
            # reads them — and the viewer's own pipeline is re-shaped to match.
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
        # The pipeline stays alive and on screen: that is what Export writes out,
        # and what the viewer keeps showing while the user decides what is next.
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
        """Write the trained gaussians to ``<run>/output.ply`` from the panel.

        Same file the automatic post-training export writes under
        ``--compute-final-metrics``; this is the on-demand path, so a run made
        without that flag is not lost. The absolute path goes to the log, since the
        panel has no file dialog to show one.
        """
        from trainer.backend.export import save_ply

        state = self.app._viewer.state
        if self.app.pipeline is None:
            state.result_failed("Nothing to export yet - train a scene first.")
            state.notify("Nothing to export", WARN)
            return
        path = self.app.paths.root / (state.export_name or "output.ply")
        state.begin_activity(f"Exporting {path.name}")
        state.log(f"exporting {path.name} …")
        # save_ply READS BACK the gaussian buffers, so the render thread has to be
        # off the queue (and off the GIL) for it — see suspend_rendering.
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

        Windows-only on purpose: ``os.startfile`` is the one platform-appropriate
        call available without adding a dependency, and a button that silently does
        nothing elsewhere would be worse than not offering it.
        """
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

        The 'work on something else' path: the window, the device and the installed
        tools stay, everything a run allocated goes, and the panel is back to
        "drag a dataset folder onto the window". The log survives — it is the record
        of what just happened, and losing it is worse than a busy panel.
        """
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
        """Fold the panel's TRAINING SETUP into this run's args.

        Called with rendering suspended, immediately before anything reads the
        args, so every option the panel offers takes effect for real. The viewer's
        own pipeline is rebuilt here too: its splat capacity / SH buffers / Mip
        raster path are fixed at construction, so a changed model needs a new one
        (no-op when the user left those alone).
        """
        from trainer.frontend.settings import viewer_stream_capacity

        state = self.app._viewer.state
        for change in setup.apply_to(self.app.args):
            state.log(f"[setup] {change}")
        self.app.viewer.reconfigure_model(
            max_splats=viewer_stream_capacity(self.app.args.max_splats,
                                              self.app.args.stream_max_splats),
            sh_degree=self.app.args.sh_degree,
            mip_enabled=self.app.args.mip_splatting,
            mip_kernel_size=(self.app.args.mip_kernel_size
                             if self.app.args.mip_splatting else 0.3),
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
        """Put the dataset's SfM point cloud on screen before Train is pressed.

        The same points training initializes from, shown as gaussians in the
        viewer's own pipeline — so a bad reconstruction (wrong scale, split into
        two halves, half the views dropped) is visible BEFORE paying for a run,
        which is the whole point of doing it here rather than after Train.

        Best-effort by design: a scene the loader rejects must leave the hub usable
        (drop another folder, reconstruct again), not kill the process. Train will
        surface the same failure with a real traceback if the user insists.
        """
        state = self.app._viewer.state
        state.dataset_loading("Reading the COLMAP model")
        state.log("loading the SfM point cloud …")
        started = time.time()
        # Rendering is suspended for the WHOLE thing, load included — not just the
        # buffer upload. ``load_colmap``'s binary parsers are per-record Python
        # loops, and the render thread holds the GIL for most of every frame (it
        # blocks in wait_for_submit and re-evaluates every panel binding), so the
        # same 0.6s parse takes over 40s next to a live viewer. This is the CPU
        # sibling of the queue-race reason the training setup path suspends.
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

        # EVERY camera here, not the eval split _attach_scene_to_viewer installs
        # once training starts: before a run you want to fly the whole capture and
        # see which views the reconstruction actually placed.
        self.app.viewer.attach_scene(
            [(entry.image_name, entry.camera) for entry in dataset.images],
            snap=True)
        # What was detected, in numbers: the panel says "148 images · 1920x1080 ·
        # 1.7M points" instead of leaving the user to trust the word "colmap".
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
