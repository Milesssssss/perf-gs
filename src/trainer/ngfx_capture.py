"""Programmatic Nsight Graphics capture trigger.

Wraps the Nsight Graphics SDK's NGFX_Injection C API so we can:
  1) Self-inject Nsight into THIS process before the Vulkan driver loads.
  2) Trigger a capture at a user-chosen iteration via a single call.

Only self-injection supports programmatic triggering: ExecuteActivityCommand
requires the SDK-owned injection state that InjectToProcess sets up. A
process launched from Nsight's UI has a different injection mechanism and
ExecuteActivityCommand there returns -2 INVALID_ARGUMENT (verified against
Nsight Graphics 2025.2.0).

Constraints (see NGFX_Injection.h):
- ``inject()`` MUST run BEFORE any Vulkan driver load (i.e. before
  ``create_device``). Otherwise ``NGFX_INJECTION_RESULT_DRIVER_STILL_LOADED``.
- For GPU Trace: a saved ``.ngfx-proj`` file with metrics + output dir is
  REQUIRED. Without it the trigger succeeds silently and records nothing.
  Frame Debugger works without a project file.
- The DLL is bundled with the Nsight Graphics install, not the app —
  path is looked up at import time and the module is a no-op if not found.

Creating an .ngfx-proj file:
    1) Open Nsight Graphics UI standalone.
    2) File > New Project. Pick the activity (GPU Trace or Frame Debugger).
    3) Configure metric set + output directory.
    4) File > Save Project As... > save to e.g. nsight/gpu_trace.ngfx-proj
       inside this workspace.
    5) Pass its path via --capture-project.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# NGFX_Injection.h enums.
_ACTIVITY_FRAME_DEBUGGER = 1
_ACTIVITY_GPU_TRACE = 4

_ACTIVITY_NAME_TO_ENUM = {
    "trace": _ACTIVITY_GPU_TRACE,
    "frame": _ACTIVITY_FRAME_DEBUGGER,
}

_RESULT_OK = 0


class _InstallationInfo(ctypes.Structure):
    """Mirror of NGFX_Injection_InstallationInfo (24 bytes on x64:
    enum + 3*u16 + 2 pad + ptr)."""
    _fields_ = [
        ("sku", ctypes.c_int32),
        ("versionMajor", ctypes.c_uint16),
        ("versionMinor", ctypes.c_uint16),
        ("versionPatch", ctypes.c_uint16),
        ("_pad", ctypes.c_uint16),
        ("installationPath", ctypes.c_void_p),
    ]


class _Activity(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("_pad", ctypes.c_uint32),
        ("description", ctypes.c_char_p),
    ]


def _default_dll_path() -> Path | None:
    """Locate NGFX_Injection.dll from a standard Nsight Graphics install.
    Users can override with the NGFX_INJECTION_DLL environment variable."""
    override = os.environ.get("NGFX_INJECTION_DLL")
    if override:
        p = Path(override)
        return p if p.is_file() else None

    if sys.platform != "win32":
        return None

    nvidia_root = Path(r"C:\Program Files\NVIDIA Corporation")
    if not nvidia_root.is_dir():
        return None
    candidates = sorted(
        nvidia_root.glob("Nsight Graphics *"),
        key=lambda p: p.name,
        reverse=True,
    )
    for install in candidates:
        dll = install / "SDKs" / "NsightGraphicsSDK" / "0.8.0" / "lib" / "x64" / "NGFX_Injection.dll"
        if dll.is_file():
            return dll
    return None


class NgfxCapture:
    """Handle for Nsight Graphics programmatic capture (self-injection).

    If the Nsight Graphics SDK isn't installed, methods no-op with a
    printed warning — training still runs normally, you just don't get
    captures."""

    def __init__(self, dll_path: str | Path | None = None):
        self._injected = False
        self._activity_enum: int | None = None
        self._dll: ctypes.CDLL | None = None
        self._installation: _InstallationInfo | None = None
        self._activity_struct: _Activity | None = None

        p = Path(dll_path) if dll_path else _default_dll_path()
        if p is None:
            return
        try:
            self._dll = ctypes.CDLL(str(p))
        except OSError as e:
            print(f"[ngfx] failed to load {p}: {e}")
            return

        d = self._dll
        d.NGFX_Injection_EnumerateInstallations.restype = ctypes.c_int32
        d.NGFX_Injection_EnumerateInstallations.argtypes = [
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(_InstallationInfo)
        ]
        d.NGFX_Injection_EnumerateActivities.restype = ctypes.c_int32
        d.NGFX_Injection_EnumerateActivities.argtypes = [
            ctypes.POINTER(_InstallationInfo),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_Activity),
        ]
        d.NGFX_Injection_InjectToProcess.restype = ctypes.c_int32
        d.NGFX_Injection_InjectToProcess.argtypes = [
            ctypes.POINTER(_InstallationInfo),
            ctypes.POINTER(_Activity),
        ]
        d.NGFX_Injection_SetActivitySettingsFromProjectFile.restype = ctypes.c_int32
        # Second arg is NGFX_Injection_PathChar* = wchar_t* on Windows.
        d.NGFX_Injection_SetActivitySettingsFromProjectFile.argtypes = [
            ctypes.POINTER(_InstallationInfo),
            ctypes.c_wchar_p,
        ]
        d.NGFX_Injection_ExecuteActivityCommand.restype = ctypes.c_int32
        d.NGFX_Injection_ExecuteActivityCommand.argtypes = []

    @property
    def available(self) -> bool:
        return self._dll is not None

    def inject(self, activity: str = "trace",
               project_path: str | Path | None = None) -> bool:
        """Perform the pre-driver injection. MUST run before any Vulkan
        device is created. Returns True on success.

        activity: 'trace' (GPU Trace with HW counters) or 'frame'
                  (Frame Debugger).
        project_path: Path to an ``.ngfx-proj`` file saved from the Nsight
                  Graphics UI. REQUIRED for the 'trace' activity to record
                  anything useful; optional for 'frame'."""
        if self._dll is None:
            print("[ngfx] NGFX_Injection.dll not found; capture disabled.")
            return False
        if activity not in _ACTIVITY_NAME_TO_ENUM:
            print(f"[ngfx] unknown activity {activity!r}, expected one of "
                  f"{list(_ACTIVITY_NAME_TO_ENUM)}. capture disabled.")
            return False
        self._activity_enum = _ACTIVITY_NAME_TO_ENUM[activity]

        d = self._dll

        # 1) Pick the first Nsight installation.
        count = ctypes.c_uint32(0)
        r = d.NGFX_Injection_EnumerateInstallations(ctypes.byref(count), None)
        if r != _RESULT_OK or count.value == 0:
            print(f"[ngfx] no Nsight installations found (result={r}, count={count.value})")
            return False
        installs = (_InstallationInfo * count.value)()
        r = d.NGFX_Injection_EnumerateInstallations(ctypes.byref(count), installs)
        if r != _RESULT_OK:
            print(f"[ngfx] EnumerateInstallations failed: {r}")
            return False
        self._installation = installs[0]

        # 2) Pick the requested activity.
        acount = ctypes.c_uint32(0)
        r = d.NGFX_Injection_EnumerateActivities(
            ctypes.byref(self._installation), ctypes.byref(acount), None)
        if r != _RESULT_OK or acount.value == 0:
            print(f"[ngfx] no activities available (result={r}, count={acount.value})")
            return False
        activities = (_Activity * acount.value)()
        r = d.NGFX_Injection_EnumerateActivities(
            ctypes.byref(self._installation), ctypes.byref(acount), activities)
        if r != _RESULT_OK:
            print(f"[ngfx] EnumerateActivities failed: {r}")
            return False

        chosen = None
        for i in range(acount.value):
            if activities[i].type == self._activity_enum:
                chosen = activities[i]
                break
        if chosen is None:
            print(f"[ngfx] activity {activity!r} (enum {self._activity_enum}) "
                  f"not available in this Nsight installation")
            return False
        # Keep the struct alive via self so InjectToProcess pointer stays valid.
        self._activity_struct = _Activity(type=chosen.type, description=chosen.description)

        # 3) Load the .ngfx-proj settings if provided. Required for GPU Trace
        # to record actual metrics; Frame Debugger works without.
        if project_path is not None:
            pp = Path(project_path).resolve()
            if not pp.is_file():
                print(f"[ngfx] project file not found: {pp} — continuing without")
            else:
                r = d.NGFX_Injection_SetActivitySettingsFromProjectFile(
                    ctypes.byref(self._installation),
                    ctypes.c_wchar_p(str(pp)),
                )
                if r != _RESULT_OK:
                    print(f"[ngfx] SetActivitySettingsFromProjectFile({pp}) "
                          f"failed: {r} — continuing")
                else:
                    print(f"[ngfx] loaded project settings from {pp}")
        elif activity == "trace":
            print("[ngfx] WARNING: --capture-activity trace without "
                  "--capture-project. GPU Trace will trigger but record "
                  "nothing. Save a .ngfx-proj from the Nsight UI and pass "
                  "its path as --capture-project.")

        # 4) Inject. Requires "no Vulkan driver loaded yet".
        r = d.NGFX_Injection_InjectToProcess(
            ctypes.byref(self._installation),
            ctypes.byref(self._activity_struct),
        )
        if r != _RESULT_OK:
            print(f"[ngfx] InjectToProcess failed: {r}. If this is -6, ensure "
                  f"NgfxCapture.inject() runs BEFORE create_device().")
            return False

        self._injected = True
        print(f"[ngfx] injected activity={activity!r}")
        return True

    def trigger(self) -> None:
        """Fire the capture command. Nsight captures on the next present /
        API-call boundary; for our loop that's the current iter's frame."""
        if not self._injected or self._dll is None:
            print("[ngfx] trigger() called but not injected")
            return
        r = self._dll.NGFX_Injection_ExecuteActivityCommand()
        if r == _RESULT_OK:
            print("[ngfx] ExecuteActivityCommand OK — capture should appear "
                  "in Nsight Graphics UI")
        else:
            print(f"[ngfx] ExecuteActivityCommand returned {r}")
