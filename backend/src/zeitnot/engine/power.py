"""Opt processes out of Windows power throttling (EcoQoS).

Windows 11 treats processes it considers background work (for example those started
by Task Scheduler) as low priority: on hybrid CPUs it confines them to the efficiency
cores. Twelve engines sharing four E-cores each get about a third of a core. A
process can opt out by setting its power-throttling state to "execution speed not
throttled" (``SetProcessInformation`` with ``ProcessPowerThrottling``); no
administrator rights are needed for our own child processes.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

log = logging.getLogger(__name__)

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLING_CURRENT_VERSION = 1
_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_SET_INFORMATION = 0x0200


class _ThrottlingState(ctypes.Structure):
    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def disable_power_throttling(pid: int | None = None) -> bool:
    """Ask Windows not to throttle ``pid`` (default: this process). False if not applied."""
    if sys.platform != "win32":
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Handles are pointer-sized; without declared signatures ctypes would pass 32-bit ints.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel32.SetProcessInformation.restype = ctypes.c_bool
    kernel32.SetProcessInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    own = pid is None or pid == os.getpid()
    handle = (
        kernel32.GetCurrentProcess()
        if own
        else kernel32.OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
    )
    if not handle:
        log.warning("cannot open process %s to disable power throttling", pid)
        return False
    try:
        # Control the execution-speed policy (ControlMask) and switch it off (StateMask 0).
        state = _ThrottlingState(_THROTTLING_CURRENT_VERSION, _THROTTLING_EXECUTION_SPEED, 0)
        ok = bool(
            kernel32.SetProcessInformation(
                handle, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)
            )
        )
        if not ok:
            log.warning("SetProcessInformation failed (error %s)", ctypes.get_last_error())
        return ok
    finally:
        if not own:
            kernel32.CloseHandle(handle)
