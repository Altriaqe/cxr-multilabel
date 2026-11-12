"""Windows 进程优先级调整。

本机有一个常驻的 Chrome 进程会长期占满 ~15 个逻辑核（实测 1500%+ CPU），
把训练进程压到普通优先级时会被严重饿死，吞吐波动可达 30 倍。
把训练进程提到 HIGH_PRIORITY_CLASS 后能稳定抢到调度。
"""

from __future__ import annotations

import os
import sys

HIGH_PRIORITY_CLASS = 0x00000080
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
NORMAL_PRIORITY_CLASS = 0x00000020


def set_high_priority() -> bool:
    """把当前进程提到高优先级。非 Windows 或失败时返回 False。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k32.SetPriorityClass.restype = ctypes.c_bool
        handle = k32.GetCurrentProcess()
        ok = k32.SetPriorityClass(handle, HIGH_PRIORITY_CLASS)
        return bool(ok)
    except Exception:
        return False


def current_priority() -> int:
    try:
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        k32.GetPriorityClass.restype = ctypes.c_uint32
        return int(k32.GetPriorityClass(k32.GetCurrentProcess()))
    except Exception:
        return -1


def limit_process_cpu(percent: int) -> bool:
    """用 Job Object 给当前进程设 CPU 速率上限（0-100）。一般用不到。"""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return False
        # JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
        class CpuRate(ctypes.Structure):
            _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]

        ENABLE = 0x1
        HARD_CAP = 0x4
        info = CpuRate(ENABLE | HARD_CAP, percent * 100)
        JobObjectCpuRateControlInformation = 15
        ok = k32.SetInformationJobObject(job, JobObjectCpuRateControlInformation,
                                         ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            return False
        return bool(k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()))
    except Exception:
        return False
