"""Windows process containment; importing this module is safe on Linux."""
import ctypes
from ctypes import wintypes
import json
import subprocess
import time

from .installation import external_process_context


class _BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IOCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in
                ('TotalUserTime', 'TotalKernelTime', 'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')]
    _fields_ += [(name, wintypes.DWORD) for name in
                 ('TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]


# The external interpreter cannot create the application/engine until the
# parent assigns it to its Job. EOF before the gate means cancellation.
_BOOTSTRAP = """import json, subprocess, sys
line = sys.stdin.buffer.readline()
if not line:
    sys.exit(125)
argv = json.loads(line)
child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
sys.exit(child.wait())
"""


class WindowsJob:
    def __init__(self):
        self._api = ctypes.WinDLL('kernel32', use_last_error=True)
        api = self._api
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.c_void_p]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        self._handle = api.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def spawn(self, spec, cancel, lifecycle_lock):
        process = None
        try:
            with external_process_context():
                process = subprocess.Popen(
                    [spec.argv[0], '-I', '-S', '-u', '-c', _BOOTSTRAP],
                    cwd=spec.cwd, env=spec.env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                )
            if not self._api.AssignProcessToJobObject(self._handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            payload = (json.dumps(spec.argv) + '\n').encode('utf-8')
            # Do not hold the lifecycle lock during interpreter creation,
            # external DLL locking, or payload preparation. Only the final
            # authorization/gate release is atomic with Supervisor.stop().
            with lifecycle_lock:
                if not cancel.is_set():
                    process.stdin.write(payload)
                    process.stdin.flush()
            process.stdin.close()
            return process
        except BaseException:
            if process is not None:
                process.kill()
                process.wait()
                process.stdin.close()
                process.stdout.close()
            raise

    def active(self):
        info = _Accounting()
        if not self._api.QueryInformationJobObject(self._handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.ActiveProcesses

    def terminate(self, timeout=3.0):
        # Hidden applications have no console for CTRL_BREAK. Terminate the
        # owned job, then verify every descendant has actually exited.
        if not self._api.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + timeout
        while self.active():
            if time.monotonic() >= deadline:
                raise OSError('Windows has not confirmed that the model processes exited')
            time.sleep(0.02)

    def close(self):
        if self._handle:
            self._api.CloseHandle(self._handle)
            self._handle = None
