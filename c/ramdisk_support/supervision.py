"""Cgroup-v2 containment and gated exec for managed engines.

The persisted authority is a canonical path relative to one delegated cgroup
root plus the leaf directory's device/inode identity.  A missing or replaced
leaf is never interpreted as process absence.  New engines execute through a
pipe gate: the unchanged child PID is attached and pidfd-verified before the
parent releases it into ``execvpe``.

The delegated root is an exclusive cooperative Colibri resource. Linux has no
inode-conditional directory-removal syscall, so the durable lifecycle lock
serializes every supported writer and out-of-band mutation of that root is
outside the containment contract.
"""

from __future__ import print_function

import errno
import hashlib
import math
import os
import select
import signal
import subprocess
import sys
import time

from .common import RamdiskError, _positive_int


CONTAINMENT_VERSION = 1
CONTAINMENT_MODE = "cgroup-v2"
DEFAULT_CGROUP_ROOT = "/sys/fs/cgroup"
_RELEASE_BYTE = b"G"
_GATE_ABORT_EXIT = 125
_GATE_PROGRAM = r"""
import os
import sys

descriptor = int(sys.argv[1])
command = sys.argv[2:]
try:
    token = os.read(descriptor, 2)
finally:
    os.close(descriptor)
if token != b"G" or not command:
    raise SystemExit(125)
os.execvpe(command[0], command, dict(os.environ))
"""


class ContainmentInconclusive(RamdiskError):
    """Kernel containment state could not authorize an irreversible action."""


class ContainmentMissing(ContainmentInconclusive):
    """A persisted containment pathname no longer exists."""


class ExecGate:
    """Parent-side handle for a child blocked before engine exec."""

    __slots__ = ("process", "release_fd", "released", "pidfd")

    def __init__(self, process, release_fd):
        self.process = process
        self.release_fd = release_fd
        self.released = False
        self.pidfd = None


def _positive_identity(value, name):
    if not _positive_int(value):
        raise RamdiskError("cgroup containment has invalid %s" % name)
    return int(value)


def validate_containment(value):
    """Validate and return the exact durable containment identity."""
    if not isinstance(value, dict):
        raise RamdiskError("managed process containment is missing")
    required = {"version", "mode", "relative_path", "device", "inode"}
    if set(value) != required:
        raise RamdiskError("managed process containment is incomplete")
    if (
        not isinstance(value.get("version"), int)
        or isinstance(value.get("version"), bool)
        or value.get("version") != CONTAINMENT_VERSION
    ):
        raise RamdiskError("managed process containment version is unsupported")
    if value.get("mode") != CONTAINMENT_MODE:
        raise RamdiskError("managed process containment mode is unsupported")
    relative = value.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or os.path.isabs(relative)
        or any(part in ("", ".", "..") for part in relative.split("/"))
        or os.path.normpath(relative) != relative
    ):
        raise RamdiskError("managed process containment path is unsafe")
    return {
        "version": CONTAINMENT_VERSION,
        "mode": CONTAINMENT_MODE,
        "relative_path": relative,
        "device": _positive_identity(value.get("device"), "device"),
        "inode": _positive_identity(value.get("inode"), "inode"),
    }


def _pidfd_exited(descriptor):
    poller = select.poll()
    poller.register(descriptor, select.POLLIN | select.POLLHUP | select.POLLERR)
    return bool(poller.poll(0))


def _pidfd_send(descriptor, signum):
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(sender):
        raise RamdiskError("cgroup supervision requires pidfd_send_signal")
    sender(descriptor, signum, None, 0)


def _open_flags():
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _file_flags(write=False):
    flags = os.O_WRONLY if write else os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _slug(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]


def spawn_exec_gate(command, *, environment, popen=None, **kwargs):
    """Spawn a child that cannot exec ``command`` until explicitly released."""
    if (
        not isinstance(command, (list, tuple))
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise RamdiskError("managed exec gate requires a non-empty command")
    if not isinstance(environment, dict):
        raise RamdiskError("managed exec gate requires an explicit environment")
    if os.name != "posix" or not hasattr(os, "pipe"):
        raise RamdiskError("managed exec gate requires POSIX descriptor passing")
    popen = subprocess.Popen if popen is None else popen
    read_fd, write_fd = os.pipe()
    try:
        gate_command = [
            sys.executable,
            "-c",
            _GATE_PROGRAM,
            str(read_fd),
        ] + list(command)
        options = dict(kwargs)
        options.update(
            env=dict(environment),
            close_fds=True,
            pass_fds=(read_fd,),
            start_new_session=True,
        )
        process = popen(gate_command, **options)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(read_fd)
    return ExecGate(process, write_fd)


def release_exec_gate(gate):
    """Release a verified gate exactly once."""
    if not isinstance(gate, ExecGate) and not hasattr(gate, "release_fd"):
        raise RamdiskError("managed exec gate handle is invalid")
    if getattr(gate, "released", False):
        return
    descriptor = gate.release_fd
    if descriptor is None:
        raise RamdiskError("managed exec gate release descriptor is closed")
    try:
        written = os.write(descriptor, _RELEASE_BYTE)
        if written != 1:
            raise RamdiskError("managed exec gate release was incomplete")
    finally:
        os.close(descriptor)
        gate.release_fd = None
    gate.released = True


def abort_exec_gate(gate, timeout=2.0):
    """Close an unreleased gate so the child exits without engine exec."""
    if gate is None:
        return
    descriptor = getattr(gate, "release_fd", None)
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass
        gate.release_fd = None
    process = getattr(gate, "process", None)
    if process is None or getattr(gate, "released", False):
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


class CgroupSupervisor:
    """Kernel containment operations rooted in one delegated cgroup subtree."""

    def __init__(
        self,
        root=None,
        *,
        pidfd_open=None,
        pidfd_send_signal=None,
        pidfd_exited=None,
        close_fd=None,
        sleep=None,
    ):
        configured_root = (
            root
            or os.environ.get("COLI_CGROUP_DELEGATED_ROOT")
            or self.discover_root()
        )
        if not isinstance(configured_root, str) or not os.path.isabs(
            configured_root
        ):
            raise RamdiskError(
                "cgroup delegated root must be an absolute path"
            )
        self.root = os.path.normpath(configured_root)
        self._pidfd_open = pidfd_open or getattr(os, "pidfd_open", None)
        self._pidfd_send_signal = pidfd_send_signal or _pidfd_send
        self._pidfd_exited = pidfd_exited or _pidfd_exited
        self._close_fd = close_fd or os.close
        self._sleep = sleep or time.sleep

    @staticmethod
    def discover_root():
        """Resolve the current process's delegated cgroup-v2 directory."""
        relative = None
        try:
            with open("/proc/self/cgroup", "r", encoding="utf-8") as stream:
                for line in stream:
                    hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
                    if hierarchy == "0" and controllers == "":
                        relative = path.lstrip("/")
                        break
        except (OSError, ValueError) as error:
            raise RamdiskError(
                "cannot discover the current cgroup-v2 delegation: %s" % error
            ) from error
        if relative is None:
            raise RamdiskError(
                "cannot discover the current cgroup-v2 delegation"
            )
        return os.path.join(DEFAULT_CGROUP_ROOT, relative)

    def _root_fd(self):
        if any(
            not isinstance(getattr(os, name, None), int)
            or getattr(os, name) == 0
            for name in ("O_DIRECTORY", "O_NOFOLLOW")
        ):
            raise RamdiskError(
                "cgroup supervision requires no-follow directory opens"
            )
        descriptor = None
        try:
            descriptor = os.open(os.path.sep, _open_flags())
            for part in self.root.split(os.path.sep):
                if not part:
                    continue
                child = os.open(part, _open_flags(), dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise RamdiskError(
                "cannot open delegated cgroup-v2 root without symbolic "
                "links %s: %s"
                % (self.root, error)
            )
        try:
            controller = os.open(
                "cgroup.controllers",
                _file_flags(),
                dir_fd=descriptor,
            )
        except OSError as error:
            os.close(descriptor)
            raise RamdiskError(
                "delegated root is not a usable cgroup-v2 hierarchy: %s"
                % error
            )
        os.close(controller)
        return descriptor

    @staticmethod
    def _open_component(parent, name, *, create=False, must_create=False):
        if create:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent)
            except FileExistsError as error:
                if must_create:
                    raise RamdiskError(
                        "managed cgroup operation already exists"
                    ) from error
        try:
            return os.open(name, _open_flags(), dir_fd=parent)
        except FileNotFoundError as error:
            raise ContainmentMissing(
                "cgroup containment component %s is missing" % name
            ) from error
        except OSError as error:
            raise RamdiskError(
                "cannot reopen cgroup component %s without symbolic links: %s"
                % (name, error)
            )

    @staticmethod
    def relative_leaf(deployment_id, operation_id):
        return "colibri/d%s/o%s" % (
            _slug(deployment_id),
            _slug(operation_id),
        )

    def create_leaf(self, deployment_id, operation_id):
        """Create one unique engine leaf and return its durable identity."""
        relative = self.relative_leaf(deployment_id, operation_id)
        current = self._root_fd()
        try:
            parts = relative.split("/")
            for index, part in enumerate(parts):
                child = self._open_component(
                    current,
                    part,
                    create=True,
                    must_create=index == len(parts) - 1,
                )
                os.close(current)
                current = child
            info = os.fstat(current)
            return {
                "version": CONTAINMENT_VERSION,
                "mode": CONTAINMENT_MODE,
                "relative_path": relative,
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
            }
        finally:
            os.close(current)

    def reconcile_leaf_intent(self, deployment_id, operation_id):
        """Observe the exact leaf authorized by a durable create intent."""
        relative = self.relative_leaf(deployment_id, operation_id)
        current = self._root_fd()
        try:
            for part in relative.split("/"):
                try:
                    child = self._open_component(current, part)
                except ContainmentMissing:
                    return None
                os.close(current)
                current = child
            info = os.fstat(current)
            return {
                "version": CONTAINMENT_VERSION,
                "mode": CONTAINMENT_MODE,
                "relative_path": relative,
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
            }
        finally:
            os.close(current)

    def reopen_verified(self, containment):
        """Reopen a leaf with no-follow traversal and verify device/inode."""
        expected = validate_containment(containment)
        current = self._root_fd()
        try:
            for part in expected["relative_path"].split("/"):
                child = self._open_component(current, part)
                os.close(current)
                current = child
            info = os.fstat(current)
            if (
                int(info.st_dev) != expected["device"]
                or int(info.st_ino) != expected["inode"]
            ):
                raise RamdiskError("cgroup containment identity changed")
            return current
        except BaseException:
            os.close(current)
            raise

    @staticmethod
    def _read_at(directory_fd, name):
        try:
            descriptor = os.open(name, _file_flags(), dir_fd=directory_fd)
        except FileNotFoundError as error:
            raise ContainmentInconclusive(
                "cgroup containment file %s is missing" % name
            ) from error
        except OSError as error:
            raise ContainmentInconclusive(
                "cannot read cgroup containment file %s: %s" % (name, error)
            ) from error
        try:
            chunks = []
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if sum(len(item) for item in chunks) > 1 << 20:
                    raise ContainmentInconclusive(
                        "cgroup containment file %s is too large" % name
                    )
            return b"".join(chunks).decode("ascii")
        except (OSError, UnicodeDecodeError) as error:
            raise ContainmentInconclusive(
                "cannot parse cgroup containment file %s: %s" % (name, error)
            ) from error
        finally:
            os.close(descriptor)

    @staticmethod
    def _parse_members(text):
        result = []
        for line in text.splitlines():
            token = line.strip()
            if not token:
                continue
            try:
                pid = int(token)
            except ValueError as error:
                raise ContainmentInconclusive(
                    "cgroup.procs contains a non-numeric PID"
                ) from error
            if not _positive_int(pid):
                raise ContainmentInconclusive(
                    "cgroup.procs contains an invalid PID"
                )
            result.append(pid)
        # cgroup.procs is a kernel snapshot, not a sorted userspace ledger.
        # Ordering is unspecified and a concurrently migrating task can be
        # repeated, so stability is compared on the canonical TGID set.
        return sorted(set(result))

    def members(self, containment):
        """Return a stable member snapshot; instability is inconclusive."""
        descriptor = self.reopen_verified(containment)
        try:
            first = self._parse_members(self._read_at(descriptor, "cgroup.procs"))
            second = self._parse_members(self._read_at(descriptor, "cgroup.procs"))
        finally:
            os.close(descriptor)
        if first != second:
            raise ContainmentInconclusive(
                "cgroup membership changed during verification"
            )
        return second

    def _write_member(self, containment, pid):
        descriptor = self.reopen_verified(containment)
        try:
            try:
                target = os.open(
                    "cgroup.procs",
                    _file_flags(write=True),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise RamdiskError("cannot attach PID to cgroup: %s" % error)
            try:
                payload = ("%d\n" % int(pid)).encode("ascii")
                written = os.write(target, payload)
                if written != len(payload):
                    raise RamdiskError("cannot attach PID to cgroup: short write")
            finally:
                os.close(target)
        finally:
            os.close(descriptor)

    def _open_bound_pidfd(self, pid):
        if not callable(self._pidfd_open):
            raise RamdiskError("cgroup supervision requires pidfd_open")
        try:
            descriptor = self._pidfd_open(int(pid), 0)
        except (OSError, ValueError) as error:
            raise ContainmentInconclusive(
                "cannot bind cgroup member PID %s with pidfd: %s" % (pid, error)
            ) from error
        if self._pidfd_exited(descriptor):
            self._close_fd(descriptor)
            raise ContainmentInconclusive(
                "cgroup member PID %s pidfd is already exited" % pid
            )
        return descriptor

    def spawn_gate(self, command, *, environment, **kwargs):
        return spawn_exec_gate(
            command,
            environment=environment,
            **kwargs,
        )

    def abort_gate(self, gate):
        abort_exec_gate(gate)
        self.close_gate(gate)

    def close_gate(self, gate):
        descriptor = getattr(gate, "pidfd", None)
        if descriptor is not None:
            self._close_fd(descriptor)
            gate.pidfd = None

    def attach_gate(self, gate, containment):
        """Attach the unchanged gate PID and retain its verified pidfd."""
        pid = getattr(getattr(gate, "process", None), "pid", None)
        if not _positive_int(pid):
            raise RamdiskError("managed exec gate has no exact child PID")
        if getattr(gate, "pidfd", None) is not None:
            raise RamdiskError("managed exec gate is already pidfd-bound")
        self._write_member(containment, pid)
        descriptor = self._open_bound_pidfd(pid)
        try:
            if self.members(containment) != [pid]:
                raise ContainmentInconclusive(
                    "managed exec gate is not the sole cgroup member"
                )
            if self._pidfd_exited(descriptor):
                raise ContainmentInconclusive(
                    "managed exec gate pidfd exited before release"
                )
            gate.pidfd = descriptor
            descriptor = None
        finally:
            if descriptor is not None:
                self._close_fd(descriptor)

    def release_gate(self, gate, containment):
        """Revalidate the attached pidfd/member pair, then release exec."""
        pid = getattr(getattr(gate, "process", None), "pid", None)
        descriptor = getattr(gate, "pidfd", None)
        if not _positive_int(pid) or descriptor is None:
            raise RamdiskError("managed exec gate is not pidfd-bound")
        if self.members(containment) != [int(pid)]:
            raise ContainmentInconclusive(
                "managed exec gate is not the sole cgroup member"
            )
        if self._pidfd_exited(descriptor):
            raise ContainmentInconclusive(
                "managed exec gate pidfd exited before release"
            )
        release_exec_gate(gate)

    def verify_gate(self, gate, containment):
        """Revalidate the retained pidfd and membership before promotion."""
        pid = getattr(getattr(gate, "process", None), "pid", None)
        descriptor = getattr(gate, "pidfd", None)
        if not _positive_int(pid) or descriptor is None:
            raise RamdiskError("managed exec gate is not pidfd-bound")
        if self._pidfd_exited(descriptor):
            raise ContainmentInconclusive(
                "managed exec gate pidfd exited before promotion"
            )
        if int(pid) not in self.members(containment):
            raise ContainmentInconclusive(
                "managed exec gate left its persisted cgroup"
            )
        if self._pidfd_exited(descriptor):
            raise ContainmentInconclusive(
                "managed exec gate pidfd exited during promotion"
            )
        return True

    def attach_and_release(self, gate, containment):
        """Compatibility wrapper for callers without durable phase writes."""
        self.attach_gate(gate, containment)
        self.release_gate(gate, containment)

    def signal_pass(self, containment, signum):
        """Open/revalidate an entire membership batch before any signal."""
        for _attempt in range(3):
            before = self.members(containment)
            if not before:
                return 0
            descriptors = []
            try:
                for pid in before:
                    descriptors.append((pid, self._open_bound_pidfd(pid)))
                after = self.members(containment)
                if after != before:
                    continue
                for _pid, descriptor in descriptors:
                    if self._pidfd_exited(descriptor):
                        raise ContainmentInconclusive(
                            "cgroup member exited before signal batch"
                        )
                for _pid, descriptor in descriptors:
                    try:
                        self._pidfd_send_signal(descriptor, signum)
                    except ProcessLookupError:
                        continue
                    except OSError as error:
                        raise ContainmentInconclusive(
                            "pidfd signal failed: %s" % error
                        ) from error
                return len(descriptors)
            finally:
                for _pid, descriptor in descriptors:
                    self._close_fd(descriptor)
        raise ContainmentInconclusive(
            "cgroup membership never stabilized for signaling"
        )

    def _events(self, containment):
        descriptor = self.reopen_verified(containment)
        try:
            text = self._read_at(descriptor, "cgroup.events")
        finally:
            os.close(descriptor)
        result = {}
        for line in text.splitlines():
            fields = line.split()
            if len(fields) != 2:
                raise ContainmentInconclusive("cgroup.events is malformed")
            try:
                result[fields[0]] = int(fields[1])
            except ValueError as error:
                raise ContainmentInconclusive(
                    "cgroup.events is malformed"
                ) from error
        if result.get("populated") not in (0, 1):
            raise ContainmentInconclusive(
                "cgroup.events has no valid populated state"
            )
        return result

    def prove_absence(self, containment):
        """Prove stable ``populated=0`` plus empty membership."""
        before = self._events(containment)
        members = self.members(containment)
        after = self._events(containment)
        if before.get("populated") != 0 or after.get("populated") != 0:
            raise ContainmentInconclusive("cgroup is still populated")
        if members:
            raise ContainmentInconclusive(
                "cgroup populated state contradicts live membership"
            )
        return True

    def prove_removed(self, containment):
        """Prove the exact persisted leaf pathname is currently absent."""
        try:
            descriptor = self.reopen_verified(containment)
        except ContainmentMissing:
            return True
        except RamdiskError:
            # A replacement is not absence and is outside this authority.
            raise
        else:
            os.close(descriptor)
            raise ContainmentInconclusive(
                "cgroup containment remains present after removal"
            )

    def terminate(
        self,
        containment,
        *,
        term_seconds=10.0,
        kill_seconds=3.0,
    ):
        """Repeatedly enumerate and pidfd-signal TERM then KILL."""
        for signum, duration, interval in (
            (signal.SIGTERM, term_seconds, 0.1),
            (signal.SIGKILL, kill_seconds, 0.05),
        ):
            passes = max(1, int(math.ceil(max(0.0, float(duration)) / interval)))
            for _attempt in range(passes):
                try:
                    if self.prove_absence(containment):
                        return {"status": "absent", "signals": signum}
                except ContainmentInconclusive:
                    pass
                try:
                    self.signal_pass(containment, signum)
                except ContainmentInconclusive:
                    pass
                self._sleep(interval)
        if self.prove_absence(containment):
            return {"status": "absent", "signals": signal.SIGKILL}
        raise ContainmentInconclusive(
            "cgroup remained populated after SIGKILL"
        )

    def verify_record(self, record):
        """Verify one promoted leader against stable cgroup membership/pidfd."""
        try:
            if not isinstance(record, dict) or not _positive_int(record.get("pid")):
                raise RamdiskError("managed process record has no exact PID")
            containment = validate_containment(record.get("containment"))
            pid = int(record["pid"])
            if pid not in self.members(containment):
                raise ContainmentInconclusive(
                    "managed leader is not in its persisted cgroup"
                )
            descriptor = self._open_bound_pidfd(pid)
            try:
                if pid not in self.members(containment):
                    raise ContainmentInconclusive(
                        "managed leader changed containment during pidfd bind"
                    )
                if self._pidfd_exited(descriptor):
                    raise ContainmentInconclusive("managed leader pidfd exited")
            finally:
                self._close_fd(descriptor)
            return True, None
        except (RamdiskError, OSError) as error:
            return False, str(error)

    def liveness(self, record):
        """Return True/False, or None when contained liveness is inconclusive."""
        try:
            containment = validate_containment(record.get("containment"))
            if self.prove_absence(containment):
                return False
        except ContainmentInconclusive:
            try:
                members = self.members(record.get("containment"))
                return True if members else None
            except (RamdiskError, OSError):
                return None
        except (RamdiskError, OSError):
            return None
        return False

    def remove_empty(self, containment):
        """Remove a verified empty leaf; missing/replacement stays inconclusive."""
        self.prove_absence(containment)
        expected = validate_containment(containment)
        parts = expected["relative_path"].split("/")
        parent = self._root_fd()
        try:
            for part in parts[:-1]:
                child = self._open_component(parent, part)
                os.close(parent)
                parent = child
            leaf = self._open_component(parent, parts[-1])
            try:
                info = os.fstat(leaf)
                if (
                    int(info.st_dev) != expected["device"]
                    or int(info.st_ino) != expected["inode"]
                ):
                    raise RamdiskError(
                        "cgroup containment identity changed before removal"
                    )
            finally:
                os.close(leaf)
            try:
                os.rmdir(parts[-1], dir_fd=parent)
            except FileNotFoundError as error:
                raise ContainmentMissing(
                    "cgroup disappeared before verified removal"
                ) from error
            except OSError as error:
                raise ContainmentInconclusive(
                    "cannot remove empty cgroup: %s" % error
                ) from error
        finally:
            os.close(parent)
        return True


def default_supervisor():
    return CgroupSupervisor()
