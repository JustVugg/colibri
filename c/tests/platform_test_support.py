"""Central, checked capability markers for cross-platform unittest discovery."""

import ast
import os
import signal
import sys
import unittest
from pathlib import Path


_LINUX_OPERATIONAL_TESTS = frozenset(
    {
        "test_ramdisk_benchmark.BenchmarkTest.test_acceptance_is_false_when_applicable_paths_fail",
        "test_ramdisk_benchmark.BenchmarkTest.test_best_runtime_knobs_are_saved_only_for_current_topology",
        "test_ramdisk_benchmark.BenchmarkTest.test_cuda_benchmark_generates_only_mmap_staged_variants",
        "test_ramdisk_benchmark.BenchmarkTest.test_full_per_node_benchmark_sizes_thread_sweep_to_target_node",
        "test_ramdisk_integration.RealTmpfsLifecycleTest.test_prepare_status_destroy_on_real_tmpfs",
        "test_ramdisk_model_planning.ScanAndPlanTest.test_protected_or_model_overlapping_mount_roots_are_blocked",
        "test_ramdisk_mounts.MountAndCopyTest.test_interrupted_mount_helper_retains_pending_recovery_without_unmount",
        "test_ramdisk_mounts.MountAndCopyTest.test_completed_mount_failure_retains_observed_mount_without_unmount",
        "test_ramdisk_mounts.MountAndCopyTest.test_failed_mount_helper_retains_pending_when_observation_is_inconclusive",
        "test_ramdisk_mounts.MountAndCopyTest.test_failed_pending_removal_save_retains_last_durable_pending_mount",
        "test_ramdisk_mounts.MountAndCopyTest.test_multi_mount_failure_preflights_all_before_any_unmount",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_cleanup_runs_even_when_error_manifest_cannot_be_saved",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_never_unmounts_identityless_successful_mount_by_path",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_persists_pending_ownership_before_mount_helper",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_promotes_only_the_exact_recorded_mount_identity",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_rollback_refuses_exact_mount_with_nested_child",
        "test_ramdisk_mounts.MountAndCopyTest.test_runner_oserror_retains_pending_without_absence_reconciliation",
        "test_ramdisk_presentation.TuiPlacementContractTest.test_cancelled_prepare_does_not_hide_rollback_failure",
        "test_ramdisk_presentation.TuiPlacementContractTest.test_clean_prepare_cancellation_removes_recovery_manifest",
        "test_ramdisk_processes.ManagedLaunchTest.test_exact_popen_attempt_is_reaped_across_registration_boundaries",
        "test_ramdisk_processes.ManagedLaunchTest.test_postfork_popen_exception_retains_and_reaps_exact_child",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_busy_mount_scan_includes_the_manager_process",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_dashboard_rss_sums_verified_wrapper_and_engine_group",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_destroy_persists_recovery_state_when_kernel_unmount_fails",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_destroy_preflights_every_busy_mount_before_unmounting",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_destroy_refuses_replaced_mount_identity",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_destroy_rejects_nested_child_mounts_before_any_unmount",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_destroy_retains_manifest_for_unrecorded_surviving_mount",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_manifest_rejects_missing_nonce_before_process_signaling",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_manifest_rejects_volatile_durable_state",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_stop_does_not_merge_when_retained_child_is_live",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_stop_persists_error_when_usage_merge_fails",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_stop_preserves_recoverable_error_for_incomplete_mount_layout",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_stop_preserves_termination_failure_until_group_absence_is_proven",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_stop_revalidates_identity_before_escalating_to_sigkill",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_stop_validates_every_pid_before_signaling_any",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_verified_stop_reaps_a_locally_owned_zombie_before_escalation",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_verified_termination_treats_retained_live_child_as_independent_evidence",
    }
)

_SIGTERM_HANDLER_TESTS = frozenset(
    {
        "test_ramdisk_cli_module.CliModuleTest.test_cli_termination_guard_restores_the_previous_handler",
        "test_ramdisk_cli_smoke.CliJsonSmokeTest.test_cli_sigterm_defers_until_stop_transaction_finishes",
        "test_ramdisk_cli_smoke.CliJsonSmokeTest.test_cli_sigterm_requests_cooperative_prepare_rollback",
        "test_ramdisk_cli_smoke.CliJsonSmokeTest.test_curses_repeated_sigterm_is_deferred_until_cleanup_guard_exits",
        "test_ramdisk_cli_smoke.CliJsonSmokeTest.test_curses_sigterm_uses_cleanup_exception_and_restores_handler",
        "test_ramdisk_curses_ui_module.CursesUiModuleTest.test_termination_guard_restores_the_previous_handler",
    }
)

_SIGINT_HANDLER_TESTS = frozenset(
    {
        "test_ramdisk_cli_module.CliModuleTest.test_prepare_and_destroy_confirmations_keep_ctrl_c_interruptible",
        "test_ramdisk_cli_module.CliModuleTest.test_prepare_restores_cooperative_ctrl_c_after_confirmation",
        "test_ramdisk_cli_module.CliModuleTest.test_real_tty_ctrl_c_interrupts_confirmation_without_input",
        "test_ramdisk_cli_smoke.CliJsonSmokeTest.test_cli_repeated_sigint_stays_cooperative_through_start_rollback",
    }
)

_POSIX_PTY_TESTS = frozenset(
    {
        "test_ramdisk_cli_module.CliModuleTest.test_real_tty_ctrl_c_interrupts_confirmation_without_input",
    }
)

_POSIX_FIFO_TESTS = frozenset(
    {
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_managed_usage_swap_to_fifo_uses_nonblocking_open",
    }
)

_NATIVE_DIRFD_TESTS = frozenset(
    {
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_atomic_temp_creation_stays_inside_bound_parent",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_existing_marker_reproves_canonical_parent_before_journal_unlink",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_managed_usage_merge_rejects_symlink_swap_during_verified_open",
        "test_ramdisk_state_lifecycle.StateAndSafetyTest.test_managed_usage_seed_write_binds_parent_identity",
    }
)

_LINUX_PIDFD_TESTS = frozenset(
    {
        "test_ramdisk_platform.LinuxOperationalReadContractTest.test_real_pidfd_group_signal_targets_each_exact_member",
    }
)

_LINUX_STDLIB_PIDFD_TESTS = frozenset(
    {
        "test_coli_stop.ColiStopIdentityTest.test_pidfd_is_used_instead_of_numeric_pid_when_available",
    }
)

_TARGET_PLATFORM = os.environ.get(
    "COLIBRI_TEST_TARGET_PLATFORM",
    sys.platform,
)


def _linux_pidfd_supported():
    """Match the managed runtime's stdlib-or-libc kernel capability probe."""
    if not _TARGET_PLATFORM.startswith("linux"):
        return False
    try:
        from ramdisk_support import linux_ops
    except ImportError:
        return False
    return linux_ops._pidfd_process_control_supported()


def _native_dirfd_supported():
    """Match the descriptor primitives required by durable managed state."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    return (
        not _TARGET_PLATFORM.startswith("win")
        and os.name == "posix"
        and os.open in getattr(os, "supports_dir_fd", set())
        and os.stat in getattr(os, "supports_dir_fd", set())
        and os.unlink in getattr(os, "supports_dir_fd", set())
        and os.rename in getattr(os, "supports_dir_fd", set())
        and all(
            isinstance(getattr(os, name, None), int)
            and getattr(os, name) != 0
            for name in required_flags
        )
    )


PLATFORM_SKIP_INVENTORY = {
    "linux_operational": {
        "supported": _TARGET_PLATFORM.startswith("linux"),
        "reason": "requires Linux mount, procfs, process-group, or benchmark operations",
        "tests": _LINUX_OPERATIONAL_TESTS,
    },
    "sigterm_handler": {
        "supported": hasattr(signal, "SIGTERM"),
        "reason": "SIGTERM is unavailable",
        "tests": _SIGTERM_HANDLER_TESTS,
    },
    "sigint_handler": {
        "supported": hasattr(signal, "SIGINT"),
        "reason": "SIGINT is unavailable",
        "tests": _SIGINT_HANDLER_TESTS,
    },
    "posix_pty": {
        "supported": os.name == "posix" and hasattr(os, "openpty"),
        "reason": "a POSIX pseudo-terminal is required",
        "tests": _POSIX_PTY_TESTS,
    },
    "posix_fifo": {
        "supported": (
            not _TARGET_PLATFORM.startswith("win")
            and os.name == "posix"
            and hasattr(os, "mkfifo")
        ),
        "reason": "a POSIX FIFO is required",
        "tests": _POSIX_FIFO_TESTS,
    },
    "native_dirfd": {
        "supported": _native_dirfd_supported(),
        "reason": "native descriptor-relative filesystem operations are unavailable",
        "tests": _NATIVE_DIRFD_TESTS,
    },
    "linux_pidfd": {
        "supported": _linux_pidfd_supported(),
        "reason": "Linux pidfd signaling is unavailable",
        "tests": _LINUX_PIDFD_TESTS,
    },
    "linux_stdlib_pidfd": {
        "supported": (
            _TARGET_PLATFORM.startswith("linux")
            and callable(getattr(os, "pidfd_open", None))
            and callable(getattr(signal, "pidfd_send_signal", None))
        ),
        "reason": "Python stdlib pidfd signaling is unavailable",
        "tests": _LINUX_STDLIB_PIDFD_TESTS,
    },
}


def _test_id(function):
    module = function.__module__.rsplit(".", 1)[-1]
    return "%s.%s" % (module, function.__qualname__)


def _requires(marker):
    entry = PLATFORM_SKIP_INVENTORY[marker]

    def decorate(function):
        identifier = _test_id(function)
        if identifier not in entry["tests"]:
            raise AssertionError(
                "unregistered %s platform marker: %s" % (marker, identifier)
            )
        return unittest.skipUnless(
            entry["supported"],
            entry["reason"],
        )(function)

    return decorate


requires_linux_operational = _requires("linux_operational")
requires_sigterm_handler = _requires("sigterm_handler")
requires_sigint_handler = _requires("sigint_handler")
requires_posix_pty = _requires("posix_pty")
requires_posix_fifo = _requires("posix_fifo")
requires_native_dirfd = _requires("native_dirfd")
requires_linux_pidfd = _requires("linux_pidfd")
requires_linux_stdlib_pidfd = _requires("linux_stdlib_pidfd")


def assert_platform_skip_inventory():
    """Fail when a checked marker is added, removed, or renamed silently."""
    decorator_markers = {
        "requires_linux_operational": "linux_operational",
        "requires_sigterm_handler": "sigterm_handler",
        "requires_sigint_handler": "sigint_handler",
        "requires_posix_pty": "posix_pty",
        "requires_posix_fifo": "posix_fifo",
        "requires_native_dirfd": "native_dirfd",
        "requires_linux_pidfd": "linux_pidfd",
        "requires_linux_stdlib_pidfd": "linux_stdlib_pidfd",
    }
    discovered = {marker: set() for marker in PLATFORM_SKIP_INVENTORY}
    tests_dir = Path(__file__).resolve().parent

    def raw_platform_skip(decorator):
        if not isinstance(decorator, ast.Call):
            return False
        target = decorator.func
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "unittest"
            and target.attr in ("skipIf", "skipUnless")
            and decorator.args
        ):
            return False
        condition_nodes = tuple(ast.walk(decorator.args[0]))
        names = {
            node.id for node in condition_nodes if isinstance(node, ast.Name)
        }
        attributes = {
            node.attr
            for node in condition_nodes
            if isinstance(node, ast.Attribute)
        }
        return (
            ("sys" in names and "platform" in attributes)
            or (
                "os" in names
                and attributes.intersection({"name", "openpty", "mkfifo"})
            )
            or "signal" in names
        )

    for source in tests_dir.glob("test_*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for class_node in (
            node for node in tree.body if isinstance(node, ast.ClassDef)
        ):
            if source.name.startswith("test_ramdisk_") and any(
                raw_platform_skip(decorator)
                for decorator in class_node.decorator_list
            ):
                raise AssertionError(
                    "raw class platform skip bypasses inventory: %s.%s"
                    % (source.stem, class_node.name)
                )
            for function in (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("test_")
            ):
                for decorator in function.decorator_list:
                    name = getattr(decorator, "id", None)
                    marker = decorator_markers.get(name)
                    if marker is not None:
                        discovered[marker].add(
                            "%s.%s.%s"
                            % (source.stem, class_node.name, function.name)
                        )

                if source.name.startswith("test_ramdisk_"):
                    for decorator in function.decorator_list:
                        if raw_platform_skip(decorator):
                            raise AssertionError(
                                "raw platform skip bypasses inventory: %s.%s.%s"
                                % (source.stem, class_node.name, function.name)
                            )

                    for call in (
                        node
                        for node in ast.walk(function)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "skipTest"
                    ):
                        reason = " ".join(
                            node.value
                            for node in call.args
                            if isinstance(node, ast.Constant)
                            and isinstance(node.value, str)
                        ).upper()
                        if any(
                            token in reason
                            for token in (
                                "SIGINT",
                                "SIGTERM",
                                "LINUX",
                                "MACOS",
                                "POSIX",
                                "WINDOWS",
                            )
                        ):
                            raise AssertionError(
                                "dynamic platform skip bypasses inventory: %s.%s.%s"
                                % (source.stem, class_node.name, function.name)
                            )

    mismatches = []
    for marker, entry in PLATFORM_SKIP_INVENTORY.items():
        expected = set(entry["tests"])
        if discovered[marker] != expected:
            mismatches.append(
                "%s missing=%r stale=%r"
                % (
                    marker,
                    sorted(expected - discovered[marker]),
                    sorted(discovered[marker] - expected),
                )
            )
    if mismatches:
        raise AssertionError("platform skip inventory drift: " + "; ".join(mismatches))
