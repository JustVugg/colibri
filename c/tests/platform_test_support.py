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
        "test_ramdisk_mounts.MountAndCopyTest.test_multi_mount_failure_preflights_all_before_any_unmount",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_cleanup_runs_even_when_error_manifest_cannot_be_saved",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_never_unmounts_identityless_successful_mount_by_path",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_persists_pending_ownership_before_mount_helper",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_promotes_only_the_exact_recorded_mount_identity",
        "test_ramdisk_mounts.MountAndCopyTest.test_prepare_rollback_refuses_exact_mount_with_nested_child",
        "test_ramdisk_presentation.TuiPlacementContractTest.test_cancelled_prepare_does_not_hide_rollback_failure",
        "test_ramdisk_presentation.TuiPlacementContractTest.test_clean_prepare_cancellation_removes_recovery_manifest",
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
        "test_ramdisk_cli_smoke.CliJsonSmokeTest.test_cli_repeated_sigint_stays_cooperative_through_start_rollback",
    }
)

_LINUX_PIDFD_TESTS = frozenset(
    {
        "test_coli_stop.ColiStopIdentityTest.test_pidfd_is_used_instead_of_numeric_pid_when_available",
    }
)

_TARGET_PLATFORM = os.environ.get(
    "COLIBRI_TEST_TARGET_PLATFORM",
    sys.platform,
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
    "linux_pidfd": {
        "supported": (
            _TARGET_PLATFORM.startswith("linux")
            and hasattr(signal, "pidfd_send_signal")
        ),
        "reason": "Linux pidfd signaling is unavailable",
        "tests": _LINUX_PIDFD_TESTS,
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
requires_linux_pidfd = _requires("linux_pidfd")


def assert_platform_skip_inventory():
    """Fail when a checked marker is added, removed, or renamed silently."""
    decorator_markers = {
        "requires_linux_operational": "linux_operational",
        "requires_sigterm_handler": "sigterm_handler",
        "requires_sigint_handler": "sigint_handler",
        "requires_linux_pidfd": "linux_pidfd",
    }
    discovered = {marker: set() for marker in PLATFORM_SKIP_INVENTORY}
    tests_dir = Path(__file__).resolve().parent
    for source in tests_dir.glob("test_*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for class_node in (
            node for node in tree.body if isinstance(node, ast.ClassDef)
        ):
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
