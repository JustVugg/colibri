"""MCP wait durations and EOF draining use elapsed time, not civil time."""
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcp_server


class Clock:
    def __init__(self, jump):
        self.elapsed = 0
        self.wall_reads = 0
        self.jump = jump

    def time(self):
        self.wall_reads += 1
        return 100 + self.elapsed + (self.jump if self.wall_reads > 1 else 0)

    def monotonic(self):
        return self.elapsed

    def sleep(self, duration):
        self.elapsed += duration
        if self.elapsed > 4:
            raise AssertionError('wait exceeded its elapsed-time bound')


class WaitClockTest(unittest.TestCase):
    def test_install_wait_keeps_duration_across_wall_clock_changes(self):
        for jump in (-3600, 0, 3600):
            with self.subTest(jump=jump):
                clock = Clock(jump)
                process = SimpleNamespace(pid=123, poll=lambda: None)
                with mock.patch.object(mcp_server, 'time', clock), \
                        mock.patch.object(mcp_server, '_launch_install',
                                          return_value=(process, 'fixture.log', ['fixture'])), \
                        mock.patch.object(mcp_server, '_current_install_state',
                                          return_value={'phase': 'build', 'pid': 123}):
                    result = mcp_server.tool_install({'wait_seconds': 2}, lambda *a: None)
                self.assertEqual(clock.elapsed, 2)
                self.assertIsNone(result['exit_code'])

    def test_start_wait_keeps_duration_across_wall_clock_changes(self):
        for jump in (-3600, 0, 3600):
            with self.subTest(jump=jump):
                clock = Clock(jump)
                with mock.patch.object(mcp_server, 'time', clock), \
                        mock.patch.object(mcp_server.setup_flow, 'load_config', return_value={'fixture': True}), \
                        mock.patch.object(mcp_server.setup_flow, 'config_ready', return_value=True), \
                        mock.patch.object(mcp_server.setup_flow, 'start_server', return_value={'state': 'loading'}), \
                        mock.patch.object(mcp_server.setup_flow, 'server_status', return_value={'state': 'loading'}):
                    result = mcp_server.tool_start({'wait_seconds': 2}, lambda *a: None)
                self.assertEqual(clock.elapsed, 2)
                self.assertEqual(result['state'], 'loading')

    def test_eof_drain_budget_ignores_wall_clock_change(self):
        for jump in (-3600, 0, 3600):
            with self.subTest(jump=jump):
                clock = Clock(jump)
                server = mcp_server.Server(io.StringIO())
                waits = []

                def join(timeout):
                    waits.append(timeout)
                    clock.elapsed += 40

                server.threads = [SimpleNamespace(join=join), SimpleNamespace(join=join)]
                with mock.patch.object(mcp_server, 'time', clock):
                    server.serve(io.StringIO(''))
                self.assertEqual(waits, [60, 20])


if __name__ == '__main__':
    unittest.main()
