"""The listen backlog of APIServer (c/openai_server.py).

socketserver.TCPServer listens with request_queue_size 5 unless told otherwise
(socket.SOMAXCONN from Python 3.15: 128 on macOS, 4096 on current Linux).
That is the kernel's accept queue: the completed handshakes the accept loop has
not drained yet. Once it is full, Linux drops the next SYN (ListenOverflows in
/proc/net/netstat) and that client sits out the SYN retransmit timer -- 1 s,
then 2 s, then 4 s -- before the server has even seen it. APIServer's caps
(MAX_CONNECTIONS 64, 8 per address) run after accept(), so they cannot see it.
APIServer asks for 128, which the kernel caps at net.core.somaxconn.

Measured on a 16-core box at load ~40, 100 barrier-synchronised connects each
doing GET /v1/models, three runs per setting: backlog 5 -> 47..87 of 100 waited
over 900 ms (max 2.2..7.9 s) and ListenOverflows rose by 91..309 per burst;
backlog 128 -> 0 of 100 (max 0.2..0.8 s), ListenOverflows unchanged.

The burst test is Linux-only: it reads the kernel's accept queue (ss, the
LISTEN socket's Send-Q) and its overflow counter. Elsewhere only the attribute
is checked.
"""

import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
from collections import Counter
from pathlib import Path

from openai_server import APIServer

BURST = 100
SLOW_SECONDS = 0.9      # a dropped SYN costs its client the 1 s retransmit timer at least


class Engine:
    """GET /v1/models never reaches the engine; APIServer only needs an object."""

    def generate(self, *args, **kwargs):
        raise AssertionError("GET /v1/models must not call the engine")


def _listen_overflows():
    """Linux's count of SYNs dropped because a listen queue was full, all listeners."""
    header = values = None
    with open("/proc/net/netstat") as f:
        for line in f:
            if line.startswith("TcpExt:"):
                if header is None:
                    header = line.split()
                else:
                    values = line.split()
    return int(values[header.index("ListenOverflows")])


def _kernel_backlog(port):
    """The accept queue the kernel gave the listener on `port`: ss prints it as
    the LISTEN socket's Send-Q. None when ss is not there to ask."""
    ss = shutil.which("ss")
    if not ss:
        return None
    try:
        listing = subprocess.run([ss, "-ltn", f"sport = :{port}"],
                                 capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    rows = [line.split() for line in listing.splitlines() if line.startswith("LISTEN")]
    if len(rows) != 1:
        raise AssertionError(listing)
    return int(rows[0][2])


def _burst(port, n):
    """n threads released together: connect, GET /v1/models, read the status
    line. Per connection: seconds from connect to the status line, and the
    status line or the error."""
    barrier = threading.Barrier(n)
    out = [None] * n

    def one(i):
        barrier.wait()
        t0 = time.perf_counter()
        s = socket.socket()
        s.settimeout(15)
        try:
            s.connect(("127.0.0.1", port))
            s.sendall(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            data = b""
            while b"\r\n" not in data:
                chunk = s.recv(512)
                if not chunk:
                    break
                data += chunk
            status = data.split(b"\r\n", 1)[0].decode(errors="replace") if data else "closed without a reply"
        except OSError as e:
            status = f"{type(e).__name__}: {e}"
        finally:
            s.close()
        out[i] = (time.perf_counter() - t0, status)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


class ListenBacklogTest(unittest.TestCase):
    def test_backlog_holds_the_connection_cap(self):
        self.assertEqual(APIServer.request_queue_size, 128)
        self.assertGreaterEqual(APIServer.request_queue_size, APIServer.MAX_CONNECTIONS)

    @unittest.skipUnless(sys.platform.startswith("linux"),
                         "reads the Linux kernel's accept queue and its overflow counter")
    def test_a_burst_of_100_is_accepted_without_a_dropped_syn(self):
        server = APIServer(("127.0.0.1", 0), Engine(), "fixture")
        server.MAX_CONNECTIONS = server.MAX_CONNECTIONS_PER_IP = 1000   # the post-accept caps are not under test
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        try:
            port = server.server_port
            backlog = _kernel_backlog(port)
            before = _listen_overflows()
            results = _burst(port, BURST)
            after = _listen_overflows()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        statuses = Counter(status for _, status in results)
        slow = sorted(round(seconds, 3) for seconds, _ in results if seconds >= SLOW_SECONDS)
        what_happened = (
            f"the accept queue overflowed {after - before} time(s) during the burst, "
            f"{len(slow)} of {BURST} connections waited "
            f"{slow[0] if slow else 0}..{slow[-1] if slow else 0} s for a status line "
            f"(kernel backlog {backlog}), statuses {dict(statuses)}"
        )
        self.assertEqual(statuses, Counter({"HTTP/1.1 200 OK": BURST}), what_happened)
        # A SYN our listener dropped costs that client the retransmit timer, a
        # second at least, and moves the counter. A slow scheduler alone can
        # delay a client but does not move the counter; another listener's
        # overflow moves the counter but does not delay ours. Both together
        # are our dropped SYN.
        self.assertFalse(after > before and slow, what_happened)
        if backlog is not None:
            somaxconn = int(Path("/proc/sys/net/core/somaxconn").read_text())
            self.assertEqual(backlog, min(APIServer.request_queue_size, somaxconn))


if __name__ == "__main__":
    unittest.main()
