"""The machine's power draw and each turn's energy, read by joules (pip install landauer-gap) when it is installed.

Optional, like the rest of the gateway's extras: without joules, or on a machine with no power counters, /power
answers {"available": false, "reason": ...} and the pages show no power. Nothing runs until a page first asks: the
first /power starts one joules session (NVIDIA, AMD and Intel GPUs, the CPU's RAPL, Apple's powermetrics) and a
thread that reads it once a second. From then on every /profile turn also carries energy_j, what the devices used
between the turn's start and its end, so a page can show joules per token.

    {"available": true, "watts": 412.5, "energy_j": 81234.0, "since": 1730700000.0, "simulated": false,
     "devices": [{"name": "GPU 0 NVIDIA H200", "kind": "gpu", "watts": 380.1}, ...]}

watts is the average over the last second, energy_j everything the devices used since the session started."""
import bisect
import collections
import threading
import time

INTERVAL = 1.0
HISTORY = 3600          # one reading a second: a turn of up to an hour gets its energy
_lock = threading.Lock()
_state = None           # what /power answers; None until the first request
_session = None
_history = collections.deque(maxlen=HISTORY)   # (monotonic time, total joules)


def _unavailable(reason):
    return {"available": False, "reason": reason}


def _open():
    try:
        import joules
    except ImportError:
        return None, _unavailable("joules is not installed (pip install landauer-gap)")
    try:
        session = joules.Session(interval=0.25)
    except joules.NoMeters as e:
        return None, _unavailable("no power counters on this machine (%s)" % e)
    except Exception as e:   # noqa: BLE001 - a driver that will not answer: no power, the pages keep working
        return None, _unavailable("%s: %s" % (type(e).__name__, e))
    return session, {"available": True, "watts": None, "energy_j": 0.0, "since": time.time(),
                     "simulated": session.discovery.simulated,
                     "devices": [{"name": d["name"], "kind": d["kind"], "watts": None} for d in session.devices]}


def _read(session, prev):
    t, snap = time.monotonic(), session.snapshot()
    j = snap["devices_j"]
    with _lock:
        _history.append((t, snap["total_j"]))
        if prev is not None and t > prev[0]:
            dt = t - prev[0]
            for d in _state["devices"]:
                d["watts"] = round(max(0.0, j.get(d["name"], 0.0) - prev[1].get(d["name"], 0.0)) / dt, 1)
            _state["watts"] = round(sum(d["watts"] for d in _state["devices"]), 1)
        _state["energy_j"] = round(snap["total_j"], 1)
    return t, j


def _loop(session):
    prev = None
    while _session is session:
        try:
            prev = _read(session, prev)
        except Exception:   # noqa: BLE001 - one bad reading: try again next second
            pass
        time.sleep(INTERVAL)


def reading():
    """The last reading; the first call starts the session and its thread."""
    global _state, _session
    with _lock:
        if _state is None:
            _session, _state = _open()
            if _session is not None:
                _history.clear()
                threading.Thread(target=_loop, args=(_session,), name="colibri-power", daemon=True).start()
        if not _state["available"]:
            return dict(_state)
        return {**_state, "devices": [dict(d) for d in _state["devices"]]}


def energy_since(seconds_ago):
    """The joules the devices used over the last `seconds_ago` seconds, up to now: a turn's energy when it ends.
    None while nothing has asked for /power, or when the turn began before the session's first reading."""
    session = _session
    if session is None or seconds_ago < 0:
        return None
    now, total = time.monotonic(), session.snapshot()["total_j"]
    t0 = now - seconds_ago
    with _lock:
        h = list(_history)
    if not h or t0 < h[0][0]:
        return None
    i = bisect.bisect_right([t for t, _ in h], t0)
    if i >= len(h):          # the turn began after the last reading: interpolate towards now
        (ta, ja), (tb, jb) = h[-1], (now, total)
    else:
        (ta, ja), (tb, jb) = h[i - 1], h[i]
    at_t0 = ja + (jb - ja) * ((t0 - ta) / (tb - ta) if tb > ta else 0.0)
    return max(0.0, total - at_t0)
