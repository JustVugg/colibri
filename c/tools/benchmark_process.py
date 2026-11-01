"""Bounded subprocess cleanup shared by the mux benchmark tools."""
import subprocess


def close_engine(process):
    """Allow EOF shutdown, then terminate and kill an engine that does not exit."""
    if process.stdin:
        try:
            process.stdin.close()
        except OSError:
            pass                        # the child may already have closed its pipe
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
