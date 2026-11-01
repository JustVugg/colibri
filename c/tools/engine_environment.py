"""Preserve tuning while keeping each tool's requested engine mode explicit."""
import os


# colibri.c dispatches these before or instead of ordinary PROMPT/SCORE/REPLAY
# execution. In particular COLI_PROMPT wins over PROMPT, and ORACLE_STRICT
# rejects those modes. Clear only inherited selectors, not hardware/arm knobs.
_MODE_KEYS = (
    "PROMPT", "COLI_PROMPT", "SCORE", "ABLATE_SCORE", "SERVE", "SERVE_BATCH",
    "REPLAY", "CONSIST", "TF", "ORACLE_STRICT", "EXPERT_WORKER",
    "I4_ACC512_TEST", "I3_AVX512_TEST", "COLI_ANS_PACK",
)


def engine_environment(**requested):
    """Copy ambient tuning, drop ambient modes, then select the requested mode.

    Explicit caller overrides can still be applied to the returned mapping;
    the parent environment is never changed.
    """
    env = os.environ.copy()
    for key in _MODE_KEYS:
        env.pop(key, None)
    env.update(requested)
    return env
