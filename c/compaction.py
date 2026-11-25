"""Context compaction for stateless chat requests.

A chat that outgrows the engine's context used to end in a 400 the user could do
nothing with but start over. A request that asks for it (`context_compaction: "auto"`,
which the chat UI and `coli chat` send) is compacted instead: the oldest turns are
summarized by the model itself, the summary joins the system message, and the turn
runs on the summary plus the last exchanges, which stay verbatim.

The server keeps the summaries, keyed by the conversation prefix each one replaces.
The client keeps sending its whole history, as the API is stateless; every later turn
finds its summary here and renders the same compacted prompt, so the engine's prefix
reuse works across turns exactly as before. A restarted server summarizes once more.
"""

import collections
import hashlib
import json
import os
import threading

COMPACT_NOTE = ("Summary of the earlier part of this conversation (its messages were removed "
                "to fit the context window):\n\n")
COMPACT_ASK = ("Summarize our conversation so far so that it can continue from your summary "
               "alone. Keep everything still needed: what I asked for and want, facts, names, "
               "numbers, decisions, code and file names, and open questions. Write it in the "
               "language of the conversation, in at most {words} words, and reply with the "
               "summary only.")
# What a kept tail may take of the context: the summary and the room to answer take the rest,
# so the next compaction is many turns away.
COMPACT_KEEP_SHARE = 0.25
# Tokens per character before the server has measured any (UTF-8 prose runs near 0.25-0.3)
COMPACT_DEFAULT_RATIO = 0.35
COMPACT_IMAGE_TOKENS = 768

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

_ERROR_CLS = None


def set_error_class(cls):
    """Bind the gateway's APIError class so validation errors keep HTTP metadata."""
    global _ERROR_CLS
    _ERROR_CLS = cls


def _error(status, message, param=None):
    if _ERROR_CLS is not None:
        return _ERROR_CLS(status, message, param)
    return ValueError(message)


def compaction_requested(body):
    """True when the request opts into context compaction; COLI_COMPACT=1 makes it the default."""
    value = body.get("context_compaction")
    if value is None:
        return os.environ.get("COLI_COMPACT", "0") == "1"
    if value not in ("auto", "off", True, False):
        raise _error(400, "`context_compaction` must be \"auto\" or \"off\".", "context_compaction")
    return value in ("auto", True)


def _message_canon(message):
    try:
        return json.dumps(message, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(message)


def compaction_keys(lead, conversation):
    """keys[k] names the conversation's first k messages after its leading system messages."""
    digest = hashlib.sha256(_message_canon(lead).encode("utf-8", "replace")).digest()
    keys = [digest]
    for message in conversation:
        digest = hashlib.sha256(digest + _message_canon(message).encode("utf-8", "replace")).digest()
        keys.append(digest)
    return keys


def split_lead(messages):
    """(the leading system/developer messages, the rest)"""
    n = 0
    while n < len(messages) and isinstance(messages[n], dict) and \
            messages[n].get("role") in ("system", "developer"):
        n += 1
    return list(messages[:n]), list(messages[n:])


def with_summary(lead, summary):
    """The leading messages with the summary appended to the system message (or as one)."""
    if summary is None:
        return list(lead)
    note = COMPACT_NOTE + summary.strip()
    if lead and lead[0].get("role") == "system":
        first = dict(lead[0])
        content = first.get("content")
        if isinstance(content, list):
            first["content"] = list(content) + [{"type": "text", "text": "\n\n" + note}]
        elif isinstance(content, str) and content.strip():
            first["content"] = content.rstrip() + "\n\n" + note
        else:
            first["content"] = note
        return [first] + list(lead[1:])
    return [{"role": "system", "content": note}] + list(lead)


def without_images(message):
    """The message with its pictures as a mark: the summary reads text, and the engine
    takes one picture per request."""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts, changed = [], False
    for part in content:
        if isinstance(part, dict) and part.get("type") not in ("text", "input_text"):
            parts.append({"type": "text", "text": "[image]"})
            changed = True
        else:
            parts.append(part)
    return dict(message, content=parts) if changed else message


def estimate_message_tokens(message, ratio):
    content = message.get("content")
    chars, images = 0, 0
    if isinstance(content, str):
        chars = len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                chars += len(str(part.get("text") or ""))
            elif isinstance(part, dict):
                images += 1
    for key in ("reasoning_content", "tool_calls"):
        if message.get(key):
            chars += len(_message_canon(message[key]))
    return int(chars * ratio) + 8 + images * COMPACT_IMAGE_TOKENS


def compaction_tail_start(conversation, done, ratio, keep_tokens):
    """Where the verbatim tail begins: the earliest user turn after `done` whose tail fits
    `keep_tokens`, and never later than the last user turn (the new request). None when
    nothing before that turn is left to summarize."""
    users = [i for i in range(done, len(conversation))
             if isinstance(conversation[i], dict) and conversation[i].get("role") == "user"]
    if not users:
        return None
    start = users[-1]
    for i in reversed(users[:-1]):
        if i <= done:
            break
        if sum(estimate_message_tokens(m, ratio) for m in conversation[i:]) > keep_tokens:
            break
        start = i
    return start if start > done else None


def close_open_thinking(prompt):
    """A prompt that opens the model's reasoning block, closed: the summary is written
    without thinking, after the same prefix the conversation was rendered with.
    Returns (prompt, whether it closed one)."""
    if prompt.endswith(THINK_OPEN + "\n"):
        return prompt + "\n" + THINK_CLOSE + "\n\n", True
    if prompt.endswith(THINK_OPEN):
        return prompt + THINK_CLOSE, True
    return prompt, False


class CompactionMemory:
    """The summaries this server wrote, by the conversation prefix each one replaces."""

    def __init__(self, capacity=256):
        self.capacity = capacity
        self._items = collections.OrderedDict()
        self._lock = threading.Lock()

    def find(self, keys):
        """(k, summary) for the longest remembered prefix, (0, None) for none."""
        with self._lock:
            for k in range(len(keys) - 1, 0, -1):
                summary = self._items.get(keys[k])
                if summary is not None:
                    self._items.move_to_end(keys[k])
                    return k, summary
        return 0, None

    def remember(self, key, summary):
        with self._lock:
            self._items[key] = summary
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
