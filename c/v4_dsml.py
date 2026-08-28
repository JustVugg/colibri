"""DeepSeek V4 DSML primitives, vendored from the checkpoint reference.

Byte-exact excerpts of ``encoding/encoding_dsv4.py`` (inside
``deepseek-ai/DeepSeek-V4-Flash-0731``) covering the DSML surfaces the gateway
speaks: the tool-declaration template, the ``<｜DSML｜tool_calls>`` block
encoding, and the strict block parser. Keeping these as verbatim reference
code (instead of a re-implementation) makes a checkpoint encoding revision a
mechanical re-vendor + diff.

What is deliberately NOT vendored:

- ``encode_messages`` / ``render_message`` — the gateway's turn structure
  (``render_chat_v4``) is authoritative: it always terminates the prompt with
  an assistant generation cue (the reference leaves a trailing assistant
  ``tool_calls`` turn uncued), renders ``reasoning_content`` for every turn
  that carries one, and merges ``tool`` messages itself.
- ``REASONING_EFFORT_PROMPTS`` — the gateway ships tuned effort prompts with
  an explicit token budget; the reference's unbounded "no shortcuts
  permitted" variants routinely consumed the whole completion budget with
  reasoning.

Gateway adaptations, each marked ADAPTED below:

- ``tool_calls_from_openai_format`` accepts ``arguments`` as a JSON string OR
  a dict (messages arrive from arbitrary OpenAI clients).
- ``parse_completion_text`` wraps the strict reference parser so malformed or
  truncated DSML degrades to plain content instead of raising, and re-shapes
  calls to the gateway contract ({"id", "type", "function": {...}}).
"""

import json
import re
import sys
import uuid

# ============================================================
# Special tokens and templates (byte-exact with the reference)
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

tool_call_template: str = (
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>"
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)
tool_calls_block_name: str = "tool_calls"

tool_output_template: str = "<tool_result>{content}</tool_result>"

# Marker that opens a tool_calls block in model output (used by the gateway's
# streaming suppression as well as the parser).
TOOL_CALLS_OPEN = f"<{dsml_token}{tool_calls_block_name}>"    # full opener (streaming suppression)
TOOL_CALLS_PREFIX = f"<{dsml_token}{tool_calls_block_name}"   # reference parse anchor (no '>')
TOOL_CALLS_BLOCK_START = "\n\n" + TOOL_CALLS_PREFIX           # as emitted in model output

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# ============================================================
# Encoding helpers (byte-exact unless marked ADAPTED)
# ============================================================

def to_json(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    return [tool.get("function", tool) for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    """ADAPTED: tolerate "arguments" given as a dict, not only a JSON string."""
    out = []
    for tool_call in tool_calls:
        fn = tool_call.get("function", tool_call) if isinstance(tool_call, dict) else {}
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = to_json(args)
        out.append({"name": fn.get("name"), "arguments": args})
    return out


def encode_arguments_to_dsml(tool_call):
    p_dsml_template = '<{dsml_token}parameter name="{key}" string="{is_str}">{value}</{dsml_token}parameter>'
    p_dsml_strs = []
    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception:
        arguments = {"arguments": tool_call["arguments"]}
    if not isinstance(arguments, dict):
        arguments = {"arguments": arguments}
    for k, v in arguments.items():
        p_dsml_strs.append(p_dsml_template.format(
            dsml_token=dsml_token, key=k,
            is_str="true" if isinstance(v, str) else "false",
            value=v if isinstance(v, str) else to_json(v)))
    return "\n".join(p_dsml_strs)


def decode_dsml_to_arguments(tool_name, tool_args):
    def _decode_value(key, value, string):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"
    tool_args_json = ("{" + ", ".join(
        [_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]) + "}")
    return dict(name=tool_name, arguments=tool_args_json)


def render_tools(tools):
    tools_json = [to_json(t) for t in tools]
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def render_tool_calls(tool_calls):
    """OpenAI-format tool_calls -> the DSML block a V4 assistant turn carries,
    including the reference's leading blank line."""
    calls = tool_calls_from_openai_format(tool_calls)
    rendered = [
        tool_call_template.format(dsml_token=dsml_token,
                                  name=tc.get("name") or "",
                                  arguments=encode_arguments_to_dsml(tc))
        for tc in calls
    ]
    return "\n\n" + tool_calls_template.format(
        dsml_token=dsml_token, tool_calls="\n".join(rendered),
        tc_block_name=tool_calls_block_name)


# ============================================================
# Parsing — strict reference core + tolerant gateway wrapper
# ============================================================

def _read_until_stop(index, text, stop):
    min_pos = len(text)
    matched_stop = None
    for s in stop:
        pos = text.find(s, index)
        if pos != -1 and pos < min_pos:
            min_pos = pos
            matched_stop = s
    if matched_stop:
        return min_pos + len(matched_stop), text[index:min_pos], matched_stop
    return len(text), text[index:], None


def _parse_tool_calls_strict(index, text):
    """Reference parse of a <dsml>tool_calls block. Raises ValueError on malformed."""
    tool_calls = []
    tool_calls_end_token = f"</{dsml_token}{tool_calls_block_name}>"
    while index < len(text):
        index, sep, stop_token = _read_until_stop(
            index, text, [f"<{dsml_token}invoke", tool_calls_end_token])
        if sep != ">\n":
            raise ValueError(f"Tool call format error: expected '>\\n' but got '{sep}'")
        if stop_token == tool_calls_end_token:
            break
        if stop_token is None:
            raise ValueError("Missing special token in tool calls")
        index, tool_name_content, stop_token = _read_until_stop(
            index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])
        p_tool_name = re.findall(r'^\s*name="(.*?)">\n$', tool_name_content, flags=re.DOTALL)
        if len(p_tool_name) != 1:
            raise ValueError(f"Tool name format error: '{tool_name_content}'")
        tool_name = p_tool_name[0]
        tool_args = {}
        while stop_token == f"<{dsml_token}parameter":
            index, param_content, stop_token = _read_until_stop(
                index, text, [f"/{dsml_token}parameter"])
            param_kv = re.findall(r'^ name="(.*?)" string="(true|false)">(.*?)<$',
                                  param_content, flags=re.DOTALL)
            if len(param_kv) != 1:
                raise ValueError(f"Parameter format error: '{param_content}'")
            param_name, string, param_value = param_kv[0]
            if param_name in tool_args:
                raise ValueError(f"Duplicate parameter name: '{param_name}'")
            tool_args[param_name] = (param_value, string)
            index, content, stop_token = _read_until_stop(
                index, text, [f"<{dsml_token}parameter", f"</{dsml_token}invoke"])
            if content != ">\n":
                raise ValueError(f"Parameter format error: expected '>\\n' but got '{content}'")
        tool_calls.append(decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args))
    return index, stop_token, tool_calls


def parse_completion_text(text):
    """ADAPTED: split `content [+ blank line + DSML tool_calls block]` from a
    finished assistant turn. Never raises.

    Returns (content, tool_calls) where tool_calls use the gateway shape
    [{"id": "call_...", "type": "function", "function": {"name", "arguments"}}].
    Malformed or truncated DSML degrades to plain content (logged to stderr);
    the caller decides whether raw markers may stay visible.
    """
    cut = text.find(TOOL_CALLS_BLOCK_START)
    parse_at = cut + len(TOOL_CALLS_BLOCK_START) if cut >= 0 else -1
    if cut < 0 and text.startswith(TOOL_CALLS_PREFIX):
        cut, parse_at = 0, len(TOOL_CALLS_PREFIX)   # reply opening directly with the block
    if cut < 0:
        return text, []
    try:
        _index, _stop, calls = _parse_tool_calls_strict(parse_at, text)
    except ValueError as exc:
        sys.stderr.write(f"[api] deepseek_v4: DSML tool_calls parse failed ({exc}) "
                         "-- treating as plain content\n")
        sys.stderr.flush()
        return text, []
    shaped = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
               "function": {"name": c["name"], "arguments": c["arguments"]}}
              for c in calls]
    return text[:cut], shaped
