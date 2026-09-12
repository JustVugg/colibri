"""DeepSeek V4.1 DSML primitives, vendored from the checkpoint reference.

Byte-exact excerpts of ``encoding/encoding.py`` (inside
``deepseek-ai/DeepSeek-V4.1-Flash``) covering the DSML surfaces the gateway
speaks: the tool-declaration template, the ``<｜DSML｜ calls>`` block encoding,
and the strict block parser. Keeping these as verbatim reference code (instead
of a re-implementation) makes a checkpoint encoding revision a mechanical
re-vendor + diff, exactly as ``v4_dsml.py`` does for V4.

THE ONE DIFFERENCE THAT BITES: V4.1's DSML tag names carry a LEADING ASCII
SPACE. V4 wrote ``<｜DSML｜tool_calls>`` / ``<｜DSML｜invoke>`` /
``<｜DSML｜parameter>``; V4.1 writes ``<｜DSML｜ calls>`` / ``<｜DSML｜ invoke>``
/ ``<｜DSML｜ parameter>``, and the block name itself changed from
``tool_calls`` to ``calls``. The space is part of the tag name, not separator
whitespace: it is inside ``tool_calls_block_name`` and friends below, it is in
every template, and the parameter regex anchors on it (``^ name="..."``). A
tag built by concatenating the V4 way parses as a tool name of ``"calls"``
with no parameters, silently, so the names live in one place here and every
template interpolates them rather than spelling them out.

What is deliberately NOT vendored:

- ``encode_messages`` / ``render_message`` -- the gateway's turn structure
  (``render_chat_dsv41``) is authoritative: it always terminates the prompt
  with an assistant generation cue (the reference leaves a trailing assistant
  turn uncued), and it models neither ``latest_reminder`` nor the internal
  classification ``task`` tokens.
- ``process_image_messages`` -- the gateway expands images itself
  (``expand_dsv41_images``), because it must hand the engine ViT patches, not
  a file path.
- the ``REASONING_EFFORT_*`` block -- ``render_chat_dsv41`` renders the
  numeric budget line from its own effort table.

Gateway adaptations, each marked ADAPTED below:

- ``tool_calls_from_openai_format`` accepts ``arguments`` as a JSON string OR
  a dict (messages arrive from arbitrary OpenAI clients), and never asserts.
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

# The leading spaces are load-bearing: see the module docstring.
tool_calls_block_name: str = " calls"
tool_call_tag_name: str = " invoke"
tool_parameter_tag_name: str = " parameter"

tool_call_template: str = (
    "<{dsml_token}{tool_call_tag_name} name=\"{name}\">\n{arguments}\n"
    "</{dsml_token}{tool_call_tag_name}>"
)
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)

tool_output_template: str = "<tool_result>{content}</tool_result>"

# Markers that open a tool_calls block in model output (used by the gateway's
# streaming suppression as well as the parser).
TOOL_CALLS_OPEN = f"<{dsml_token}{tool_calls_block_name}>"    # full opener (streaming suppression)
TOOL_CALLS_PREFIX = f"<{dsml_token}{tool_calls_block_name}"   # reference parse anchor (no '>')
TOOL_CALL_PREFIX = f"<{dsml_token}{tool_call_tag_name}"       # a stray invoke, no calls block
TOOL_CALLS_BLOCK_START = "\n\n" + TOOL_CALLS_PREFIX           # as emitted in model output

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

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


def _split_tool_name(name, namespace=None):
    """Split a qualified `namespace::name`. ADAPTED: a conflicting namespace is
    dropped rather than asserted, so one odd client cannot 500 the request."""
    prefix, separator, bare_name = (name or "").partition("::")
    if separator:
        namespace, name = prefix, bare_name
    return namespace, name


def _tool_name_for_encoding(tool):
    """The wire name: `namespace::name` when the tool carries a namespace."""
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace.get("name")
    namespace, name = _split_tool_name(tool.get("name") or "", namespace)
    return name if namespace is None else f"{namespace}::{name}"


def tools_from_openai_format(tools):
    """Unwrap {"type": "function", "function": {...}} and qualify the name."""
    functions = []
    for tool in tools or []:
        function = dict(tool.get("function", tool) if isinstance(tool, dict) else {})
        if isinstance(tool, dict) and tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = (
                namespace["description"] + "\n" + (function.get("description") or ""))
        functions.append(function)
    return functions


def tool_calls_from_openai_format(tool_calls):
    """ADAPTED: tolerate "arguments" given as a dict, not only a JSON string."""
    calls = []
    for tool_call in tool_calls or []:
        fn = tool_call.get("function", tool_call) if isinstance(tool_call, dict) else {}
        namespace, name = _split_tool_name(
            fn.get("name") or "",
            (tool_call.get("namespace") if isinstance(tool_call, dict) else None)
            or fn.get("namespace"))
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = to_json(args)
        call = {"name": name, "arguments": args}
        if namespace is not None:
            call["namespace"] = namespace
        calls.append(call)
    return calls


def encode_arguments_to_dsml(tool_call):
    """Encode one call's arguments into V4.1 DSML parameter tags."""
    p_dsml_template = ('<{dsml_token}{tool_parameter_tag_name} name="{key}" string="{is_str}">'
                       '{value}</{dsml_token}{tool_parameter_tag_name}>')
    p_dsml_strs = []
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        # the reference tolerates JSON strings, including double-encoded ones
        for _ in range(2):
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    break
            else:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call.get("arguments")}
    for key, value in arguments.items():
        p_dsml_strs.append(p_dsml_template.format(
            dsml_token=dsml_token,
            tool_parameter_tag_name=tool_parameter_tag_name,
            key=key,
            is_str="true" if isinstance(value, str) else "false",
            value=value if isinstance(value, str) else to_json(value)))
    return "\n".join(p_dsml_strs)


def decode_dsml_to_arguments(tool_name, tool_args):
    """DSML parameters back to a call dict, arguments as a JSON string."""
    def _decode_value(key, value, string):
        if string == "true":
            value = to_json(value)
        return f"{to_json(key)}: {value}"
    tool_args_json = ("{" + ", ".join(
        [_decode_value(k, v, string=is_str) for k, (v, is_str) in tool_args.items()]) + "}")
    namespace, name = _split_tool_name(tool_name)
    tool_call = dict(name=name, arguments=tool_args_json)
    if namespace is not None:
        tool_call["namespace"] = namespace
    return tool_call


def render_tools(tools):
    """The `## Tools` block a V4.1 system message carries. One JSON schema per
    line, not a JSON array."""
    tools_json = [to_json(t) for t in tools]
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        tc_block_name=tool_calls_block_name,
        tool_call_tag_name=tool_call_tag_name,
        tool_parameter_tag_name=tool_parameter_tag_name,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def render_tool_calls(tool_calls):
    """OpenAI-format tool_calls -> the DSML block a V4.1 assistant turn carries,
    including the reference's leading blank line."""
    calls = tool_calls_from_openai_format(tool_calls)
    rendered = [
        tool_call_template.format(dsml_token=dsml_token,
                                  tool_call_tag_name=tool_call_tag_name,
                                  name=_tool_name_for_encoding(call),
                                  arguments=encode_arguments_to_dsml(call))
        for call in calls
    ]
    return "\n\n" + tool_calls_template.format(
        dsml_token=dsml_token, tool_calls="\n".join(rendered),
        tc_block_name=tool_calls_block_name)


def render_tool_result(content):
    """One tool message, as it appears inside the following user turn."""
    return tool_output_template.format(content=content)


# ============================================================
# Parsing -- strict reference core + tolerant gateway wrapper
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
    """Reference parse of a <｜DSML｜ calls> block. Raises ValueError on malformed."""
    tool_calls = []
    stop_token = None
    tool_calls_end_token = f"</{dsml_token}{tool_calls_block_name}>"
    tool_call_start_token = f"<{dsml_token}{tool_call_tag_name}"
    tool_call_end_token = f"</{dsml_token}{tool_call_tag_name}"
    tool_parameter_start_token = f"<{dsml_token}{tool_parameter_tag_name}"
    tool_parameter_end_token = f"/{dsml_token}{tool_parameter_tag_name}"
    while index < len(text):
        index, sep, stop_token = _read_until_stop(
            index, text, [tool_call_start_token, tool_calls_end_token])
        if sep != ">\n":
            raise ValueError(f"Tool call format error: expected '>\\n' but got '{sep}'")
        if stop_token == tool_calls_end_token:
            break
        if stop_token is None:
            raise ValueError("Missing special token in tool calls")
        index, tool_name_content, stop_token = _read_until_stop(
            index, text, [tool_parameter_start_token, tool_call_end_token])
        p_tool_name = re.findall(r'^\s*name="(.*?)">\n$', tool_name_content, flags=re.DOTALL)
        if len(p_tool_name) != 1:
            raise ValueError(f"Tool name format error: '{tool_name_content}'")
        tool_name = p_tool_name[0]
        tool_args = {}
        while stop_token == tool_parameter_start_token:
            index, param_content, stop_token = _read_until_stop(
                index, text, [tool_parameter_end_token])
            param_kv = re.findall(r'^ name="(.*?)" string="(true|false)">(.*?)<$',
                                  param_content, flags=re.DOTALL)
            if len(param_kv) != 1:
                raise ValueError(f"Parameter format error: '{param_content}'")
            param_name, string, param_value = param_kv[0]
            if param_name in tool_args:
                raise ValueError(f"Duplicate parameter name: '{param_name}'")
            tool_args[param_name] = (param_value, string)
            index, content, stop_token = _read_until_stop(
                index, text, [tool_parameter_start_token, tool_call_end_token])
            if content != ">\n":
                raise ValueError(f"Parameter format error: expected '>\\n' but got '{content}'")
        tool_calls.append(decode_dsml_to_arguments(tool_name=tool_name, tool_args=tool_args))
    return index, stop_token, tool_calls


def parse_completion_text(text):
    """ADAPTED: split `content [+ blank line + DSML calls block]` from a
    finished assistant turn. Never raises.

    Returns (content, tool_calls) where tool_calls use the gateway shape
    [{"id": "call_...", "type": "function", "function": {"name", "arguments"}}].
    Malformed or truncated DSML degrades to plain content (logged to stderr);
    the caller decides whether raw markers may stay visible.

    V4.1 puts no id on the wire, so ids are minted here -- the same contract as
    every other family, where the id only has to round-trip within one
    conversation.
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
        sys.stderr.write(f"[api] deepseek_v41: DSML tool calls parse failed ({exc}) "
                         "-- treating as plain content\n")
        sys.stderr.flush()
        return text, []
    shaped = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
               "function": {"name": c["name"], "arguments": c["arguments"]}}
              for c in calls]
    return text[:cut], shaped
