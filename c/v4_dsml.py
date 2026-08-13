"""
DeepSeek V4 DSML tool-call encoding for the OpenAI/Anthropic gateway.

Vendored and adapted from the official DeepSeek-V4-Flash checkpoint reference
(encoding/encoding_dsv4.py inside deepseek-ai/DeepSeek-V4-Flash-0731). The
encode side is kept byte-exact with the reference (validated against the
checkpoint's own fixtures); the parse side is made fault-tolerant for gateway
use: malformed model output degrades to plain content instead of raising, and
tool_calls are re-shaped to the gateway contract
({"id", "type", "function": {"name", "arguments"}}).

Adaptations vs the reference:
- tool_calls_from_openai_format: accept "arguments" as JSON string OR dict
  (gateway messages come from arbitrary OpenAI clients).
- parse_completion_text(): tolerant wrapper around the strict reference
  parser; returns (content, tool_calls, ok) and never raises.
- License note for the PR: verify the checkpoint LICENSE covers the reference
  code (DeepSeek code is typically MIT; the model license covers weights).
"""

import copy
import json
import re
import sys
import uuid

# ============================================================
# Special Tokens (byte-exact with the checkpoint reference)
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

system_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning_content}"

response_format_template: str = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
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
TOOL_CALLS_OPEN = f"<{dsml_token}{tool_calls_block_name}>"      # full opener (streaming suppression)
TOOL_CALLS_PREFIX = f"<{dsml_token}{tool_calls_block_name}"      # reference parse anchor (no '>')
TOOL_CALLS_BLOCK_START = "\n\n" + TOOL_CALLS_PREFIX            # as emitted in model output

REASONING_EFFORT_PROMPTS = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum \u2014 exhaustive, relentless, and uncompromising.\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n"
    ),
}
DEFAULT_REASONING_EFFORT = "low"

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
# Utility Functions (byte-exact unless noted)
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


def tool_calls_to_openai_format(tool_calls):
    return [
        {"type": "function",
         "function": {"name": tc["name"], "arguments": tc["arguments"]}}
        for tc in tool_calls
    ]


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


def find_last_user_index(messages):
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") in ["user", "developer"]:
            last_user_index = idx
            break
    return last_user_index


# ============================================================
# Message Rendering (byte-exact)
# ============================================================

def render_message(index, messages, thinking_mode, drop_thinking=True, reasoning_effort=None):
    assert 0 <= index < len(messages)
    assert thinking_mode in ["chat", "thinking"], f"Invalid thinking_mode `{thinking_mode}`"

    prompt = ""
    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    assert reasoning_effort in REASONING_EFFORT_PROMPTS
    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]

    if role == "system":
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(schema=to_json(response_format))

    elif role == "developer":
        content_developer = USER_SP_TOKEN + (content or "")
        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(schema=to_json(response_format))
        prompt += content_developer

    elif role == "user":
        prompt += USER_SP_TOKEN
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + (content or "")

    elif role == "tool":
        raise NotImplementedError("deepseek_v4 merges tool messages into user; "
                                  "preprocess with merge_tool_messages()")

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""
        if tool_calls:
            tc_list = [
                tool_call_template.format(dsml_token=dsml_token, name=tc.get("name"),
                                          arguments=encode_arguments_to_dsml(tc))
                for tc in tool_calls
            ]
            tc_content += '\n\n' + tool_calls_template.format(
                dsml_token=dsml_token, tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name)
        summary_content = content or ""
        rc = reasoning_content or ""
        if thinking_mode == "thinking":
            if not drop_thinking or index > last_user_idx:
                thinking_part = thinking_template.format(reasoning_content=rc) + thinking_end_token
        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part, content=summary_content, tool_calls=tc_content)
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part, content=summary_content, tool_calls=tc_content)
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in ["assistant", "latest_reminder"]:
        return prompt

    if role in ["user", "developer"]:
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt

# ============================================================
# Preprocessing (byte-exact)
# ============================================================

def merge_tool_messages(messages):
    merged = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")
        if role == "tool":
            tool_block = {"type": "tool_result",
                          "tool_use_id": msg.get("tool_call_id", ""),
                          "content": msg.get("content", "")}
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif role == "user":
            text_block = {"type": "text", "text": msg.get("content", "")}
            if (merged and merged[-1].get("role") == "user"
                    and "content_blocks" in merged[-1] and merged[-1].get("task") is None):
                merged[-1]["content_blocks"].append(text_block)
            else:
                merged.append({"role": "user", "content": msg.get("content", ""),
                               "content_blocks": [text_block]})
        else:
            merged.append(msg)
    return merged


def sort_tool_results_by_call_order(messages):
    last_tool_call_order = {}
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx
        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0))
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks
    return messages


def _drop_thinking_messages(messages):
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)
    return result


# ============================================================
# Main Encoding Function (byte-exact, minus "task"/context paths
# the gateway never produces)
# ============================================================

def encode_messages(messages, thinking_mode, context=None, drop_thinking=True,
                    add_default_bos_token=True, reasoning_effort=None):
    context = context if context else []
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context):]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages
    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(idx + context_len, full_messages,
                                 thinking_mode=thinking_mode,
                                 drop_thinking=effective_drop_thinking,
                                 reasoning_effort=reasoning_effort)
    return prompt


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
    """Tolerant gateway wrapper: split `content [+ blank line + DSML tool_calls block]`
    from a finished assistant turn (think already split off upstream). Never raises.

    Returns (content, tool_calls) where tool_calls use the gateway shape:
    [{"id": "call_...", "type": "function", "function": {"name", "arguments"}}].
    Malformed or truncated DSML degrades to plain content (the raw markers stay
    visible, like GLM salvage-off behaviour) and is logged to stderr.
    """
    cut = text.find(TOOL_CALLS_BLOCK_START)
    parse_at = cut + len(TOOL_CALLS_BLOCK_START) if cut >= 0 else -1
    if cut < 0 and text.startswith(TOOL_CALLS_PREFIX):
        cut, parse_at = 0, len(TOOL_CALLS_PREFIX)   # reply opening directly with the block
    if cut < 0:
        if TOOL_CALLS_OPEN in text or f"</{dsml_token}" in text:
            sys.stderr.write("[api] deepseek_v4: DSML markers present but no complete "
                             "tool_calls block -- treating as plain content\n")
            sys.stderr.flush()
        return text.strip(), []
    content = text[:cut]
    try:
        _index, _stop, calls = _parse_tool_calls_strict(parse_at, text)
    except ValueError as exc:
        sys.stderr.write(f"[api] deepseek_v4: DSML tool_calls parse failed ({exc}) "
                         "-- treating as plain content\n")
        sys.stderr.flush()
        return text.strip(), []
    shaped = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
               "function": {"name": c["name"], "arguments": c["arguments"]}}
              for c in calls]
    if shaped:
        sys.stderr.write("[api] deepseek_v4 tool-calls: %d [CLEAN]\n" % len(shaped))
        sys.stderr.flush()
    return content.strip(), shaped
