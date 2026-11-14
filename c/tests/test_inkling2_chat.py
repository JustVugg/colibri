"""Inkling-2 on the inkling engine: its family, its name, and the chat its template renders.

Inkling-2's config says model_type inkling2 (inkling2_text in a flat snapshot or a GGUF import);
the inkling family serves it and calls it Inkling-2-Small. Its chat_template.jinja is not
Inkling's: the effort level before the first message that is not a system one, reasoning as a
message of its own, names after the role tokens, and no generation prompt (the model opens its
own turn). The expected strings below are what thinkingmachines/Inkling-2-Small's
chat_template.jinja renders for the same messages (checked with jinja2), plus the opened text
block render_chat_inkling2 adds when thinking is off.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gw
from family_registry import default_model_id as registry_default_model_id, display_for, resolve_model

EFFORT0 = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
OPEN = "<|message_model|><|content_text|>"


def render(messages, thinking=False, effort=None, **kw):
    return gw.render_chat_inkling2(messages, thinking, effort, **kw)


class Inkling2ChatTest(unittest.TestCase):
    def test_one_user_message_thinking_off_opens_the_text_block(self):
        self.assertEqual(render([{"role": "user", "content": "Hi"}]),
                         EFFORT0 + "<|message_user|><|content_text|>Hi<|end_message|>" + OPEN)

    def test_thinking_on_leaves_the_turn_to_the_model(self):
        self.assertEqual(
            render([{"role": "user", "content": "Hi"}], thinking=True),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>")
        self.assertEqual(
            render([{"role": "user", "content": "Hi"}], effort="xhigh"),
            "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_effort_after_the_leading_system_messages(self):
        self.assertEqual(
            render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], effort="high"),
            "<|message_system|><|content_text|>Be brief.<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_history_reasoning_names_and_parts(self):
        got = render([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "think"},
            {"role": "assistant", "name": "bob", "content": "", "reasoning_content": "only"},
            {"role": "tool", "name": "lookup", "content": "42"},
            {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
        ], effort="medium")
        self.assertEqual(
            got,
            "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>"
            "<|message_user|><|content_text|>a<|end_message|>"
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>b<|end_message|><|content_model_end_sampling|>"
            "<|message_model|>bob<|content_thinking|>only<|end_message|><|content_model_end_sampling|>"
            "<|message_tool|>lookup<|content_text|>42<|end_message|>"
            "<|message_user|><|content_text|>one<|end_message|>"
            "<|message_user|><|content_text|>two<|end_message|>")

    def test_continuing_an_assistant_turn_leaves_it_open(self):
        self.assertEqual(
            render([{"role": "user", "content": "a"}, {"role": "assistant", "content": "partial"}],
                   effort="none", add_generation_prompt=False),
            EFFORT0 + "<|message_user|><|content_text|>a<|end_message|>"
            "<|message_model|><|content_text|>partial")

    def test_a_picture_not_expanded_is_refused(self):
        # expand_inkling2_images turns a picture into its placeholders before rendering; one that
        # reaches the renderer as it came means the engine has no tower to read it
        with self.assertRaises(gw.APIError):
            render([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])

    def test_the_reply_splits_into_reasoning_and_answer(self):
        text, reasoning = gw.split_inkling(
            "<|message_model|><|content_thinking|>let me see<|end_message|>"
            "<|message_model|><|content_text|>Paris.<|end_message|>")
        self.assertEqual((text, reasoning), ("Paris.", "let me see"))
        self.assertEqual(gw.split_inkling2(
            "<|message_model|><|content_thinking|>let me see<|end_message|>"
            "<|message_model|><|content_text|>Paris.<|end_message|><|content_model_end_sampling|>"),
            ("Paris.", "let me see"))


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"unit": {"type": "string"}, "city": {"type": "string"}},
                   "required": ["city"]}}}
NOW = {"type": "function", "function": {"name": "now"}}
# the template's tojson: keys sorted, no spaces
DECL_WEATHER = ('{"description":"Weather for a city","name":"get_weather","parameters":{"properties":'
                '{"city":{"type":"string"},"unit":{"type":"string"}},"required":["city"],"type":"object"},'
                '"type":"function"}')
DECL_NOW = '{"description":"","name":"now","parameters":{},"type":"function"}'
CALL = ('<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"city":"Rome","opts":{"z":1,"a":[1,2]},"unit":"C"}}<|end_message|>')
# the args' own keys sorted (the template's dictsort), a value as tojson leaves it
ARGS = {"unit": "C", "city": "Rome", "opts": {"z": 1, "a": [1, 2]}}


def call(arguments, id_="call_1"):
    return {"id": id_, "type": "function", "function": {"name": "get_weather", "arguments": arguments}}


class Inkling2ToolsTest(unittest.TestCase):
    """The strings below are what Inkling-2-Small's chat_template.jinja renders, compiled as
    transformers compiles it (its tojson sorts the keys and drops the spaces)."""

    def test_the_declaration_comes_before_every_message(self):
        got = render([{"role": "system", "content": "S"}, {"role": "user", "content": "Weather?"}],
                     effort="high", tools=[WEATHER, NOW])
        self.assertEqual(
            got,
            "<|message_system|>tool_declare<|content_xml|>[" + DECL_WEATHER + "," + DECL_NOW + "]<|end_message|>"
            "<|message_system|><|content_text|>S<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Weather?<|end_message|>")

    def test_a_call_and_its_result_in_the_history(self):
        args = ARGS
        want = ("<|message_system|>tool_declare<|content_xml|>[" + DECL_WEATHER + "]<|end_message|>"
                "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
                "<|message_user|><|content_text|>Weather in Rome?<|end_message|>"
                + CALL + "<|content_model_end_sampling|>"
                "<|message_tool|>get_weather<|content_text|>21 C<|end_message|>")
        # OpenAI's wire form (arguments a JSON string) and the object the template takes
        for arguments in (json.dumps(args), args):
            got = render([{"role": "user", "content": "Weather in Rome?"},
                          {"role": "assistant", "content": "", "tool_calls": [call(arguments)]},
                          {"role": "tool", "tool_call_id": "call_1", "content": "21 C"}],
                         effort="low", tools=[WEATHER])
            self.assertEqual(got, want)

    def test_reasoning_and_text_before_a_call(self):
        got = render([{"role": "user", "content": "x"},
                      {"role": "assistant", "content": "Let me check.", "reasoning_content": "need it",
                       "tool_calls": [call(ARGS)]},
                      {"role": "tool", "name": "get_weather", "content": "21 C"}],
                     effort="medium", tools=[WEATHER])
        self.assertTrue(got.endswith(
            "<|message_user|><|content_text|>x<|end_message|>"
            "<|message_model|><|content_thinking|>need it<|end_message|>"
            "<|message_model|><|content_text|>Let me check.<|end_message|>"
            + CALL + "<|content_model_end_sampling|>"
            "<|message_tool|>get_weather<|content_text|>21 C<|end_message|>"), got)

    def test_with_tools_thinking_off_leaves_the_turn_to_the_model(self):
        # an opened text block never calls a tool: the turn is the model's, as the template has it
        got = render([{"role": "user", "content": "Weather in Rome?"}], tools=[WEATHER])
        self.assertTrue(got.endswith(EFFORT0 + "<|message_user|><|content_text|>Weather in Rome?<|end_message|>"),
                        got)
        got = render([{"role": "user", "content": "Weather in Rome?"}], tools=[WEATHER], tool_choice="none")
        self.assertTrue(got.endswith(OPEN), got)   # no tools offered: the text block, as without tools

    def test_tool_choice(self):
        msgs = [{"role": "user", "content": "What time is it?"}]
        instruction = ("<|message_system|><|content_text|>" + gw.TOOL_CHOICE_REQUIRED_INSTRUCTION.strip()
                       + "<|end_message|>")
        got = render(msgs, effort="low", tools=[WEATHER, NOW], tool_choice="required")
        self.assertTrue(got.startswith("<|message_system|>tool_declare<|content_xml|>[" + DECL_WEATHER + ","
                                       + DECL_NOW + "]<|end_message|>" + instruction), got)
        got = render(msgs, effort="low", tools=[WEATHER, NOW],
                     tool_choice={"type": "function", "function": {"name": "now"}})
        self.assertTrue(got.startswith("<|message_system|>tool_declare<|content_xml|>[" + DECL_NOW
                                       + "]<|end_message|>" + instruction), got)
        self.assertNotIn("tool_declare", render(msgs, effort="low", tools=[WEATHER], tool_choice="none"))
        with self.assertRaises(gw.APIError):
            render(msgs, tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}})
        with self.assertRaises(gw.APIError):
            render(msgs, tool_choice="required")

    def test_bad_calls_are_refused(self):
        for arguments in ("{not json", "[1, 2]"):
            with self.assertRaises(gw.APIError):
                render([{"role": "user", "content": "x"},
                        {"role": "assistant", "content": "", "tool_calls": [call(arguments)]}], tools=[WEATHER])
        with self.assertRaises(gw.APIError):   # a tool needs a name
            render([{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {}}])


REPLY = ("<|message_model|><|content_thinking|>I should look it up.<|end_message|>"
         "<|message_model|>get_weather<|content_invoke_tool_json|>"
         '{"name":"get_weather","args":{"city":"Rome"}}<|end_message|><|content_model_end_sampling|>')


class Inkling2ReplyTest(unittest.TestCase):
    def stream(self, reply, pieces, keep_calls=True):
        out, why, ends = [], [], []
        split = gw.Inkling2StreamSplit(out.append, why.append, lambda: ends.append(1), keep_calls=keep_calls)
        for k in range(0, len(reply), pieces):
            split.feed(reply[k:k + pieces])
        split.close()
        return "".join(out), "".join(why), len(ends)

    def test_reasoning_then_a_call(self):
        for pieces in (1, 3, 7, len(REPLY)):   # markers split across chunks anywhere
            content, reasoning, ends = self.stream(REPLY, pieces)
            self.assertEqual((reasoning, ends), ("I should look it up.", 1))
            text, calls = gw.parse_inkling2_tool_calls(content, [WEATHER])
            self.assertEqual(text, "")
            self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                             [("get_weather", {"city": "Rome"})])
            self.assertTrue(calls[0]["id"].startswith("call_"))

    def test_without_tools_a_call_is_dropped(self):
        self.assertEqual(self.stream(REPLY, 5, keep_calls=False), ("", "I should look it up.", 1))

    def test_text_and_a_call(self):
        reply = ("<|message_model|><|content_text|>Let me check.<|end_message|>"
                 "<|message_model|>get_weather<|content_invoke_tool_json|>"
                 '{"name":"get_weather","args":{"city":"Rome"}}<|end_message|>')
        text, calls = gw.parse_inkling2_tool_calls(gw.split_inkling2(reply, keep_calls=True)[0], [WEATHER])
        self.assertEqual((text, len(calls)), ("Let me check.", 1))

    def test_calls_that_cannot_be_served_are_dropped(self):
        cut = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"ci'
        other = '<|message_model|>rm<|content_invoke_tool_json|>{"name":"rm","args":{}}<|end_message|>'
        not_object = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":[1]}'
        for reply in (cut, other, not_object):
            content, _ = gw.split_inkling2(reply, keep_calls=True)
            self.assertEqual(gw.parse_inkling2_tool_calls(content, [WEATHER]), ("", []), reply)

    def test_text_that_looks_like_a_marker_is_kept(self):
        out = []
        split = gw.Inkling2StreamSplit(out.append, keep_calls=True)
        for piece in ("<|content_text|>a <", " b <|", "x and <", "|end_message|>"):
            split.feed(piece)
        split.close()
        self.assertEqual("".join(out), "a < b <|x and ")

    def test_the_gateway_split_follows_the_flavor(self):
        saved = gw.CHAT_FLAVOR
        try:
            gw.CHAT_FLAVOR = "inkling2"
            content, reasoning = gw.ink_split(REPLY, tools=[WEATHER])
            self.assertEqual(len(gw.parse_inkling2_tool_calls(content, [WEATHER])[1]), 1)
            self.assertEqual(gw.ink_split(REPLY), ("", "I should look it up."))
        finally:
            gw.CHAT_FLAVOR = saved


try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inkling2_image  # noqa: E402

PAD = (-1 / 255 - 0.5) / 0.5   # the processor's pad, -1 on the 0-255 scale, normalized


def png(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@unittest.skipUnless(np is not None and Image is not None, "Pillow and numpy")
class Inkling2ImageTest(unittest.TestCase):
    """tools/inkling2_image.py against Inkling2ImageProcessor's rules (transformers'
    models/inkling2/image_processing_inkling2.py): pad with -1 to whole 40-pixel tiles, x / 255,
    mean and std 0.5, the tiles in raster order, transparency onto an 8-pixel checkerboard."""

    def test_tiles_padding_and_order(self):
        rgb = np.random.default_rng(0).integers(0, 256, (50, 90, 3), dtype=np.uint8)
        tiles, rows, cols = inkling2_image.preprocess(png(Image.fromarray(rgb)), None)
        self.assertEqual((rows, cols, tiles.shape, tiles.dtype), (2, 3, (6, 40, 40, 3), np.float32))
        norm = (rgb.astype(np.float64) / 255 - 0.5) / 0.5
        np.testing.assert_allclose(tiles[0], norm[:40, :40], atol=1e-6)
        np.testing.assert_allclose(tiles[1], norm[:40, 40:80], atol=1e-6)       # raster: the next column
        np.testing.assert_allclose(tiles[3][:10], norm[40:50, :40], atol=1e-6)  # the second row
        np.testing.assert_allclose(tiles[3][10:], PAD, atol=1e-6)               # below the picture
        np.testing.assert_allclose(tiles[5][:10, :10], norm[40:50, 80:90], atol=1e-6)
        np.testing.assert_allclose(tiles[5][:, 10:], PAD, atol=1e-6)            # right of it

    def test_transparency_on_the_checkerboard(self):
        rgba = np.zeros((16, 40, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[0, 0, 3], rgba[0, 1, 3], rgba[0, 2, 3] = 0, 255, 128
        rgba[1:, :, 3] = 0
        tiles, rows, cols = inkling2_image.preprocess(png(Image.fromarray(rgba, "RGBA")), None)
        got = np.rint((tiles[0][:16] * 0.5 + 0.5) * 255).astype(int)
        self.assertEqual(got[0, 0].tolist(), [204, 204, 204])                  # transparent: the light square
        self.assertEqual(got[0, 8].tolist(), [153, 153, 153])                  # the next square is dark
        self.assertEqual(got[8, 8].tolist(), [204, 204, 204])
        self.assertEqual(got[0, 1].tolist(), [255, 0, 0])                      # opaque: itself
        self.assertEqual(got[0, 2].tolist(), [(255 * 128 + 204 * 127 + 127) // 255,
                                              (204 * 127 + 127) // 255, (204 * 127 + 127) // 255])

    def test_a_large_picture_is_scaled_to_the_ceiling(self):
        data = png(Image.new("RGB", (900, 500), (10, 20, 30)))
        tiles, rows, cols = inkling2_image.preprocess(data, None, max_tokens=64)
        self.assertLessEqual(rows * cols, 64)
        self.assertGreaterEqual(rows * cols, 40)       # scaled to fit, not far below
        self.assertEqual(tiles.shape[0], rows * cols)
        self.assertEqual(inkling2_image.preprocess(data, None)[1:], (13, 23))   # no ceiling: as it is

    def test_the_tile_from_the_config(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = Path(tmp.name) / "config.json"
        cfg.write_text(json.dumps({"vision_config": {"native_patch_size": 16, "adapter_merge_size": 2}}))
        self.assertEqual(inkling2_image.tile_of(tmp.name), 32)
        cfg.write_text(json.dumps({"vision_config": {"patch_size": 4, "spatial_merge_size": 2}}))
        self.assertEqual(inkling2_image.tile_of(tmp.name), 8)
        cfg.write_text("{}")
        self.assertEqual(inkling2_image.tile_of(tmp.name), 40)

    def test_the_gateway_renders_a_picture_as_its_placeholders(self):
        import base64
        url = "data:image/png;base64," + base64.b64encode(png(Image.new("RGB", (100, 60), (0, 0, 255)))).decode()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        rewritten, images = gw.expand_inkling2_images(msgs, None, 256)
        self.assertEqual([(t.shape, r, c) for t, r, c in images], [((6, 40, 40, 3), 2, 3)])
        self.assertEqual(
            render(rewritten, effort="low"),
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
            "<|message_user|><|content_text|>What is it?<|end_message|>"
            "<|message_user|><|content_image|>" + "<|unused_200054|>" * 6 + "<|end_message|>")
        with self.assertRaises(gw.APIError):   # the template takes pictures from users and tools only
            gw.expand_inkling2_images([{"role": "assistant", "content": msgs[0]["content"]}], None, 256)


class Inkling2FamilyTest(unittest.TestCase):
    def snapshot(self, config):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "config.json").write_text(json.dumps(config))
        return tmp.name

    def test_checkpoint_and_flat_configs(self):
        nested = self.snapshot({"model_type": "inkling2", "eos_token_id": 200006,
                                "text_config": {"hidden_size": 4608, "num_hidden_layers": 48}})
        flat = self.snapshot({"model_type": "inkling2_text", "hidden_size": 128})
        v1 = self.snapshot({"model_type": "inkling_mm_model", "text_config": {"model_type": "inkling"}})
        for d in (nested, flat):
            r = resolve_model(d)
            self.assertEqual(r.descriptor.id, "inkling")
            self.assertEqual(display_for(r), ("Inkling-2-Small", "594B"))
            self.assertEqual(registry_default_model_id(r), "inkling-2-small-colibri")
            self.assertEqual(gw.detect_chat_flavor("inkling", d), "inkling2")
        r = resolve_model(v1)
        self.assertEqual(display_for(r), ("Inkling", "975B"))
        self.assertEqual(registry_default_model_id(r), "inkling-colibri")
        self.assertIsNone(gw.detect_chat_flavor("inkling", v1))

    def test_the_gateway_renders_inkling2_for_an_inkling2_snapshot(self):
        saved = gw.ARCH, gw.CHAT_FLAVOR
        try:
            # thinking on: Inkling-2 ends at the user's message, Inkling hands the turn over
            gw.ARCH, gw.CHAT_FLAVOR = "inkling", "inkling2"
            got = gw.render_chat_for_arch([{"role": "user", "content": "Hi"}], True, None)
            self.assertTrue(got.endswith("Hi<|end_message|>"), got)
            gw.CHAT_FLAVOR = None
            got = gw.render_chat_for_arch([{"role": "user", "content": "Hi"}], True, None, audio_out=[])
            self.assertTrue(got.endswith("<|message_model|>"), got)
        finally:
            gw.ARCH, gw.CHAT_FLAVOR = saved


if __name__ == "__main__":
    unittest.main()
