#!/usr/bin/env python3
"""The gateway's Mistral-Large-4 side: its chat format, its reasoning, its calls.

Mistral-Large-4 writes <s>, its tools and settings before the first user turn, [INST] turns,
reasoning in [THINK]...[/THINK] and calls as [TOOL_CALLS]name[ARGS]{json}. The template is not
vendored (its license travels with every copy): COLI_ML4_TEMPLATE=path/to/chat_template.jinja
(the checkpoint's, or the GGUF's tokenizer.chat_template) runs the byte-for-byte check against
it with jinja2, as transformers renders it.
"""
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import openai_server  # noqa: E402

try:
    import jinja2
    import jinja2.ext
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

TEMPLATE = os.environ.get("COLI_ML4_TEMPLATE")

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city, in °C.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}

CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "ciao"}]},
    "system and user": {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"}]},
    "two user turns merged": {"messages": [
        {"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
    "text parts": {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]},
    "history with reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "one and one"},
        {"role": "user", "content": "and 2+2?"}]},
    "history without reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "and 2+2?"}]},
    "tools": {"tools": [WEATHER, SEARCH], "messages": [{"role": "user", "content": "Weather in Rome?"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Weather in Rome and Milan for 3 days?"},
        {"role": "assistant", "content": "Checking both.", "reasoning_content": "two calls", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome", "days": 3}}},
            {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Milan\"}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sun, 24 C"},
        {"role": "tool", "tool_call_id": "b", "content": "rain, 17 C"},
        {"role": "assistant", "content": "Rome is sunny, Milan rainy."},
        {"role": "user", "content": "And Naples?"}]},
    "a call with no arguments": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "search", "arguments": ""}}]},
        {"role": "tool", "tool_call_id": "a", "content": "nothing"}]},
    "special text in a turn": {"messages": [{"role": "user", "content": "say [INST] and </s>\nnew line"}]},
}


def reference_render(case, effort=None, response_format=None):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                      extensions=["jinja2.ext.loopcontrols"])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(Path(TEMPLATE).read_text(encoding="utf-8"))
    kwargs = {"messages": case["messages"], "bos_token": "<s>", "eos_token": "</s>", "add_generation_prompt": True}
    if case.get("tools"):
        kwargs["tools"] = case["tools"]
    if effort:
        kwargs["reasoning_effort"] = effort
    if response_format:
        kwargs["response_format"] = response_format
    return template.render(**kwargs)


class MistralLarge4ChatTest(unittest.TestCase):

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "glm", "mistral_large4"

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_ML4_TEMPLATE (the release's chat template) and jinja2")
    def test_byte_identical(self):
        for name, case in CASES.items():
            for effort in (None, "low", "medium"):
                with self.subTest(name, effort=effort):
                    want = reference_render(case, effort)
                    got = openai_server.render_chat_mistral_large4(case["messages"], True, effort, case.get("tools"))
                    self.assertEqual(got, want)

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_ML4_TEMPLATE (the release's chat template) and jinja2")
    def test_a_response_format_is_a_setting(self):
        case = CASES["one user turn"]
        for rf in ({"type": "json_object"}, {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}):
            with self.subTest(rf=rf["type"]):
                self.assertEqual(openai_server.render_chat_mistral_large4(case["messages"], True, None, None,
                                                                          response_format=rf),
                                 reference_render(case, None, rf))

    def test_the_prompt_and_its_settings(self):
        got = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], True, None)
        self.assertEqual(got, '<s>[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS][INST]x[/INST]')
        low = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], False, None)
        self.assertIn('"reasoning_effort": "low"', low)

    def test_its_calls_parse(self):
        reply = 'Checking.[TOOL_CALLS]get_weather[ARGS]{"city": "Rome", "days": 3}[TOOL_CALLS]search[ARGS]{"query": "x"}'
        text, calls = openai_server.parse_arch_tool_calls(reply, [WEATHER, SEARCH])
        self.assertEqual(text, "Checking.")
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather", "search"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome", "days": 3})
        # an undeclared name, or arguments that are not one object, stay text
        self.assertEqual(openai_server.parse_arch_tool_calls('[TOOL_CALLS]nope[ARGS]{}', [WEATHER])[1], [])
        self.assertEqual(openai_server.parse_arch_tool_calls('[TOOL_CALLS]search[ARGS][1, 2]', [SEARCH])[1], [])
        self.assertEqual(openai_server._tool_stream_markers(), ("[TOOL_CALLS]",))

    def test_its_reasoning_reads_as_think(self):
        out = []
        feed = openai_server.ml4_think_filter(out.append)
        for piece in ("[TH", "INK]one and ", "one[/THI", "NK]2 [", "x"):
            feed(piece)
        feed("", True)
        self.assertEqual("".join(out), "<think>one and one</think>2 [x")
        self.assertFalse(openai_server.starts_in_reasoning(True))   # the answer opens it itself

    def test_the_flavor_comes_from_the_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({"model_type": "mistral_large4"}))
            self.assertEqual(openai_server.detect_chat_flavor("glm", directory), "mistral_large4")
            (Path(directory) / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
            self.assertIsNone(openai_server.detect_chat_flavor("glm", directory))

    def test_wrong_orders_are_refused(self):
        with self.assertRaises(openai_server.APIError):
            openai_server.render_chat_mistral_large4([{"role": "assistant", "content": "x"}], True)
        with self.assertRaises(openai_server.APIError):
            openai_server.render_chat_mistral_large4([{"role": "user", "content": "x"},
                                                      {"role": "tool", "content": "y"}], True)



class Ml4ImageTest(unittest.TestCase):
    """A picture for Mistral-Large-4: tools/mistral_image.py's patches (the stub below stands in for
    it), a row of [IMG] a merged token closed by [IMG_BREAK], the last by [IMG_END], the patch grid
    in the IMAGE frame, a ceiling on the tokens, the other parts untouched, and no expansion
    without a linked mmproj (the renderer then says what is missing)."""

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR, openai_server._preprocess_ml4_image)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "glm", "mistral_large4"
        self.ceilings = []

        def fake(data, model_dir, max_tokens=None):
            self.ceilings.append(max_tokens)
            return [0.0] * (4 * 6 * 588), 2, 3   # 2 x 3 merged tokens: 4 x 6 patches
        openai_server._preprocess_ml4_image = fake

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR, openai_server._preprocess_ml4_image = self.saved
        os.environ.pop("ML4_MAX_IMAGE_TOKENS", None)

    def expand(self, folder, messages=None):
        class Engine:
            model_dir = folder
            vision = True

        class Server:
            engine = Engine()

        class Handler:
            server = Server()
        url = "data:image/png;base64,iVBORw0KGgo="
        messages = messages or [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                             {"type": "text", "text": "What is it?"}]}]
        return openai_server.APIHandler.expand_images(Handler(), messages)

    def linked(self, folder):
        (Path(folder) / "gguf.json").write_text(json.dumps({"gguf": "x.gguf", "mmproj": "mmproj-BF16.gguf"}))

    def test_the_rows_of_a_picture(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            self.linked(folder)
            messages, image = self.expand(folder)
        parts = messages[0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "[IMG][IMG][IMG][IMG_BREAK][IMG][IMG][IMG][IMG_END]"})
        self.assertEqual(parts[1], {"type": "text", "text": "What is it?"})
        self.assertEqual((image[1], image[2]), (4, 6))   # the patch grid, as the engine's IMAGE frame takes it
        self.assertEqual(self.ceilings, [openai_server.ML4_MAX_IMAGE_TOKENS])
        prompt = openai_server.render_chat_mistral_large4(messages)
        self.assertIn("[INST][IMG][IMG][IMG][IMG_BREAK][IMG][IMG][IMG][IMG_END]What is it?[/INST]", prompt)

    def test_the_ceiling_comes_from_the_environment(self):
        import tempfile
        os.environ["ML4_MAX_IMAGE_TOKENS"] = "100"
        with tempfile.TemporaryDirectory() as folder:
            self.linked(folder)
            self.expand(folder)
        self.assertEqual(self.ceilings, [100])

    def test_other_turns_keep_their_parts(self):
        import tempfile
        url = "data:image/png;base64,iVBORw0KGgo="
        messages = [{"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "Hi"}]},
                    {"role": "user", "content": [{"type": "text", "text": "Look:"},
                                                 {"type": "image_url", "image_url": {"url": url}}]}]
        with tempfile.TemporaryDirectory() as folder:
            self.linked(folder)
            out, image = self.expand(folder, messages)
        self.assertEqual(out[:2], messages[:2])
        self.assertEqual(out[2]["content"][0], {"type": "text", "text": "Look:"})
        self.assertTrue(out[2]["content"][1]["text"].endswith("[IMG_END]"))

    def test_a_converted_container_carries_its_own_tower(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:   # tools/convert_fp8_to_int4.py: no gguf.json
            (Path(folder) / "mmproj.gguf").write_bytes(b"GGUF")
            messages, image = self.expand(folder)
        self.assertTrue(messages[0]["content"][0]["text"].endswith("[IMG_END]"))
        self.assertEqual((image[1], image[2]), (4, 6))

    def test_no_tower_no_expansion_and_a_clear_refusal(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "gguf.json").write_text(json.dumps({"gguf": "x.gguf"}))
            messages, image = self.expand(folder)
        self.assertIsNone(image)
        with self.assertRaises(openai_server.APIError) as caught:
            openai_server.render_chat_mistral_large4(messages)
        self.assertEqual(caught.exception.status, 400)
        self.assertIn("mmproj", str(caught.exception))


class Ml4ImageGeometryTest(unittest.TestCase):
    """tools/mistral_image.py's canvas and placeholder: the reference's resize (floor under the
    longest edge, then whole merged tokens) and our token ceiling, without Pillow."""

    def setUp(self):
        sys.path.insert(0, str(HERE.parent / "tools"))
        import mistral_image
        self.mi = mistral_image

    def test_whole_tokens_stay(self):
        self.assertEqual(self.mi.canvas(224, 336), (224, 336, 8, 12))

    def test_the_longest_edge_then_whole_tokens(self):
        # 3000 x 2000: ratio 3000 / 1540, floor to 1540 x 1026, then up to 28: 1540 x 1036
        self.assertEqual(self.mi.canvas(3000, 2000), (1540, 1036, 55, 37))

    def test_the_token_ceiling_shrinks_never_crops(self):
        h, w, gh, gw = self.mi.canvas(3000, 2000, max_tokens=256)
        self.assertLessEqual(gh * gw, 256)
        self.assertAlmostEqual(gh / gw, 55 / 37, delta=0.1)

    def test_the_placeholder(self):
        self.assertEqual(self.mi.placeholder(2, 3), "[IMG][IMG][IMG][IMG_BREAK][IMG][IMG][IMG][IMG_END]")
        self.assertEqual(self.mi.placeholder(1, 1), "[IMG][IMG_END]")

    def test_patches_in_the_tower_kernel_order(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("needs numpy")
        px = np.arange(3 * 28 * 28, dtype=np.float32).reshape(3, 28, 28)
        p = self.mi.patchify(px, np)
        self.assertEqual(p.shape, (4, 3 * 14 * 14))
        # patch 1 is row 0, column 1: channel, row, column of its 14 x 14 pixels
        self.assertTrue(np.array_equal(p[1], px[:, 0:14, 14:28].reshape(-1)))
        self.assertTrue(np.array_equal(p[2], px[:, 14:28, 0:14].reshape(-1)))


class Ml4MmprojImportTest(unittest.TestCase):
    """coli import links Mistral's mmproj (projector pixtral, under clip.projector_type, where
    LFM3's file says clip.vision.projector_type) and writes its geometry for the gateway."""

    def mmproj(self, folder, projector, key="clip.projector_type"):
        import struct
        sys.path.insert(0, str(HERE.parent / "tools"))
        from ml4_mmproj import _kv_bytes   # no torch: CI's Python job has none
        U32, F32, STR, ARR = 4, 6, 8, 9
        kv = [("general.architecture", STR, "clip"), (key, STR, projector),
              ("clip.vision.patch_size", U32, 14), ("clip.vision.spatial_merge_size", U32, 2),
              ("clip.vision.image_size", U32, 1540),
              ("clip.vision.image_mean", ARR, (F32, [0.5, 0.25, 0.125])),
              ("clip.vision.image_std", ARR, (F32, [0.25, 0.5, 0.75]))]
        path = Path(folder) / "mmproj-BF16.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + _kv_bytes(kv)[0])
        return path

    def test_pixtral_is_mistrals_and_lfm3vl_is_not(self):
        import tempfile
        import gguf_import
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(gguf_import.mmproj_of(self.mmproj(folder, "pixtral"), "mistral-large4"), (True, ""))
            ok, why = gguf_import.mmproj_of(self.mmproj(folder, "lfm3vl", "clip.vision.projector_type"), "mistral-large4")
            self.assertFalse(ok)
            self.assertIn("pixtral", why)

    def test_the_geometry_beside_the_link(self):
        import tempfile
        import gguf_import
        with tempfile.TemporaryDirectory() as folder:
            gguf_import.write_pixtral_preprocessor(folder, self.mmproj(folder, "pixtral"))
            conf = json.loads((Path(folder) / "preprocessor_config.json").read_text())
        self.assertEqual(conf, {"patch_size": 14, "spatial_merge_size": 2, "longest_edge": 1540,
                                "image_mean": [0.5, 0.25, 0.125], "image_std": [0.25, 0.5, 0.75]})


class Ml4MmprojFromReleaseTest(unittest.TestCase):
    """tools/ml4_mmproj.py: the release's tower tensors (Mistral's names, BF16) as the published
    mmproj has them (the matrices BF16 as they are, the norms, the patch kernel and the merger
    widened to F32, dims in ggml order, the clip keys from the release's configs), and the
    converter's stash written at the end, or kept with the command when a tensor is missing."""

    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("needs numpy")
        sys.path.insert(0, str(HERE.parent / "tools"))
        import ml4_mmproj
        self.np, self.mp = np, ml4_mmproj
        rng = np.random.default_rng(3)
        D, F, P, OH = 8, 12, 2, 16
        shapes = {"vision_encoder.patch_conv.weight": [D, 3, P, P], "vision_encoder.ln_pre.weight": [D],
                  "pre_mm_projector_norm.weight": [D], "patch_merger.merging_layer.weight": [D, D * 4],
                  "vision_language_adapter.w_in.weight": [OH, D], "vision_language_adapter.w_out.weight": [OH, OH]}
        for k, shape in (("attention.wq", [D, D]), ("attention.wk", [D, D]), ("attention.wv", [D, D]),
                         ("attention.wo", [D, D]), ("attention_norm", [D]), ("ffn_norm", [D]),
                         ("feed_forward.w1", [F, D]), ("feed_forward.w3", [F, D]), ("feed_forward.w2", [D, F])):
            shapes[f"vision_encoder.transformer.layers.0.{k}.weight"] = shape
        self.tensors = {n: ("BF16", sh, rng.integers(0x3000, 0x4000, int(np.prod(sh)), dtype=np.uint16).tobytes())
                        for n, sh in shapes.items()}
        self.config = {"model_type": "mistral_large4", "spatial_merge_size": 2,
                       "vision_config": {"hidden_size": D, "intermediate_size": F, "num_hidden_layers": 1,
                                         "num_attention_heads": 2, "patch_size": P, "image_size": 1540,
                                         "rope_parameters": {"rope_theta": 10000.0}}}
        self.processor = {"image_processor": {"image_mean": [0.5, 0.25, 0.125], "image_std": [0.25, 0.5, 0.75],
                                              "size": {"longest_edge": 1540}}}

    def read(self, path):
        from gguf_reader import GGUFReader
        with GGUFReader(str(path)) as r:
            return r.metadata, {t["name"]: (t["type_name"], list(t["dims"]), r.read_tensor(t)) for t in r.tensors}

    def test_the_published_layout(self):
        import tempfile
        np = self.np
        with tempfile.TemporaryDirectory() as folder:
            self.mp.write(Path(folder) / "mmproj.gguf", self.tensors, self.config, self.processor)
            meta, got = self.read(Path(folder) / "mmproj.gguf")
        self.assertEqual(len(got), 15)
        q = self.tensors["vision_encoder.transformer.layers.0.attention.wq.weight"][2]
        self.assertEqual(got["v.blk.0.attn_q.weight"], ("BF16", [8, 8], q))   # q's rows as they are
        self.assertEqual(got["v.blk.0.ffn_down.weight"][:2], ("BF16", [12, 8]))
        self.assertEqual(got["v.blk.0.ffn_gate.weight"][2],
                         self.tensors["vision_encoder.transformer.layers.0.feed_forward.w1.weight"][2])
        for name in ("v.patch_embd.weight", "v.pre_ln.weight", "v.blk.0.ln1.weight", "v.blk.0.ln2.weight",
                     "mm.input_norm.weight", "mm.patch_merger.weight"):
            self.assertEqual(got[name][0], "F32", name)
        wide = np.frombuffer(got["mm.patch_merger.weight"][2], dtype=np.float32)
        bits = np.frombuffer(self.tensors["patch_merger.merging_layer.weight"][2], dtype=np.uint16)
        self.assertTrue(np.array_equal(wide.view(np.uint32), bits.astype(np.uint32) << 16))
        self.assertEqual(got["v.patch_embd.weight"][1], [2, 2, 3, 8])
        self.assertEqual((meta["clip.projector_type"], meta["clip.vision.projection_dim"], meta["clip.vision.block_count"],
                          meta["clip.vision.attention.head_count"], meta["clip.vision.spatial_merge_size"]),
                         ("pixtral", 16, 1, 2, 2))
        self.assertEqual([round(x, 6) for x in meta["clip.vision.image_mean"]], [0.5, 0.25, 0.125])

    def test_the_converters_stash(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        names = sorted(self.tensors)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.hf.json").write_text(json.dumps(self.config))
            (Path(folder) / "processor_config.json").write_text(json.dumps(self.processor))
            self.mp.stash(Path(folder) / "_mmproj", "a", {n: self.tensors[n] for n in names[:7]})
            with redirect_stdout(io.StringIO()) as said:
                self.assertFalse(self.mp.finish(folder, "mistralai/X"))   # a shard of the tower missing
            self.assertIn("tools/ml4_mmproj.py --release mistralai/X", said.getvalue())
            self.assertTrue((Path(folder) / "_mmproj").is_dir())
            self.mp.stash(Path(folder) / "_mmproj", "b", {n: self.tensors[n] for n in names[7:]})
            with redirect_stdout(io.StringIO()):
                self.assertTrue(self.mp.finish(folder))
            self.assertFalse((Path(folder) / "_mmproj").exists())
            _meta, got = self.read(Path(folder) / "mmproj.gguf")
            conf = json.loads((Path(folder) / "preprocessor_config.json").read_text())
        self.assertEqual(len(got), 15)
        self.assertEqual(conf, {"patch_size": 2, "spatial_merge_size": 2, "longest_edge": 1540,
                                "image_mean": [0.5, 0.25, 0.125], "image_std": [0.25, 0.5, 0.75]})

    def test_names(self):
        self.assertTrue(self.mp.is_vision("vision_language_adapter.w_in.weight"))
        self.assertFalse(self.mp.is_vision("layers.0.attention.wq.weight"))
        self.assertEqual(self.mp.gguf_name("vision_encoder.transformer.layers.29.feed_forward.w3.weight"),
                         "v.blk.29.ffn_up.weight")
        self.assertIsNone(self.mp.gguf_name("vision_encoder.transformer.layers.0.attention.wq.bias"))


if __name__ == "__main__":
    unittest.main()
