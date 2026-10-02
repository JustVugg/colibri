"""The image family from the registry up to the commands a user types.

Detection (model_index.json, no config.json), dispatch (engine, tokenizer),
and every entry point the project rule names: `coli info`, `coli plan`,
`coli run`, `coli chat` in both modes, and `coli doctor` not crashing. The
engine is tools/qwenimage_stub.py, reached through COLI_ENGINE exactly as a
packaged engine would be.
"""
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_image_fixture import (STUB, TENSORS, elements, expected_resident,  # noqa: E402
                                make_fake_pipeline, stub_env)

import image_engine  # noqa: E402
from family_registry import (FamilyConfigError, UnknownFamilyError,  # noqa: E402
                             family_by_id, family_for_config, resolve_model)

C_DIR = Path(__file__).resolve().parent.parent
CLI = C_DIR / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_image_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class FamilyDetectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_model_index_without_config_is_the_image_family(self):
        model = make_fake_pipeline(self.root / "m")
        resolved = resolve_model(model)
        family = resolved.descriptor
        self.assertEqual(family.id, "qwen_image")
        self.assertEqual(family.modality, "image")
        self.assertEqual(family.engine_artifact, "qwenimage")
        self.assertEqual(family.default_model_id, "qwen-image-2.1-colibri")
        self.assertEqual(family.display_name, "Qwen-Image-2.1")
        self.assertEqual(family.tokenizer_file, "processor/tokenizer.json")
        self.assertEqual(resolved.model_type, "qwenimage21pipeline")
        self.assertEqual(family_for_config({"_class_name": "QwenImage21Pipeline"}).id,
                         "qwen_image")

    def test_config_json_wins_over_an_index(self):
        model = make_fake_pipeline(self.root / "m")
        (Path(model) / "config.json").write_text(json.dumps({"model_type": "olmoe"}))
        self.assertEqual(resolve_model(model).descriptor.id, "olmoe")
        self.assertFalse(image_engine.is_image_model_dir(model))

    def test_unknown_or_broken_indexes_are_refused(self):
        model = self.root / "m"
        model.mkdir()
        (model / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
        with self.assertRaisesRegex(UnknownFamilyError, "FluxPipeline"):
            resolve_model(model)
        (model / "model_index.json").write_text("{not json")
        with self.assertRaises(FamilyConfigError):
            resolve_model(model)
        (model / "model_index.json").write_text(json.dumps({"no": "class"}))
        with self.assertRaises(FamilyConfigError):
            resolve_model(model)
        # A text model_type is not a pipeline class.
        (model / "model_index.json").write_text(json.dumps({"_class_name": "olmoe"}))
        with self.assertRaises(UnknownFamilyError):
            resolve_model(model)

    def test_no_config_and_no_index_names_both_files(self):
        with self.assertRaisesRegex(FamilyConfigError, "model_index.json"):
            resolve_model(self.root)


class LauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = make_fake_pipeline(Path(self.tmp.name) / "m")

    def test_dispatch(self):
        self.assertEqual(self.cli.model_arch(self.model), "qwen_image")
        engine = os.path.basename(self.cli.engine_for(self.model))
        self.assertEqual(engine, "qwenimage" + self.cli._EXE)
        self.assertTrue(self.cli.is_image_model(self.model))
        self.assertIn("text-to-image", self.cli.model_banner_line(self.model))

    def test_need_model_wants_the_processor_tokenizer(self):
        os.remove(Path(self.model) / "processor" / "tokenizer.json")
        with self.assertRaises(SystemExit) as caught:
            self.cli.need_model(self.model, engine=str(STUB))
        self.assertIn("processor/tokenizer.json", str(caught.exception))
        (Path(self.model) / "processor" / "tokenizer.json").write_text("{}")
        self.cli.need_model(self.model, engine=str(STUB))      # no exit
        # and a root tokenizer.json is not what an image model is asked for
        self.assertFalse((Path(self.model) / "tokenizer.json").exists())

    def test_ctx_is_refused_for_an_image_model(self):
        class Args:
            ctx = 4096
            ram = 0
            temp = None
            model = self.model

            def __getattr__(self, name):
                return None
        with self.assertRaises(SystemExit) as caught:
            self.cli.env_for_engine(Args(), "qwen_image")
        self.assertIn("--ctx", str(caught.exception))

    def test_image_commands_share_the_completion_machinery(self):
        self.assertEqual(set(self.cli.IMAGE_CHAT_COMMANDS),
                         {"size", "steps", "seed", "render", "save", "help", "quit"})
        self.assertEqual(self.cli.chat_command("/size 1024x576"), ("size", "1024x576"))
        # a picture's path at the start of a line is a message, not a command
        self.assertEqual(self.cli.chat_command("/home/me/foto.png cosa vedi?"), (None, None))
        self.assertEqual(self.cli.chat_command("~/Desktop/a.JPG"), (None, None))
        self.assertEqual(self.cli.chat_command("/brio a | b"), ("brio", "a | b"))
        self.assertEqual(self.cli.chat_command("/save /tmp/x/nome.png"), ("save", "/tmp/x/nome.png"))


def run_coli(args, env, stdin=None, timeout=60):
    return subprocess.run([sys.executable, str(CLI), *args], env=env, input=stdin,
                          capture_output=True, text=True, timeout=timeout, cwd=str(C_DIR))


class CommandsTest(unittest.TestCase):
    """The commands, as a user runs them, against the stub engine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = make_fake_pipeline(Path(self.tmp.name) / "qwen-image")
        self.images = Path(self.tmp.name) / "images"
        self.env = stub_env(COLI_ENGINE=str(STUB), COLI_IMAGE_DIR=str(self.images))
        for key in ("COLI_MODEL", "COLI_COLOR", "COLI_API_KEY"):
            self.env.pop(key, None)

    def test_info_describes_the_pipeline(self):
        result = run_coli(["info", "--model", self.model], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        for needle in ("Qwen-Image-2.1", "text-to-image", "QwenImage21Transformer2DModel",
                       "Qwen3VLForConditionalGeneration", "AutoencoderKLQwenImage21",
                       "processor/tokenizer.json", str(self.images), "Research License",
                       "/v1/images/generations"):
            self.assertIn(needle, out)
        self.assertNotIn("expert/layer", out)

    def test_plan_gives_both_schedules(self):
        result = run_coli(["plan", "--model", self.model, "--json"], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["kind"], "image")
        parts = {c["name"]: c for c in plan["components"]}
        te, dit, vae = (expected_resident(n) for n in ("text_encoder", "transformer", "vae"))
        self.assertEqual(parts["text_encoder"]["resident_bytes"], te)
        self.assertEqual(parts["transformer"]["resident_bytes"], dit)
        self.assertEqual(parts["vae"]["resident_bytes"], vae)
        self.assertEqual(plan["modes"]["resident"]["peak_bytes"], te + dit + vae)
        self.assertEqual(plan["modes"]["text_encoder_on_demand"]["peak_bytes"],
                         max(te, dit + vae))
        self.assertEqual(plan["modes"]["text_encoder_on_demand"]["idle_bytes"], dit + vae)
        vision_and_head = sum(elements(shape) for name, _dtype, shape in TENSORS["text_encoder"]
                              if not name.startswith("model.language_model."))
        self.assertEqual(parts["text_encoder"]["unloaded_params"], vision_and_head)
        text = run_coli(["plan", "--model", self.model], self.env)
        self.assertEqual(text.returncode, 0, text.stderr)
        for needle in ("all resident", "text encoder on demand", "int8", "working buffers"):
            self.assertIn(needle, text.stdout)
        self.assertNotIn("experts", text.stdout)
        refused = run_coli(["plan", "--model", self.model, "--ctx", "4096"], self.env)
        self.assertNotEqual(refused.returncode, 0)

    def test_doctor_runs_the_image_checks(self):
        result = run_coli(["doctor", "--model", self.model, "--json"], self.env)
        report = json.loads(result.stdout)
        ids = {check["id"]: check["status"] for check in report["checks"]}
        self.assertEqual(ids["model.family"], "pass")
        self.assertEqual(ids["model.tokenizer"], "pass")
        self.assertEqual(ids["model.components"], "pass")
        self.assertNotIn("model.config", ids)
        self.assertIsNotNone(report["image_plan"])

    def test_run_writes_the_png(self):
        out = Path(self.tmp.name) / "fox.png"
        result = run_coli(["run", "--model", self.model, "--size", "512x288", "--steps", "2",
                           "--seed", "5", "--out", str(out), "a red fox"], self.env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], str(out))
        width, height, channels, _px = image_engine.decode_png(out.read_bytes())
        self.assertEqual((width, height, channels), (512, 288, 4))
        refused = run_coli(["run", "--model", self.model, "--size", "768x432", "x"], self.env)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("multiple of 32", refused.stderr + refused.stdout)

    def test_chat_private_mode(self):
        script = "/size 256x256\n/steps 2\n/seed 7\n/bogus\na lighthouse at dusk\n/size\n:q\n"
        result = run_coli(["chat", "--model", self.model, "--no-attach"], self.env,
                          stdin=script)
        self.assertEqual(result.returncode, 0, result.stderr)
        saved = sorted(self.images.glob("*.png"))
        self.assertEqual(len(saved), 1, result.stdout)
        self.assertTrue(saved[0].name.endswith("-a-lighthouse-at-dusk-s7.png"), saved[0].name)
        self.assertEqual(image_engine.decode_png(saved[0].read_bytes())[:2], (256, 256))
        self.assertIn("unknown command", result.stdout)
        self.assertIn("1024x576", result.stdout)             # /size lists the presets
        self.assertIn("seed 7", result.stdout)

    def test_chat_save_copies_the_last_image(self):
        # The picture is always saved in the images folder; /save adds a copy
        # where the user says: a folder keeps the name, a bare name gets .png,
        # and an existing file is never overwritten.
        import tempfile
        with tempfile.TemporaryDirectory() as elsewhere:
            named = Path(elsewhere) / "poster"
            script = (f"/save {named}\n"                            # nothing yet
                      "/size 256x256\n/steps 2\n/seed 9\nan owl on a branch\n"
                      f"/save\n/save {named}\n/save {named}\n/save {elsewhere}\n"
                      f"/save {elsewhere}/missing/x.png\n:q\n")
            result = run_coli(["chat", "--model", self.model, "--no-attach"], self.env,
                              stdin=script)
            self.assertEqual(result.returncode, 0, result.stderr)
            auto = sorted(self.images.glob("*.png"))
            self.assertEqual(len(auto), 1, result.stdout)
            self.assertIn("no image yet", result.stdout)
            self.assertIn(f"the last image is at {auto[0]}", result.stdout)
            copy = named.with_suffix(".png")
            self.assertEqual(copy.read_bytes(), auto[0].read_bytes())
            self.assertIn("already exists", result.stdout)
            self.assertEqual((Path(elsewhere) / auto[0].name).read_bytes(), auto[0].read_bytes())
            self.assertIn("no such folder", result.stdout)

    def test_chat_attached_to_a_running_serve(self):
        from openai_server import APIServer
        engine = image_engine.ImageEngine(STUB, self.model, env=stub_env(),
                                          on_log=lambda _m: None)
        server = APIServer(("127.0.0.1", 0), engine, "qwen-image-2.1-colibri", None, 16)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            result = run_coli(["chat", "--attach", base], self.env,
                              stdin="/size 512x256\n/seed 9\na paper boat\n:q\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("attached", result.stdout)
            saved = sorted(self.images.glob("*.png"))
            self.assertEqual(len(saved), 1, result.stdout)
            self.assertTrue(saved[0].name.endswith("-s9.png"))
            self.assertEqual(image_engine.decode_png(saved[0].read_bytes())[:2], (512, 256))
        finally:
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            engine.close(timeout=5)

    def test_serve_front_door(self):
        """openai_server.py's own main() on an image model: spawns the image
        engine and answers the images endpoint."""
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = dict(self.env)
        # The log goes to a file, not to a pipe nobody reads: a full pipe blocks
        # the gateway mid-write, and macOS pipes hold a quarter of Linux's.
        log = open(Path(self.tmp.name) / "serve.log", "w+b")
        self.addCleanup(log.close)
        process = subprocess.Popen(
            [sys.executable, str(C_DIR / "openai_server.py"), "--model", self.model,
             "--engine", str(STUB), "--port", str(port)],
            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(C_DIR))
        try:
            import time
            from urllib.request import Request, urlopen
            deadline = time.time() + 120           # a loaded CI runner, not a slow engine
            while True:
                try:
                    with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                        entry = json.loads(r.read())["data"][0]
                    break
                except OSError:
                    if time.time() > deadline or process.poll() is not None:
                        log.seek(0)
                        print(log.read()[-3000:].decode("utf-8", "replace"), file=sys.stderr)
                        raise
                    time.sleep(0.2)
            self.assertEqual(entry["id"], "qwen-image-2.1-colibri")
            self.assertEqual(entry["capabilities"], ["image_generation"])
            body = json.dumps({"model": entry["id"], "prompt": "x", "size": "256x256",
                               "steps": 2}).encode()
            req = Request(f"http://127.0.0.1:{port}/v1/images/generations", data=body,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as r:
                self.assertEqual(json.loads(r.read())["colibri"]["width"], 256)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
