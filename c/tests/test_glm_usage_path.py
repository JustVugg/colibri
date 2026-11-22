"""#1078: the GLM-5.2 engine (colibri.c) honours COLI_USAGE, and its autopin line
reports the RAM pin it actually loads.

colibri.c built <model>/.coli_usage unconditionally, for the history load, for PIN=auto
and for every usage_save(), so COLI_USAGE (docs/ENVIRONMENT.md; set per run by
tools/usage_start_profile.py) was silently ignored and the model directory's history
was read and rewritten instead. And the "[PIN] auto: ... -> pinning X GB" line printed
the clamped plan before the 0.5 GB floor that then skipped the RAM pin. Both live in
main(), which needs a model to run, so this checks the source, like
test_glm53_telemetry.py does for glm53."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GLM = (ROOT / "colibri.c").read_text(encoding="utf-8")
TELEMETRY = (ROOT / "telemetry.h").read_text(encoding="utf-8")


def startup():
    """main()'s pin and usage-history block, up to the LRU clamp."""
    start = GLM.index('gguf_experts_random(&m);')
    end = GLM.index('cap_for_ram(&m, ram_env, ebits, est_ctx);', start)
    return GLM[start:end]


class GlmUsagePathTest(unittest.TestCase):
    def test_coli_usage_names_the_history(self):
        block = startup()
        pin = block.index('if(getenv("PIN")){')
        self.assertIn('getenv("COLI_USAGE")', block[:pin])
        # <model>/.coli_usage is only the fallback, resolved once before PIN=auto
        self.assertEqual(GLM.count('"%s/.coli_usage",snap'), 1)
        fallback = block.index('"%s/.coli_usage",snap')
        self.assertLess(block.index('getenv("COLI_USAGE")'), fallback)
        self.assertLess(fallback, pin)
        self.assertEqual(block.count('snprintf(g_usage_path'), 2)
        self.assertLess(block.rindex('snprintf(g_usage_path'), pin)

    def test_pin_auto_seeds_from_the_same_history(self):
        block = startup()
        start = block.index('if(!strcmp(pin,"auto")){')
        end = block.index('if(pin){', start)
        auto = block[start:end]
        self.assertRegex(auto, r'snprintf\(pauto,sizeof\(pauto\),"%s",g_usage_path\);')
        self.assertNotIn('.coli_usage",snap', auto)

    def test_load_autopin_and_save_use_it(self):
        block = startup()
        self.assertIn('usage_load(&m,g_usage_path)', block)
        self.assertIn('pin_load(&m, g_usage_path, pin_gb, 0);', block)
        self.assertIn('static void usage_save(Model *m){ if(g_usage_path[0]) '
                      'stats_dump_q(m,g_usage_path,1); }', TELEMETRY)

    def test_autopin_line_reports_the_floored_pin(self):
        block = startup()
        start = block.index('double pin_bytes=autopin_preserve_lru(')
        end = block.index('else if(vram_tier) pin_load(', start)
        autopin = block[start:end]
        gate = re.search(r'if\(pin_gb>=([0-9.]+)\) pin_load\(&m, g_usage_path, pin_gb, 0\);', autopin)
        self.assertIsNotNone(gate, "the RAM pin gate moved: update this test with it")
        floor = gate.group(1)
        line = autopin.index('fprintf(stderr,"[PIN] auto: budget')
        call = autopin[line:autopin.index(');', line)]
        self.assertIn('-> pinning %.1f GB', call)
        self.assertLess(autopin.index('double pin_gb=pin_bytes/1e9;'), line)
        # the printed figure is the one the gate passes on: 0 under the floor
        self.assertIn('pin_gb>=%s ? pin_gb : 0.0' % floor, call)
        self.assertNotRegex(call, r',\s*pin_bytes/1e9,')
        # and the line says why, with the dropped pin in whole MB rounded down, so that it
        # never prints as the floor itself (499 MB, not "0.50 GB")
        tag = re.search(r'if\(pin_bytes>0\.0&&pin_gb<([0-9.]+)\)\s*snprintf\(under_floor,'
                        r'sizeof under_floor,"  \[a %\.0f MB pin is under the ([0-9.]+) GB floor: '
                        r'no RAM pin\]",\s*floor\(pin_bytes/1e6\)\);', autopin[:line])
        self.assertIsNotNone(tag, "the line no longer says the pin fell under the floor")
        self.assertEqual((tag.group(1), tag.group(2)), (floor, floor))
        self.assertRegex(call, r',\s*under_floor\s*$', "the tag is not what the line prints last")


if __name__ == '__main__':
    unittest.main()
