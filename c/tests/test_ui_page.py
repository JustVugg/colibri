"""Static checks of the chat page's Monitor (ui/index.html): no browser, the HTML read as text.

The page has no build step and no JS test runner, so these pin what a regression would break
first: the ids the Monitor's renderers write to, an English entry for every Italian string the
GPU/Metal additions show, and that the renderers tolerate a /health without `metal` (an engine
that sent no METAL frame) or without `system.gpus` (Linux without nvidia-smi).
"""
import json
import re
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[2] / "ui" / "index.html"

# Every Italian string the GPU/Metal additions to the Monitor show (labels, sub-labels, legend,
# tooltips); numbers are {0}, {1}, ... and free text {s}, as the page's tr() looks them up.
NEW_STRINGS = (
    "GPU", "CPU", "Cache esperti", "{0} GB in uso", "nessun dato Metal",
    "{0} caricati", "{0} trovati in cache", "uso %, ultimi {0} minuti",
    "Motore GPU", "Catena Metal accesa", "spenta: {s}", "Catena Metal spenta: {s}",
    "Riuso della cache", "Lettura anticipata", "Attesa SSD",
    "{0} slot", "{0} giuste su {1}", "{0} s su {1} s del turno", "spenta",
    "nessun turno ancora", "passi sulla GPU", "rifiutati", "KV ricresciuto",
    "esperti sulla GPU", "esperti sulla CPU", "dati del turno appena finito",
    "totale da quando gira",
)


def page():
    return PAGE.read_text(encoding="utf-8")


def i18n_en(txt):
    line = next(l for l in txt.splitlines() if l.startswith("const I18N="))
    return json.loads(line[len("const I18N="):].rstrip().rstrip(";"))["en"]


def js_function(txt, name):
    """The text of `function name(...){...}` up to the next top-level function."""
    m = re.search(r"\nfunction " + re.escape(name) + r"\(", txt)
    assert m, name
    nxt = re.search(r"\n(?:async )?function \w+\(", txt[m.end():])
    return txt[m.start():m.end() + (nxt.start() if nxt else len(txt))]


class MonitorPageTest(unittest.TestCase):
    def test_monitor_has_gpu_cells(self):
        txt = page()
        for i in ("lvGpu", "lvGpuSub", "lvCache", "lvCacheSub", "metalCard", "stEngine", "lvLegend"):
            self.assertRegex(txt, r'id="%s"' % i)
        mon = txt[txt.index('id="view-monitor"'):txt.index('id="mon-prof"')]
        self.assertIn('id="lvGpu"', mon)
        self.assertIn('id="metalCard"', mon)

    def test_new_strings_translated(self):
        en = i18n_en(page())
        missing = [s for s in NEW_STRINGS if s not in en or not en[s]]
        self.assertEqual(missing, [])

    def test_new_strings_used(self):
        txt = page()
        for s in NEW_STRINGS:
            probe = s.split("{")[0].strip() or s.split("}")[-1].strip()
            self.assertIn(probe, txt.split("const I18N=")[0] + txt.split("const I18N=")[1].split("\n", 1)[1], s)

    def test_render_tolerates_missing_metal(self):
        txt = page()
        mon = js_function(txt, "renderMonitor") + js_function(txt, "renderLive") + js_function(txt, "renderMetal")
        self.assertIn("h.metal", mon)
        self.assertIn("renderMetal(", js_function(txt, "renderMonitor"))
        # h.metal is read once behind a guard; no bare property access on it
        self.assertNotRegex(mon, r"h\.metal\.")
        self.assertRegex(js_function(txt, "renderMetal"), r"if\(!M\)")
        # system.gpus may be absent (Linux without nvidia-smi, a failed ioreg)
        self.assertIn("gpus", mon)
        self.assertNotRegex(mon, r"\.gpus\[")
        self.assertRegex(mon, r"\.gpus\|\|\[\]")

    def test_metal_values_coerced(self):
        """The gateway sends numbers, but a field may be missing or a string: coerced in JS."""
        self.assertRegex(js_function(page(), "renderMetal"), r"Number\(")

    def test_page_has_no_external_requests(self):
        """The page must work fully offline: no href/src/url()/@import pointing at http(s)."""
        html = re.sub(r"<!--.*?-->", "", page(), flags=re.S)
        pats = (r"""(?:href|src)\s*=\s*["']?\s*https?://""", r"""url\(\s*["']?\s*https?://""",
                r"""@import\s+(?:url\()?\s*["']?\s*https?://""",
                r"""fetch\(\s*["'`]https?://""", r"""new URL\(\s*["'`]https?://""")
        for p in pats:
            hits = [m.group(0) for m in re.finditer(p, html)]
            self.assertEqual(hits, [], "external request: " + p)



if __name__ == "__main__":
    unittest.main()
