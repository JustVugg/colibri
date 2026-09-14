#!/usr/bin/env python3
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "coli"

class GLM53CapLaunchSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = SRC.read_text()

    def test_glm53_one_shot_uses_cap_for_launch(self):
        self.assertIn('cap = cap_for_launch(a.cap, e, 0)', self.s)

    def test_cap_is_positional_before_glm53_flags(self):
        self.assertIn('cmd=[engine,str(cap),"--model",os.path.abspath(a.model),"--prompt",prompt,', self.s)

    def test_old_cap_dropping_form_is_gone(self):
        self.assertNotIn('cmd=[engine,"--model",os.path.abspath(a.model),"--prompt",prompt,', self.s)

if __name__ == "__main__":
    unittest.main()
