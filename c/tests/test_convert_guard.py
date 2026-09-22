"""La guardia di famiglia del convertitore GLM-5.2 (#1304).

`coli convert` accetta qualunque --repo ma il convertitore serve una famiglia
sola. Su Qwen3.8 (BF16) i tensori non riconosciuti passavano upcastati a f32:
ogni shard raddoppiava, e il reporter di #1304 si e' fermato a 424 GB scritti.

Le due meta' del test:
- comportamento: GLM-5.2 passa, ogni altra famiglia viene rifiutata con
  l'indicazione giusta, l'ignoto viene rifiutato senza indovinare;
- contratto: gli insiemi della guardia sono copie di family_registry.py, e una
  copia che diverge e' il difetto che la guardia esiste per curare. Chi
  aggiunge o rinomina una famiglia fallisce qui, non su un disco pieno.
"""
import os
import sys
import unittest

try:
    import numpy  # noqa: F401 -- il modulo del convertitore lo importa in testa
except ImportError:
    raise unittest.SkipTest("numpy not installed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from convert_fp8_to_int4 import (  # noqa: E402
    GLM52_MODEL_TYPES,
    OTHER_FAMILY_PATHS,
    check_model_family,
)
from family_registry import FAMILIES  # noqa: E402


class ConvertGuardTest(unittest.TestCase):
    def test_glm52_passes(self):
        for model_type in sorted(GLM52_MODEL_TYPES):
            check_model_family({"model_type": model_type}, "x")  # nessuna eccezione

    def test_every_other_family_is_refused_with_its_own_pointer(self):
        for family in FAMILIES:
            if family.id == "glm":
                continue
            for model_type in family.model_types:
                with self.assertRaises(SystemExit) as caught:
                    check_model_family({"model_type": model_type}, "some/repo")
                message = str(caught.exception)
                self.assertIn(model_type, message)
                # il rifiuto deve dire DOVE andare, non solo no: e' la
                # differenza fra una guardia e un muro
                self.assertTrue("tools/" in message or "docs/" in message or
                                "FP8" in message,
                                f"{model_type}: refusal has no pointer:\n{message}")

    def test_qwen38_refusal_says_no_conversion_exists(self):
        with self.assertRaises(SystemExit) as caught:
            check_model_family({"model_type": "qwen4_exp"}, "Qwen/Qwen3.8-Flash-Next")
        self.assertIn("NOT converted", str(caught.exception))
        self.assertIn("FP8", str(caught.exception))

    def test_unknown_and_missing_types_are_refused_not_guessed(self):
        for config in ({"model_type": "llama"}, {"model_type": ""}, {}):
            with self.assertRaises(SystemExit):
                check_model_family(config, "x")

    def test_guard_sets_match_the_registry(self):
        """Il contratto anti-divergenza: la guardia deve conoscere ESATTAMENTE
        le famiglie del registry. Se una famiglia nuova arriva senza una voce
        qui, il suo checkpoint passerebbe alla via 'ignoto' con un messaggio
        generico invece del puntatore giusto -- non un disastro, ma il test lo
        segnala subito; se GLM-5.2 rinomina un model_type, la guardia
        rifiuterebbe il modello che esiste per servire, e questo E' un
        disastro."""
        glm = next(f for f in FAMILIES if f.id == "glm")
        self.assertEqual(GLM52_MODEL_TYPES, set(glm.model_types),
                         "la guardia e family_registry non concordano su cosa "
                         "sia GLM-5.2: il convertitore rifiuterebbe il suo "
                         "stesso modello")
        others = {mt for f in FAMILIES if f.id != "glm" for mt in f.model_types}
        self.assertEqual(others, set(OTHER_FAMILY_PATHS),
                         "una famiglia del registry non ha la sua voce nella "
                         "guardia (o la guardia ne elenca una rimossa): il "
                         "puntatore per-famiglia non verrebbe mostrato")


if __name__ == "__main__":
    unittest.main()
