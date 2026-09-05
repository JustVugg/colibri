"""Un tensore core mancante deve spiegarsi, non solo nominarsi (#1317).

Il reporter aveva 141 file su 142 e ha ricevuto una riga sola:
"missing tensor: model.embed_tokens.weight". La diagnostica che conta gli
shard, vede i buchi e suggerisce il resume ESISTEVA gia' (st_die_missing,
dal 2026-07-25) ma sette punti fatali di st.h uscivano con la riga nuda
senza chiamarla -- incluso il percorso dell'embed, cioe' il primo tensore
che ogni caricamento tocca: il caso piu' comune al mondo era l'unico senza
spiegazione.

Il container qui e' costruito a mano, byte per byte: niente numpy, niente
torch, nessun fixture esterno. Un config GLM tiny valido piu' uno shard che
NON contiene l'embed: il motore deve morire spiegando, non nominando.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("colibri.exe" if sys.platform == "win32" else "colibri")

TINY_GLM_CONFIG = json.loads('{"transformers_version":"5.12.1","architectures":null,"output_hidden_states":false,"return_dict":true,"dtype":null,"chunk_size_feed_forward":0,"is_encoder_decoder":false,"id2label":{"0":"LABEL_0","1":"LABEL_1"},"label2id":{"LABEL_0":0,"LABEL_1":1},"problem_type":null,"vocab_size":8192,"hidden_size":1024,"intermediate_size":2048,"moe_intermediate_size":512,"num_hidden_layers":8,"num_attention_heads":16,"num_key_value_heads":16,"n_shared_experts":1,"n_routed_experts":32,"routed_scaling_factor":2.5,"kv_lora_rank":128,"q_lora_rank":256,"qk_rope_head_dim":32,"v_head_dim":64,"qk_nope_head_dim":64,"n_group":1,"topk_group":1,"num_experts_per_tok":8,"norm_topk_prob":true,"hidden_act":"silu","max_position_embeddings":4096,"initializer_range":0.02,"rms_norm_eps":1e-05,"use_cache":true,"pad_token_id":null,"bos_token_id":0,"eos_token_id":1,"tie_word_embeddings":false,"rope_parameters":{"rope_type":"default","rope_theta":10000.0},"mlp_layer_types":["dense","dense","dense","sparse","sparse","sparse","sparse","sparse"],"attention_bias":false,"attention_dropout":0.0,"index_topk":4096,"index_head_dim":32,"index_n_heads":4,"mlp_bias":false,"num_experts":256,"head_dim":32,"first_k_dense_replace":3,"layer_types":["deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention","deepseek_sparse_attention"],"indexer_types":["full","full","full","full","full","full","full","full"],"qk_head_dim":96,"_name_or_path":"","model_type":"glm_moe_dsa","output_attentions":false}')


def write_safetensors(path, tensors):
    """Scrive un safetensors minimale: 8 byte di lunghezza header + JSON + dati."""
    header, blobs, off = {}, [], 0
    for name, values in tensors.items():
        blob = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": [len(values)],
                        "data_offsets": [off, off + len(blob)]}
        blobs.append(blob)
        off += len(blob)
    hdr = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr)))
        fh.write(hdr)
        for blob in blobs:
            fh.write(blob)


@unittest.skipUnless(ENGINE.exists(), "colibri is not built")
class MissingTensorDiagnosticsTest(unittest.TestCase):
    def test_missing_embed_explains_instead_of_naming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(TINY_GLM_CONFIG))
            # uno shard c'e', ma l'embed no: lo scenario esatto di #1317
            write_safetensors(root / "out-00001.safetensors",
                              {"padding.dummy": [0.0, 0.0, 0.0, 0.0]})
            r = subprocess.run([str(ENGINE), "4"], cwd=HERE, text=True,
                               capture_output=True, timeout=120,
                               env=dict(os.environ, SNAP=str(root),
                                        OMP_NUM_THREADS="2"))
            out = r.stdout + r.stderr
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("model.embed_tokens.weight", out)
            for fragment, why in (
                ("shards numbered", "non conta gli shard presenti"),
                ("core tensor", "non dice che il tensore e' irrinunciabile"),
                ("HF_HUB_DISABLE_XET", "non suggerisce il resume del download"),
            ):
                self.assertIn(fragment, out,
                              f"#1317 di nuovo: la morte per tensore mancante "
                              f"{why}. Output:\n{out[-800:]}")
            self.assertNotIn("missing tensor:", out,
                             "la riga nuda e' tornata: un sito fatale di st.h "
                             "non passa piu' da st_die_missing")


if __name__ == "__main__":
    unittest.main()
