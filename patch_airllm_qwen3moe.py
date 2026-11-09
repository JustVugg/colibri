"""Patch air_llm to add a Qwen3MoeForCausalLM-aware subclass with expert streaming.

Add to D:/Colibri and run from the project venv once after air_llm install, e.g.:
  .venv/Scripts/python.exe patch_airllm_qwen3moe.py

This is needed because air_llm 4.0.0 has no Qwen3MoeForCausalLM entry in ARCH_OVERRIDES,
and its generic AirLLMBaseModel path does not set expert_prefix, so per-expert streaming
of Qwen3Moe is not enabled and forward fails with
  AttributeError: 'Qwen3MoeExperts' object has no attribute '0'
when the first expert is selected.
"""
import sys
sys.path.insert(0, "D:/Colibri/.venv/Lib/site-packages")

from air_llm.auto_model import ARCH_OVERRIDES
from air_llm.airllm_base import AirLLMBaseModel
import air_llm

class AirLLMQwen3Moe(AirLLMBaseModel):
    """Qwen3MoE / Qwen3-Coder MoE (Qwen3MoeForCausalLM).

    Qwen3MoE lays out experts as
        model.layers.<n>.mlp.experts.<expert_idx>.<proj>.weight
    so we set expert_prefix to "mlp.experts" so AirLLM's per-expert streaming can find
    and stream individual experts instead of materialising a whole layer.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "model.embed_tokens",
            "layer_prefix": "model.layers",
            "norm": "model.norm",
            "lm_head": "lm_head",
            "expert_prefix": "mlp.experts",
        }


# Make AutoModel.from_pretrained dispatch to this class for Qwen3MoeForCausalLM.
ARCH_OVERRIDES["Qwen3MoeForCausalLM"] = "AirLLMQwen3Moe"
print("patched ARCH_OVERRIDES Qwen3MoeForCausalLM ->", ARCH_OVERRIDES["Qwen3MoeForCausalLM"])

# Also register on the air_llm package so the class is importable as air_llm.AirLLMQwen3Moe.
air_llm.AirLLMQwen3Moe = AirLLMQwen3Moe
print("registered air_llm.AirLLMQwen3Moe:", hasattr(air_llm, "AirLLMQwen3Moe"))
print("DONE")
