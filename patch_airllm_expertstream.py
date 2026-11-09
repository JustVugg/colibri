"""Monkeypatch air_llm so expert streaming works for Qwen3MoeExperts.

Qwen3MoeExperts stores experts as 3D tensors and has no __len__, so the guard
    if expert_idx >= len(experts_container):
in air_llm's _setup_expert_streaming TypeError's. This patch replaces that guard
with an attribute-based check (num_experts) when present, falling back to len()
for module-list containers (Kimi K3 style).

Run once in the project venv:
  .venv/Scripts/python.exe patch_airllm_expertstream.py
"""
import sys
sys.path.insert(0, "D:/Colibri/.venv/Lib/site-packages")

from air_llm.airllm_base import AirLLMBaseModel
import types

_src = """
def _setup_expert_streaming(self):
    self._expert_streaming = False
    self._expert_keys = {}
    self._non_expert_keys = {}

    expert_prefix = self.layer_names_dict.get('expert_prefix')
    if not expert_prefix:
        return
    if type(ModelPersister.get_model_persister()).__name__ != 'SafetensorModelPersister':
        return

    layer_prefix = self.layer_names_dict['layer_prefix']
    hooked = 0

    for idx in self._streamed_indices:
        layer_name = self.layer_names[idx]
        if not layer_name.startswith(layer_prefix + '.'):
            continue
        try:
            names = layer_tensor_names(self.checkpoint_path, layer_name)
        except Exception:
            continue

        marker = f'.{expert_prefix}.'
        per_expert = {}
        others = []
        for key in names:
            pos = key.find(marker)
            if pos == -1:
                others.append(key)
                continue
            rest = key[pos + len(marker):]
            head = rest.split('.', 1)[0]
            if not head.isdigit():
                others.append(key)
                continue
            per_expert.setdefault(int(head), []).append(key)

        if not per_expert:
            continue

        layer_module = self.layers[idx]
        experts_container = layer_module
        try:
            for attr in expert_prefix.split('.'):
                experts_container = getattr(experts_container, attr)
        except AttributeError:
            continue

        self._non_expert_keys[idx] = others
        self._expert_keys[idx] = per_expert

        for expert_idx, keys in per_expert.items():
            # Qwen3MoeExperts has no __len__; use num_experts attribute when present.
            num_experts = getattr(experts_container, 'num_experts', None)
            if num_experts is not None and expert_idx >= num_experts:
                continue
            if num_experts is None and expert_idx >= len(experts_container):
                continue
            expert_module = experts_container[expert_idx]
            expert_module._airllm_expert = (idx, expert_idx)
            expert_module.register_forward_pre_hook(self._expert_pre_hook)
            expert_module.register_forward_hook(self._expert_post_hook)
            hooked += 1

    if hooked:
        self._expert_streaming = True
        n_layers = len(self._expert_keys)
        print(f"per-expert streaming enabled: {hooked} experts across {n_layers} layers "
              f"load on demand, so only the experts a token routes to are materialised.")
"""

ns = {}
exec(compile(_src, "<patch_airllm_expertstream>", "exec"), ns)
AirLLMBaseModel._setup_expert_streaming = ns["_setup_expert_streaming"]
print("patched AirLLMBaseModel._setup_expert_streaming")
print("DONE")
