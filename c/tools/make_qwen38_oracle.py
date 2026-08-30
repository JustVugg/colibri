#!/usr/bin/env python3
"""Run the released Qwen4-Exp text forward as an independent oracle.

This is intended for the text-only fixture produced by ``make_qwen38_tiny.py``
and records both a cache prefill and a one-token cached decode.  It also emits
greedy generated IDs, SHA-256 digests of logits/hidden states, and the final
prediction logits used by the C numeric gate. Earlier intermediate tensors stay
as digests so the JSON remains compact while still detecting layout changes.

The released Qwen3.8 checkpoint is multimodal and includes vision/MTP tensors;
this script intentionally refuses a multimodal config rather than silently
loading the wrong wrapper.  Use a text-only directory (such as the tiny
fixture) for an oracle run.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import torch
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}. Install torch and transformers.")

TRANSFORMERS_VERSION = "5.16.1"


def _digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _load_model(model_dir):
    try:
        import transformers
        from transformers import AutoConfig, Qwen4ExpForCausalLM
    except ImportError as exc:
        sys.exit(
            f"Missing Qwen4-Exp support. Install transformers=={TRANSFORMERS_VERSION}: {exc}"
        )
    if transformers.__version__ != TRANSFORMERS_VERSION:
        sys.exit(
            f"Qwen3.8 oracle generation requires transformers=={TRANSFORMERS_VERSION}; "
            f"found {transformers.__version__}."
        )
    try:
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    except Exception as exc:
        sys.exit(f"Unable to read text fixture config from {model_dir}: {exc}")
    if getattr(config, "model_type", None) != "qwen4_exp_text":
        sys.exit(
            "This oracle accepts only text-only qwen4_exp_text configs; the "
            "released qwen3.8 directory is multimodal (vision/MTP excluded)."
        )
    try:
        model = Qwen4ExpForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    except Exception as exc:
        sys.exit(f"Unable to load Qwen4-Exp text weights from {model_dir}: {exc}")
    model.eval()
    return model, transformers.__version__


def run(model, prompt_ids, max_new, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        prefill = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
        )
        decode_ids = prefill.logits[:, -1:, :].argmax(dim=-1)
        decode = model(
            input_ids=decode_ids,
            past_key_values=prefill.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            output_router_logits=True,
        )
        # Run exactly max_new steps.  This intentionally continues across EOS
        # so the fixture covers Qwen4-Exp's PLE history reset as well as normal
        # prefill/decode cache updates.
        generated = input_ids
        current = input_ids
        past = None
        for _ in range(max_new):
            step = model(input_ids=current, past_key_values=past, use_cache=True)
            token = step.logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((generated, token), dim=1)
            current = token
            past = step.past_key_values

    def summary(outputs):
        hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
        return {
            "logits_shape": list(outputs.logits.shape),
            "logits_sha256": _digest(outputs.logits),
            "last_logits_argmax": int(outputs.logits[:, -1, :].argmax(dim=-1).item()),
            "last_hidden_shape": list(hidden.shape) if hidden is not None else None,
            "last_hidden_sha256": _digest(hidden) if hidden is not None else None,
            "router_layers": len(outputs.router_logits) if outputs.router_logits else 0,
        }

    full_ids = generated[0].tolist()
    return {
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "generated_ids": full_ids[len(prompt_ids) :],
        # Greedy IDs alone can remain unchanged through substantial arithmetic
        # or layout drift, so retain the logits that predicted the final ID.
        "final_logits": step.logits[0, -1, :].float().cpu().tolist(),
        "prefill": summary(prefill),
        "decode": {"input_ids": decode_ids[0].tolist(), **summary(decode)},
        "cache_class": type(prefill.past_key_values).__name__,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local text-only HF model directory")
    parser.add_argument("--out", required=True, type=Path, help="Reference JSON output")
    parser.add_argument("--prompt-ids", help="Comma-separated IDs (default: fixture prompt)")
    parser.add_argument("--max-new", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    model, transformers_version = _load_model(args.model)
    if args.prompt_ids:
        prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    else:
        prompt_ids = [1, 3, 4, 5, 6]
    if not prompt_ids or any(token < 0 or token >= model.config.vocab_size for token in prompt_ids):
        sys.exit(f"prompt ids must be in [0, {model.config.vocab_size}) and non-empty")
    if args.max_new < 1:
        sys.exit("--max-new must be at least 1")
    payload = {
        "schema_version": 2,
        "model": str(args.model),
        "seed": args.seed,
        "transformers_version": transformers_version,
        "text_only": True,
    }
    payload.update(run(model, prompt_ids, args.max_new, args.seed))
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Reference written to {args.out}")
    print(f"prompt_ids={payload['prompt_ids']}")
    print(f"full_ids={payload['full_ids']}")


if __name__ == "__main__":
    main()
