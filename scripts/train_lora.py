"""QLoRA fine-tune for Qwen3-Coder-30B-A3B-Instruct (Colibri).

HONEST VRAM MATH — read before running:
  Full fine-tune (AdamW, bf16): ~ 30B * (2 bytes weights + 4 opt + 4 grad) ≈ 330GB VRAM.
    Needs A100/H100-class hardware. Impossible on RTX 4060 8GB.
  QLoRA 4-bit (NF4): weights ~15GB (fp4) + LoRA adapters + activations.
    Still > 8GB for 30B. Will OOM on your 4060.
  QLoRA on 8GB is only possible for models <= ~7B (e.g. Qwen3-8B).

What this script does on THIS machine:
  It does NOT pretrain / full-finetune 30B. It trains small LoRA adapters
  on top of the frozen 4-bit base — and even that needs >8GB for 30B, so
  the realistic local path is: fine-tune a smaller coder helper (4-8B)
  locally, or run this script on a big-GPU cloud instance.

Usage (cloud big GPU):
  pip install torch transformers peft bitsandbytes trl datasets
  python scripts/train_lora.py --data data.jsonl --out adapter/

Dataset data.jsonl format: {"prompt": ..., "completion": ...} per line.
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--data", default="data.jsonl")
    ap.add_argument("--out", default="adapter-colibri")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    a = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing dep: {e}")
        print("pip install torch transformers peft bitsandbytes trl datasets")
        print("NOTE: QLoRA-30B needs ~16-24GB VRAM (4-bit + adapters). Your 4060 (8GB) will OOM.")
        print("Local alternative: weight-space patch + decoding tuning (scripts above),")
        print("or QLoRA a 4-8B model locally, or run this script on a cloud A100.")
        return

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"Free VRAM: {free/1e9:.1f}GB / {total/1e9:.1f}GB")
        if free < 16e9:
            print("WARNING: <16GB free VRAM. 30B QLoRA will likely OOM. Use a bigger GPU.")
    else:
        print("No CUDA GPU detected. Training will run on CPU (extremely slow).")

    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.base, load_in_4bit=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.grad_ckpt)
    peft_cfg = LoraConfig(r=a.r, lora_alpha=a.r*2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=a.data)
    def fmt(ex):
        return {"text": f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['completion']}<|im_end|>"}
    trainer = SFTTrainer(model=model, train_dataset=ds.map(fmt),
        args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=a.lr, logging_steps=5, save_strategy="epoch",
            bf16=True, gradient_checkpointing=True))
    trainer.train()
    trainer.save_model(a.out)
    print(f"Saved adapter to {a.out} — merge with PEFT before GGUF conversion.")

if __name__ == "__main__":
    main()
