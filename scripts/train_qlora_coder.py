"""QLoRA fine-tune of Qwen2.5-Coder-3B-Instruct on execution-verified data.

Fits an 8 GB RTX 4060: 4-bit NF4 base (~2.2 GB) + LoRA adapters + activations
with gradient checkpointing, batch 1, seq 1024. The 30B teacher cannot be
trained on this GPU (QLoRA-30B needs >16 GB even at 4-bit), so the trainable
deliverable is this distilled student.

Run:
  .venv/Scripts/python.exe scripts/train_qlora_coder.py
  .venv/Scripts/python.exe scripts/train_qlora_coder.py --merge   # after training
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "sft_verified.jsonl"
ADAPTER = ROOT / "results" / "qlora-coder-3b"
MERGED = ROOT / "results" / "qlora-coder-3b-merged"
STUDENT = "Qwen/Qwen2.5-Coder-3B-Instruct"


def load_rows():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} verified rows; run gen_sft_data.py first")
    return rows


def train(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(STUDENT)
    rows = load_rows()

    def to_text(row):
        return tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)

    ds = Dataset.from_list([{"text": to_text(r)} for r in rows])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, quantization_config=bnb, device_map={"": 0},
        dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=str(ADAPTER), per_device_train_batch_size=1,
        gradient_accumulation_steps=8, num_train_epochs=args.epochs,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=2,  # transformers 5.x removed warmup_ratio
        logging_steps=5, save_strategy="epoch", bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", max_length=1024, report_to=[],
        dataset_text_field="text", seed=7)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    trainer.save_model(str(ADAPTER))
    tok.save_pretrained(str(ADAPTER))
    print("adapter saved ->", ADAPTER)


def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload()
    merged.save_pretrained(str(MERGED))
    AutoTokenizer.from_pretrained(STUDENT).save_pretrained(str(MERGED))
    print("merged model ->", MERGED)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        train(a)