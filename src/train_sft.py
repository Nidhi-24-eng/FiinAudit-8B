"""
Day 3 — Supervised Fine-Tuning (SFT) via Unsloth + QLoRA
=========================================================
Fine-tunes Llama-3.1-8B-Instruct on financial extraction instructions
while keeping peak GPU memory under 10 GB VRAM.

Designed to run on:
  • Google Colab (free T4 — 15 GB VRAM)
  • Local RTX 3060/4060 (8 GB VRAM)
  • RunPod / Vast.ai RTX 4090

Usage:
    python -m src.train_sft \
        --train_file data/train.jsonl \
        --val_file data/val.jsonl \
        --output_dir checkpoints/sft-adapter \
        --epochs 3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# ── Model & LoRA hyper-parameters ────────────────────────────────────────────
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE = 2
GRAD_ACCUM = 4            # effective batch = 8
WARMUP_RATIO = 0.03
LR = 2e-4
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"
FP16 = True               # set False on Ampere+ and use bf16=True instead
BF16 = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(sample: dict[str, Any], tokenizer: Any) -> str:
    """Convert our instruction-format sample to the model's chat template."""
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user",   "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


@app.command()
def main(
    train_file: Path = typer.Option(Path("data/train.jsonl")),
    val_file: Path   = typer.Option(Path("data/val.jsonl")),
    output_dir: Path = typer.Option(Path("checkpoints/sft-adapter")),
    epochs: int      = typer.Option(3),
    run_name: str    = typer.Option("finaudit-sft"),
) -> None:
    """Run SFT fine-tuning on Llama 3.1 8B with QLoRA via Unsloth."""
    console.rule("[bold]FinAudit-8B — SFT Training (Day 3)[/bold]")

    # ── Lazy imports (GPU environment only) ───────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import Dataset
    except ImportError as e:
        console.log(f"[red]✗ Missing dependency: {e}[/red]")
        console.log("Run:  pip install unsloth trl transformers datasets")
        raise typer.Exit(1)

    # ── Load model & tokenizer ────────────────────────────────────────────────
    console.log(f"[cyan]Loading base model: {MODEL_NAME}[/cyan]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # auto-detect bfloat16 on Ampere+, float16 otherwise
        load_in_4bit=True,   # QLoRA — keeps peak VRAM under 10 GB
    )

    # ── Apply QLoRA adapters ──────────────────────────────────────────────────
    console.log(f"[cyan]Applying QLoRA adapters (r={LORA_R}, α={LORA_ALPHA}) …[/cyan]")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",   # saves ~30% VRAM
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # ── Load datasets ─────────────────────────────────────────────────────────
    console.log(f"[cyan]Loading datasets from {train_file} …[/cyan]")
    train_raw = load_jsonl(train_file)
    val_raw   = load_jsonl(val_file)

    train_texts = [format_prompt(s, tokenizer) for s in train_raw]
    val_texts   = [format_prompt(s, tokenizer) for s in val_raw]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds   = Dataset.from_dict({"text": val_texts})
    console.log(f"[green]✓ Train: {len(train_ds):,}  |  Val: {len(val_ds):,}[/green]")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        fp16=FP16,
        bf16=BF16,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",               # swap to "wandb" if desired
        run_name=run_name,
        dataloader_num_workers=0,       # important on Windows
        seed=42,
    )

    # ── SFT Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=True,                   # pack short samples together → 2x speed
        args=training_args,
    )

    console.log("[bold yellow]Starting SFT training …[/bold yellow]")
    trainer_stats = trainer.train()
    console.log(f"[green]✓ Training complete. Stats: {trainer_stats}[/green]")

    # ── Save adapter ──────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    console.log(f"[bold green]✓ Adapter checkpoint saved → {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
