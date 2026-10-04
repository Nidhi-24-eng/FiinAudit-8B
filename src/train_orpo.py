"""
Day 4 — Preference Alignment via ORPO (Odds Ratio Preference Optimization)
===========================================================================
Builds a chosen/rejected preference dataset and runs ORPO training via
Unsloth's ORPOTrainer — combining instruction tuning + preference alignment
in a single forward pass (no separate reference model needed).

Usage:
    python -m src.train_orpo \
        --sft_adapter checkpoints/sft-adapter \
        --output_dir checkpoints/orpo-merged \
        --epochs 1
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# ── ORPO hyper-parameters ─────────────────────────────────────────────────────
ORPO_BETA = 0.1              # λ in ORPO loss — controls alignment strength
ORPO_MAX_LENGTH = 2048
ORPO_LR = 8e-6
ORPO_BATCH = 1
ORPO_GRAD_ACCUM = 8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_preference_pairs(
    samples: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    For each correct (chosen) extraction sample, fabricate a rejected variant:
      - 50% chance: plain conversational response (schema violation)
      - 25% chance: truncated / malformed JSON
      - 25% chance: JSON with hallucinated numeric fields

    Returns a list of dicts with keys: prompt, chosen, rejected
    """
    random.seed(seed)
    pairs: list[dict[str, Any]] = []

    HALLUCINATED_NUMBERS = [999.9, 0.01, -42.5, 12345.0, 0.0]
    REJECTED_PROSE_TEMPLATES = [
        "Based on my analysis of the filing, the company appears to be in a strong financial position "
        "with solid revenue growth and manageable risk factors. I would recommend further review.",
        "The filing discusses several important metrics. Revenue seems healthy and the company faces "
        "typical industry risks. Management expressed cautious optimism in their forward guidance.",
        "I cannot determine exact figures from this excerpt alone. The company operates in a competitive "
        "landscape and is subject to various financial and regulatory risks as disclosed.",
    ]

    for sample in samples:
        prompt = sample["instruction"]
        chosen_output = sample["output"]

        # Determine rejection strategy
        strategy = random.random()

        if strategy < 0.50:
            # Plain prose — outright schema violation
            rejected_output = random.choice(REJECTED_PROSE_TEMPLATES)

        elif strategy < 0.75:
            # Malformed / truncated JSON
            try:
                obj = json.loads(chosen_output)
                keys = list(obj.keys())
                # Keep only first half of keys → truncated JSON
                partial = {k: obj[k] for k in keys[: len(keys) // 2]}
                rejected_output = json.dumps(partial, indent=2)[:-2]  # cut off end
            except Exception:
                rejected_output = '{"error": "extraction failed"}'

        else:
            # Hallucinated numbers — all floats replaced with garbage
            try:
                obj = json.loads(chosen_output)
                for k, v in obj.items():
                    if isinstance(v, (int, float)) and v is not None:
                        obj[k] = random.choice(HALLUCINATED_NUMBERS)
                rejected_output = json.dumps(obj, indent=2)
            except Exception:
                rejected_output = '{"revenue_usd_millions": 999.9}'

        pairs.append(
            {
                "prompt": f"<|system|>\n{sample['system']}\n<|user|>\n{prompt}\n<|assistant|>\n",
                "chosen": chosen_output,
                "rejected": rejected_output,
            }
        )

    return pairs


@app.command()
def main(
    train_file: Path = typer.Option(Path("data/train.jsonl")),
    sft_adapter: Path = typer.Option(Path("checkpoints/sft-adapter")),
    output_dir: Path  = typer.Option(Path("checkpoints/orpo-merged")),
    epochs: int       = typer.Option(1),
    seed: int         = typer.Option(42),
) -> None:
    """Run ORPO preference alignment on top of the SFT adapter."""
    console.rule("[bold]FinAudit-8B — ORPO Alignment (Day 4)[/bold]")

    # ── Lazy imports ──────────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
        from trl import ORPOConfig, ORPOTrainer
        from datasets import Dataset
    except ImportError as e:
        console.log(f"[red]✗ Missing dependency: {e}[/red]")
        raise typer.Exit(1)

    # ── Load SFT-adapted model ────────────────────────────────────────────────
    console.log(f"[cyan]Loading SFT adapter from {sft_adapter} …[/cyan]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(sft_adapter),
        max_seq_length=ORPO_MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # ── Build preference dataset ──────────────────────────────────────────────
    console.log("[cyan]Building chosen/rejected preference pairs …[/cyan]")
    raw_samples = load_jsonl(train_file)
    pairs = build_preference_pairs(raw_samples, seed=seed)
    ds = Dataset.from_list(pairs)
    console.log(f"[green]✓ Preference pairs: {len(ds):,}[/green]")

    # ── ORPO config ───────────────────────────────────────────────────────────
    orpo_config = ORPOConfig(
        beta=ORPO_BETA,
        learning_rate=ORPO_LR,
        per_device_train_batch_size=ORPO_BATCH,
        gradient_accumulation_steps=ORPO_GRAD_ACCUM,
        num_train_epochs=epochs,
        max_length=ORPO_MAX_LENGTH,
        max_prompt_length=1024,
        output_dir=str(output_dir / "orpo-checkpoints"),
        logging_steps=5,
        save_steps=100,
        report_to="none",
        seed=seed,
        fp16=True,
        dataloader_num_workers=0,
    )

    trainer = ORPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=orpo_config,
        train_dataset=ds,
    )

    console.log("[bold yellow]Starting ORPO alignment …[/bold yellow]")
    trainer.train()
    console.log("[green]✓ ORPO training complete.[/green]")

    # ── Merge & save FP16 weights ─────────────────────────────────────────────
    console.log("[cyan]Merging adapter into full FP16 weights …[/cyan]")
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    console.log(f"[bold green]✓ Merged FP16 model saved → {output_dir}[/bold green]")
    console.log("[dim]Next step: run src/quantize_awq.py to compress to 4-bit AWQ[/dim]")


if __name__ == "__main__":
    app()
