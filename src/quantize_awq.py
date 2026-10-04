"""
Day 5 — AWQ Quantization & Model Compression
=============================================
Compresses the merged FP16 model (~16 GB) down to 4-bit AWQ (~5.5 GB)
using AutoAWQ with a financial calibration subset.

Usage:
    python -m src.quantize_awq \
        --model_dir checkpoints/orpo-merged \
        --output_dir checkpoints/finaudit-awq \
        --calib_file data/val.jsonl
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

# ── AWQ Configuration ─────────────────────────────────────────────────────────
AWQ_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}
CALIB_DATA_SIZE = 512   # number of calibration samples


def load_calibration_data(calib_file: Path, n: int = CALIB_DATA_SIZE) -> list[str]:
    """Load financial texts as AWQ calibration data."""
    with calib_file.open("r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    random.shuffle(samples)
    texts = []
    for s in samples[:n]:
        text = s.get("instruction", "") + "\n" + s.get("output", "")
        texts.append(text[:1024])   # clip to avoid OOM during calibration
    return texts


@app.command()
def main(
    model_dir: Path  = typer.Option(Path("checkpoints/orpo-merged"),   help="Path to merged FP16 model"),
    output_dir: Path = typer.Option(Path("checkpoints/finaudit-awq"), help="Where to save AWQ model"),
    calib_file: Path = typer.Option(Path("data/val.jsonl"),            help="Calibration JSONL dataset"),
    calib_size: int  = typer.Option(CALIB_DATA_SIZE,                   help="Number of calibration samples"),
) -> None:
    """Quantize the fine-tuned model to 4-bit AWQ format."""
    console.rule("[bold]FinAudit-8B — AWQ Quantization (Day 5)[/bold]")

    # ── Lazy imports ──────────────────────────────────────────────────────────
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        console.log(f"[red]✗ Missing: {e}[/red]  →  pip install autoawq")
        raise typer.Exit(1)

    # ── Load tokenizer & model ────────────────────────────────────────────────
    console.log(f"[cyan]Loading model from {model_dir} …[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoAWQForCausalLM.from_pretrained(
        str(model_dir),
        safetensors=True,
        device_map="auto",
    )

    # ── Calibration data ──────────────────────────────────────────────────────
    console.log(f"[cyan]Loading {calib_size} calibration samples from {calib_file} …[/cyan]")
    calib_data = load_calibration_data(calib_file, n=calib_size)
    console.log(f"[green]✓ Calibration samples: {len(calib_data)}[/green]")

    # ── Quantize ──────────────────────────────────────────────────────────────
    console.log("[bold yellow]Running AWQ quantization (this may take 30–60 min) …[/bold yellow]")
    model.quantize(tokenizer, quant_config=AWQ_CONFIG, calib_data=calib_data)

    # ── Save quantized model ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ── Size report ───────────────────────────────────────────────────────────
    size_gb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / (1024**3)
    console.log(f"[bold green]✓ AWQ model saved → {output_dir}[/bold green]")
    console.log(f"[green]  Compressed size: {size_gb:.2f} GB  (target: ~5.5 GB)[/green]")
    console.log("[dim]Next step: run src/serve_vllm.py or directly use vLLM[/dim]")


if __name__ == "__main__":
    app()
