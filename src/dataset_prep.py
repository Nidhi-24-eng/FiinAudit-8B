"""
Day 2 — Financial Dataset Engineering & Schema Structuring
==========================================================
Loads open financial datasets from Hugging Face and formats them into
strict instruction-following JSONL pairs suitable for QLoRA fine-tuning.

Usage:
    python -m src.dataset_prep --output_dir data/ --seed 42
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import typer
from datasets import concatenate_datasets, load_dataset
from rich.console import Console
from rich.progress import track

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# ── Pydantic-style schema (used as string template for LLM output) ──────────
EXTRACTION_SCHEMA: dict[str, Any] = {
    "company_name": "string | null",
    "fiscal_year": "integer | null",
    "revenue_usd_millions": "float | null",
    "net_income_usd_millions": "float | null",
    "operating_margin_pct": "float | null",
    "eps_diluted": "float | null",
    "total_assets_usd_millions": "float | null",
    "total_debt_usd_millions": "float | null",
    "free_cash_flow_usd_millions": "float | null",
    "risk_factors": ["string"],         # top 3 material risk factors
    "forward_guidance": "string | null",
}

SYSTEM_PROMPT = (
    "You are FinAudit-8B, a domain-adapted financial analysis model. "
    "Your task is to extract structured financial metrics and risk information "
    "from SEC 10-K filings and financial reports. "
    "ALWAYS respond with a single, valid JSON object matching the required schema. "
    "Return null for any field not explicitly mentioned in the source text. "
    "Do NOT hallucinate numbers. Do NOT add commentary outside the JSON object."
)

INSTRUCTION_TEMPLATE = (
    "Extract the key financial metrics and risk factors from the following "
    "SEC 10-K filing excerpt. Return a JSON object with these fields:\n"
    "{schema}\n\n"
    "### Filing Excerpt:\n{text}\n\n"
    "### Extracted JSON:"
)


def _build_schema_hint() -> str:
    lines = []
    for k, v in EXTRACTION_SCHEMA.items():
        lines.append(f'  "{k}": {v}')
    return "{\n" + ",\n".join(lines) + "\n}"


def _make_dummy_output(text: str) -> dict[str, Any]:
    """
    Generate a plausible-looking (but clearly synthetic) extraction output
    for dataset samples that don't already have structured labels.
    Values are extracted heuristically from the text via simple regex.
    """
    revenue_match = re.search(r"\$\s*([\d,.]+)\s*(billion|million)?", text, re.IGNORECASE)
    revenue = None
    if revenue_match:
        val = float(revenue_match.group(1).replace(",", ""))
        if revenue_match.group(2) and "billion" in revenue_match.group(2).lower():
            val *= 1000
        revenue = round(val, 2)

    # Pull risk-like sentences
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
    risk_keywords = ["risk", "uncertain", "adverse", "competition", "regulatory", "litigation"]
    risks = [s for s in sentences if any(kw in s.lower() for kw in risk_keywords)][:3]

    return {
        "company_name": None,
        "fiscal_year": None,
        "revenue_usd_millions": revenue,
        "net_income_usd_millions": None,
        "operating_margin_pct": None,
        "eps_diluted": None,
        "total_assets_usd_millions": None,
        "total_debt_usd_millions": None,
        "free_cash_flow_usd_millions": None,
        "risk_factors": risks if risks else ["Information not available in excerpt"],
        "forward_guidance": None,
    }


def _format_sample(text: str, output: dict[str, Any]) -> dict[str, str]:
    """Format a single sample into the Alpaca-style instruction format."""
    schema_hint = _build_schema_hint()
    instruction = INSTRUCTION_TEMPLATE.format(schema=schema_hint, text=text[:2000])
    return {
        "system": SYSTEM_PROMPT,
        "instruction": instruction,
        "output": json.dumps(output, indent=2),
    }


def load_and_process_datasets(seed: int = 42) -> list[dict[str, str]]:
    """Load HuggingFace financial datasets and process into instruction pairs."""
    all_samples: list[dict[str, str]] = []

    # ── 1. financial_phrasebank ───────────────────────────────────────────────
    console.log("[bold cyan]Loading financial_phrasebank …[/bold cyan]")
    try:
        pb = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
        for row in track(pb, description="financial_phrasebank"):
            text = row["sentence"]
            output = _make_dummy_output(text)
            all_samples.append(_format_sample(text, output))
    except Exception as exc:
        console.log(f"[yellow]⚠ financial_phrasebank skipped: {exc}[/yellow]")

    # ── 2. FinGPT / fingpt-sentiment-train ────────────────────────────────────
    console.log("[bold cyan]Loading FinGPT sentiment dataset …[/bold cyan]")
    try:
        fg = load_dataset("FinGPT/fingpt-sentiment-train", split="train", trust_remote_code=True)
        for row in track(fg, description="fingpt-sentiment"):
            text = row.get("input", "") or row.get("instruction", "")
            if not text:
                continue
            output = _make_dummy_output(text)
            all_samples.append(_format_sample(text, output))
    except Exception as exc:
        console.log(f"[yellow]⚠ FinGPT dataset skipped: {exc}[/yellow]")

    # ── 3. FinGPT / fintech-instruction ──────────────────────────────────────
    console.log("[bold cyan]Loading FinGPT fintech-instruction …[/bold cyan]")
    try:
        ft = load_dataset("FinGPT/fingpt-fintech-instruction", split="train", trust_remote_code=True)
        for row in track(ft, description="fingpt-fintech"):
            text = row.get("instruction", "") + " " + row.get("input", "")
            text = text.strip()
            if not text:
                continue
            output = _make_dummy_output(text)
            all_samples.append(_format_sample(text, output))
    except Exception as exc:
        console.log(f"[yellow]⚠ FinGPT fintech skipped: {exc}[/yellow]")

    console.log(f"[green]✓ Total raw samples loaded: {len(all_samples)}[/green]")
    return all_samples


def split_and_save(
    samples: list[dict[str, str]],
    output_dir: Path,
    seed: int = 42,
    target_total: int = 5000,
) -> None:
    """Shuffle, cap, and split samples into train/val/test JSONL files."""
    random.seed(seed)
    random.shuffle(samples)

    # Cap to target total
    samples = samples[:target_total]
    n = len(samples)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    splits = {
        "train": samples[:n_train],
        "val": samples[n_train : n_train + n_val],
        "test": samples[n_train + n_val :],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        console.log(f"[green]✓ Saved {len(split_data):,} samples → {path}[/green]")


@app.command()
def main(
    output_dir: Path = typer.Option(Path("data"), help="Directory to write JSONL splits"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    target_total: int = typer.Option(5000, help="Max samples to use (train+val+test)"),
) -> None:
    """Prepare financial instruction-following datasets from HuggingFace sources."""
    console.rule("[bold]FinAudit-8B — Dataset Preparation (Day 2)[/bold]")
    samples = load_and_process_datasets(seed=seed)
    if not samples:
        console.log("[red]✗ No samples loaded. Check internet connection and dataset availability.[/red]")
        raise typer.Exit(1)
    split_and_save(samples, output_dir=output_dir, seed=seed, target_total=target_total)
    console.rule("[bold green]Dataset preparation complete![/bold green]")


if __name__ == "__main__":
    app()
