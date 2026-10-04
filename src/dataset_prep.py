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
from datasets import load_dataset
from rich.console import Console

console = Console(force_terminal=False, highlight=False)
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

    # ── 1. nickmuchi/financial-classification (modern Parquet-based dataset) ──
    console.log("Loading nickmuchi/financial-classification ...")
    try:
        ds1 = load_dataset("nickmuchi/financial-classification", split="train")
        count = 0
        for row in ds1:
            text = row.get("text", "") or row.get("sentence", "")
            if not text or len(text) < 20:
                continue
            output = _make_dummy_output(text)
            all_samples.append(_format_sample(text, output))
            count += 1
        console.log(f"financial-classification: {count} samples loaded")
    except Exception as exc:
        console.log(f"[SKIP] financial-classification: {exc}")

    # ── 2. zeroshot/twitter-financial-news-topic ──────────────────────────────
    console.log("Loading zeroshot/twitter-financial-news-topic ...")
    try:
        ds2 = load_dataset("zeroshot/twitter-financial-news-topic", split="train")
        count = 0
        for row in ds2:
            text = row.get("text", "") or row.get("sentence", "")
            if not text or len(text) < 20:
                continue
            output = _make_dummy_output(text)
            all_samples.append(_format_sample(text, output))
            count += 1
        console.log(f"twitter-financial-news: {count} samples loaded")
    except Exception as exc:
        console.log(f"[SKIP] twitter-financial-news: {exc}")

    # ── 3. FinGPT/fingpt-sentiment-train (with safe float parsing) ────────────
    console.log("Loading FinGPT/fingpt-sentiment-train ...")
    try:
        ds3 = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
        count = 0
        for row in ds3:
            text = row.get("input", "") or row.get("instruction", "")
            if not text or len(text) < 20:
                continue
            try:
                output = _make_dummy_output(text)
            except Exception:
                continue
            all_samples.append(_format_sample(text, output))
            count += 1
        console.log(f"fingpt-sentiment: {count} samples loaded")
    except Exception as exc:
        console.log(f"[SKIP] fingpt-sentiment: {exc}")

    # ── 4. Synthetic SEC 10-K samples (always available, no download needed) ──
    console.log("Generating synthetic SEC 10-K samples ...")
    SEC_TEMPLATES = [
        ("Apple Inc. reported net revenue of $394.3 billion for fiscal year 2023, "
         "a 3% decrease year-over-year. Net income was $97.0 billion. Diluted EPS was $6.13. "
         "The company faces supply chain concentration risk, foreign exchange volatility, "
         "and intense competition in the smartphone market.",
         {"company_name": "Apple Inc.", "fiscal_year": 2023, "revenue_usd_millions": 394300.0,
          "net_income_usd_millions": 97000.0, "eps_diluted": 6.13,
          "risk_factors": ["Supply chain concentration risk", "Foreign exchange volatility",
                           "Intense competition in the smartphone market"]}),
        ("Microsoft Corporation reported total revenue of $211.9 billion for fiscal year 2023. "
         "Operating income was $88.5 billion with an operating margin of 41.8%. "
         "Risks include cloud market competition, cybersecurity threats, and regulatory compliance.",
         {"company_name": "Microsoft Corporation", "fiscal_year": 2023,
          "revenue_usd_millions": 211900.0, "operating_margin_pct": 41.8,
          "risk_factors": ["Cloud market competition", "Cybersecurity threats", "Regulatory compliance"]}),
        ("Amazon.com, Inc. reported net sales of $554.0 billion for the year ended December 31, 2023. "
         "AWS revenue grew 13% to $90.8 billion. Free cash flow was $32.0 billion. "
         "Material risks: retail competition, international regulatory exposure, supply chain disruptions.",
         {"company_name": "Amazon.com, Inc.", "fiscal_year": 2023,
          "revenue_usd_millions": 554000.0, "free_cash_flow_usd_millions": 32000.0,
          "risk_factors": ["Retail competition", "International regulatory exposure",
                           "Supply chain disruptions"]}),
        ("Tesla, Inc. reported total revenues of $97.7 billion for fiscal 2023, a 19% increase. "
         "Automotive gross margin was 17.1%. Total assets were $106.6 billion. "
         "Key risks: EV adoption uncertainty, production ramp challenges, raw material volatility.",
         {"company_name": "Tesla, Inc.", "fiscal_year": 2023,
          "revenue_usd_millions": 97700.0, "operating_margin_pct": 17.1,
          "total_assets_usd_millions": 106600.0,
          "risk_factors": ["EV adoption uncertainty", "Production ramp challenges",
                           "Raw material volatility"]}),
        ("Alphabet Inc. reported revenues of $307.4 billion for fiscal 2023. "
         "Net income was $73.8 billion. Diluted EPS was $5.80. Total debt was $13.2 billion. "
         "Risks include search disruption from AI competitors, antitrust actions, ad market cyclicality.",
         {"company_name": "Alphabet Inc.", "fiscal_year": 2023,
          "revenue_usd_millions": 307400.0, "net_income_usd_millions": 73800.0,
          "eps_diluted": 5.80, "total_debt_usd_millions": 13200.0,
          "risk_factors": ["Search disruption from AI competitors", "Antitrust regulatory actions",
                           "Advertising market cyclicality"]}),
        ("JPMorgan Chase reported net revenue of $158.1 billion for full-year 2023. "
         "Net income was $49.6 billion. Total assets were $3.9 trillion. "
         "Risk factors include credit risk from loan defaults, interest rate sensitivity, "
         "and heightened regulatory capital requirements under Basel III endgame rules.",
         {"company_name": "JPMorgan Chase", "fiscal_year": 2023,
          "revenue_usd_millions": 158100.0, "net_income_usd_millions": 49600.0,
          "total_assets_usd_millions": 3900000.0,
          "risk_factors": ["Credit risk from loan defaults", "Interest rate sensitivity",
                           "Regulatory capital requirements under Basel III"]}),
        ("Johnson & Johnson reported sales of $85.2 billion for fiscal year 2023. "
         "Operating margin was 22.7%. Free cash flow was $14.2 billion. "
         "Material risks include product liability litigation, generic drug competition, "
         "and healthcare regulatory changes.",
         {"company_name": "Johnson & Johnson", "fiscal_year": 2023,
          "revenue_usd_millions": 85200.0, "operating_margin_pct": 22.7,
          "free_cash_flow_usd_millions": 14200.0,
          "risk_factors": ["Product liability litigation", "Generic drug competition",
                           "Healthcare regulatory changes"]}),
        ("Exxon Mobil Corporation reported revenues of $398.7 billion for fiscal 2023. "
         "Net income was $36.0 billion. Capital expenditures were $26.3 billion. "
         "The company faces risks from crude oil price volatility, energy transition policies, "
         "and geopolitical disruptions in key operating regions.",
         {"company_name": "Exxon Mobil Corporation", "fiscal_year": 2023,
          "revenue_usd_millions": 398700.0, "net_income_usd_millions": 36000.0,
          "risk_factors": ["Crude oil price volatility", "Energy transition policies",
                           "Geopolitical disruptions"]}),
    ]
    # Repeat and vary templates to reach a meaningful synthetic count
    random.seed(seed)
    synthetic_count = 0
    for _ in range(500):
        text, output_dict = random.choice(SEC_TEMPLATES)
        # Add slight variation by prepending a phrase
        prefixes = [
            "According to the annual report, ", "The 10-K filing states that ",
            "Management disclosed that ", "Per the SEC filing, ",
            "In its annual report to shareholders, ",
        ]
        varied_text = random.choice(prefixes) + text
        output_copy = dict(output_dict)
        output_copy.setdefault("operating_margin_pct", None)
        output_copy.setdefault("eps_diluted", None)
        output_copy.setdefault("total_assets_usd_millions", None)
        output_copy.setdefault("total_debt_usd_millions", None)
        output_copy.setdefault("free_cash_flow_usd_millions", None)
        output_copy.setdefault("net_income_usd_millions", None)
        output_copy.setdefault("revenue_usd_millions", None)
        output_copy.setdefault("forward_guidance", None)
        all_samples.append(_format_sample(varied_text, output_copy))
        synthetic_count += 1
    console.log(f"synthetic SEC samples: {synthetic_count} samples generated")

    console.log(f"Total raw samples loaded: {len(all_samples)}")
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
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        console.log(f"Saved {len(split_data):,} samples -> {path}")


@app.command()
def main(
    output_dir: Path = typer.Option(Path("data"), help="Directory to write JSONL splits"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    target_total: int = typer.Option(5000, help="Max samples to use (train+val+test)"),
) -> None:
    """Prepare financial instruction-following datasets from HuggingFace sources."""
    console.log("=" * 60)
    console.log("FinAudit-8B -- Dataset Preparation (Day 2)")
    console.log("=" * 60)
    samples = load_and_process_datasets(seed=seed)
    if not samples:
        console.log("ERROR: No samples loaded. Check internet connection and dataset availability.")
        raise typer.Exit(1)
    split_and_save(samples, output_dir=output_dir, seed=seed, target_total=target_total)
    console.log("=" * 60)
    console.log("Dataset preparation complete!")
    console.log("=" * 60)


if __name__ == "__main__":
    app()
