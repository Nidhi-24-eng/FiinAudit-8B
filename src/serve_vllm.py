"""
Day 6 — vLLM Inference Engine Setup & Startup Verification
===========================================================
Configures and launches the vLLM engine with:
  • AWQ 4-bit quantization
  • PagedAttention (KV-Cache memory efficiency)
  • Guided Decoding via Outlines (enforces Pydantic JSON schema at logit level)

Usage (standalone):
    python -m src.serve_vllm

Or imported by app/main.py for the FastAPI layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ── vLLM engine configuration ─────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "checkpoints/finaudit-awq")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))
GPU_MEM_UTILIZATION = float(os.getenv("GPU_MEM_UTILIZATION", "0.90"))
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "64"))      # concurrent sequences
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL", "1"))  # increase for multi-GPU


# ── Pydantic schema for guided decoding ──────────────────────────────────────
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company_name":                 {"type": ["string", "null"]},
        "fiscal_year":                  {"type": ["integer", "null"]},
        "revenue_usd_millions":         {"type": ["number", "null"]},
        "net_income_usd_millions":      {"type": ["number", "null"]},
        "operating_margin_pct":         {"type": ["number", "null"]},
        "eps_diluted":                  {"type": ["number", "null"]},
        "total_assets_usd_millions":    {"type": ["number", "null"]},
        "total_debt_usd_millions":      {"type": ["number", "null"]},
        "free_cash_flow_usd_millions":  {"type": ["number", "null"]},
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "forward_guidance":             {"type": ["string", "null"]},
    },
    "required": [
        "company_name", "fiscal_year", "revenue_usd_millions",
        "net_income_usd_millions", "operating_margin_pct",
        "risk_factors",
    ],
    "additionalProperties": False,
}


def get_engine():
    """
    Lazy-load and return a singleton vLLM LLM engine instance.
    Called once at API startup — cached via module-level variable.
    """
    global _engine
    if "_engine" not in globals() or _engine is None:
        try:
            from vllm import LLM
        except ImportError:
            raise RuntimeError(
                "vLLM is not installed. Run:  pip install vllm\n"
                "Note: vLLM requires Linux with CUDA. On Windows, use WSL2 or Docker."
            )

        print(f"[vLLM] Loading model from: {MODEL_PATH}")
        _engine = LLM(
            model=MODEL_PATH,
            quantization="awq",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTILIZATION,
            max_num_seqs=MAX_NUM_SEQS,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=False,
            dtype="float16",
        )
        print("[vLLM] Engine ready ✓")
    return _engine


def build_sampling_params(
    temperature: float = 0.1,
    max_tokens: int = 512,
    guided_json: bool = True,
):
    """Build vLLM SamplingParams with optional guided JSON decoding."""
    try:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        raise RuntimeError("vLLM not available")

    guided = None
    if guided_json:
        guided = GuidedDecodingParams(json=json.dumps(EXTRACTION_JSON_SCHEMA))

    return SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
        guided_decoding=guided,
        stop=["</s>", "<|eot_id|>"],
    )


def run_inference(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """Run a single synchronous inference request."""
    engine = get_engine()
    params = build_sampling_params(temperature=temperature, max_tokens=max_tokens)
    outputs = engine.generate([prompt], params)
    return outputs[0].outputs[0].text.strip()


if __name__ == "__main__":
    import time
    from rich.console import Console
    console = Console()

    console.rule("[bold]FinAudit-8B — vLLM Engine Verification (Day 6)[/bold]")

    TEST_PROMPT = (
        "<|system|>\nYou are FinAudit-8B. Extract financial metrics as JSON.\n"
        "<|user|>\nApple Inc. reported revenue of $394.3 billion and net income of $97.0 billion "
        "for fiscal year 2023. Diluted EPS was $6.13. The company faces risks from supply chain "
        "concentration, foreign exchange volatility, and intense smartphone market competition.\n"
        "<|assistant|>\n"
    )

    console.log("[cyan]Running test inference …[/cyan]")
    t0 = time.perf_counter()
    result = run_inference(TEST_PROMPT)
    latency_ms = (time.perf_counter() - t0) * 1000

    console.log(f"[green]✓ Inference complete in {latency_ms:.1f} ms[/green]")
    console.log(f"[green]Output:[/green]\n{result}")

    try:
        parsed = json.loads(result)
        console.log("[bold green]✓ Valid JSON output — guided decoding working[/bold green]")
    except json.JSONDecodeError:
        console.log("[bold red]✗ Output is not valid JSON — check guided decoding config[/bold red]")
