"""
app/main.py — FinAudit-8B FastAPI Microservice (Day 7)
======================================================
Production-grade async REST API with:
  • POST /v1/extract        — synchronous JSON extraction
  • POST /v1/extract-stream — Server-Sent Events streaming extraction
  • GET  /health            — model status & GPU memory usage

Start with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from rich.console import Console

from app.schemas import (
    ErrorResponse,
    ExtractionRequest,
    ExtractionResponse,
    FinancialExtraction,
    HealthResponse,
)
from src.serve_vllm import (
    EXTRACTION_JSON_SCHEMA,
    MODEL_PATH,
    build_sampling_params,
    get_engine,
)

console = Console()

# ── Prompt builder ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are FinAudit-8B, a domain-adapted financial analysis model. "
    "Extract structured financial metrics and risk information from SEC 10-K "
    "filing excerpts. ALWAYS respond with a single valid JSON object. "
    "Return null for any field not present. Do NOT hallucinate numbers."
)


def build_prompt(text: str) -> str:
    schema_hint = json.dumps(EXTRACTION_JSON_SCHEMA, indent=2)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n"
        f"Extract financial metrics as JSON matching this schema:\n{schema_hint}\n\n"
        f"### Filing Excerpt:\n{text}\n\n"
        f"### Extracted JSON:\n"
        f"<|assistant|>\n"
    )


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the vLLM engine on startup."""
    console.log("[bold cyan]FinAudit-8B API starting up …[/bold cyan]")
    try:
        get_engine()   # pre-load model weights into GPU memory
        console.log("[bold green]✓ Model loaded and ready.[/bold green]")
    except Exception as exc:
        console.log(f"[yellow]⚠ vLLM engine not available: {exc}[/yellow]")
        console.log("[yellow]  API will run in stub mode (returns mock data).[/yellow]")
    yield
    console.log("[bold]FinAudit-8B API shutting down.[/bold]")


app = FastAPI(
    title="FinAudit-8B",
    description=(
        "Production-grade Financial SEC 10-K Risk & Metric Extraction Engine. "
        "Fine-tuned Llama 3.1 8B with Unsloth QLoRA + ORPO, served via vLLM "
        "with 4-bit AWQ quantization and guided JSON decoding."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, status_code=exc.status_code
        ).model_dump(),
    )


# ── Helper: run inference in thread pool ─────────────────────────────────────
def _sync_inference(prompt: str, req: ExtractionRequest) -> tuple[str, float]:
    """Blocking vLLM call — run in executor to avoid blocking event loop."""
    engine = get_engine()
    params = build_sampling_params(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        guided_json=req.guided_json,
    )
    t0 = time.perf_counter()
    outputs = engine.generate([prompt], params)
    latency_ms = (time.perf_counter() - t0) * 1000
    return outputs[0].outputs[0].text.strip(), latency_ms


def _mock_extraction(text: str) -> FinancialExtraction:
    """Fallback stub when vLLM is unavailable (dev / CPU mode)."""
    import re
    revenue = None
    m = re.search(r"\$\s*([\d,.]+)\s*(billion|million)", text, re.I)
    if m:
        val = float(m.group(1).replace(",", ""))
        if "billion" in (m.group(2) or "").lower():
            val *= 1000
        revenue = round(val, 2)
    return FinancialExtraction(
        company_name=None,
        fiscal_year=None,
        revenue_usd_millions=revenue,
        risk_factors=["[STUB MODE — vLLM not available]"],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/v1/extract",
    response_model=ExtractionResponse,
    summary="Extract financial metrics from a 10-K excerpt",
    tags=["Extraction"],
)
async def extract(req: ExtractionRequest) -> ExtractionResponse:
    """
    Accepts raw financial text and returns structured JSON with extracted
    metrics and risk factors. Uses vLLM guided decoding to guarantee
    valid JSON output conforming to the FinancialExtraction schema.
    """
    prompt = build_prompt(req.text)
    loop = asyncio.get_event_loop()

    try:
        raw_output, latency_ms = await loop.run_in_executor(
            None, _sync_inference, prompt, req
        )
    except RuntimeError:
        # vLLM not available → stub mode
        extraction = _mock_extraction(req.text)
        return ExtractionResponse(extraction=extraction, latency_ms=None)

    # Parse & validate model output
    try:
        data = json.loads(raw_output)
        extraction = FinancialExtraction(**data)
    except (json.JSONDecodeError, Exception) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model output failed validation: {exc}. Raw: {raw_output[:200]}",
        )

    return ExtractionResponse(
        extraction=extraction,
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/v1/extract-stream",
    summary="Stream financial metric extraction via SSE",
    tags=["Extraction"],
)
async def extract_stream(req: ExtractionRequest) -> StreamingResponse:
    """
    Streams token-by-token extraction via Server-Sent Events.
    Clients receive partial JSON as it is generated.
    """
    prompt = build_prompt(req.text)

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        except ImportError:
            # Stub streaming
            mock = _mock_extraction(req.text)
            mock_json = mock.model_dump_json(indent=2)
            for char in mock_json:
                yield f"data: {json.dumps({'token': char})}\n\n"
                await asyncio.sleep(0.005)
            yield "data: [DONE]\n\n"
            return

        params = build_sampling_params(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            guided_json=req.guided_json,
        )
        # For true streaming, use AsyncLLMEngine — this is a simplified version
        loop = asyncio.get_event_loop()
        raw, _ = await loop.run_in_executor(None, _sync_inference, prompt, req)
        for char in raw:
            yield f"data: {json.dumps({'token': char})}\n\n"
            await asyncio.sleep(0)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """Returns model load status and current GPU memory utilization."""
    model_loaded = "_engine" in globals() and globals()["_engine"] is not None

    vram_used = vram_total = vram_pct = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_total = round(props.total_memory / (1024**3), 2)
        vram_used  = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        vram_pct   = round(vram_used / vram_total * 100, 1) if vram_total else None

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        vram_used_gb=vram_used,
        vram_total_gb=vram_total,
        vram_utilization_pct=vram_pct,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
