"""
app/schemas.py — Pydantic v2 Request & Response Schemas (Day 7)
===============================================================
Defines strict type-validated models for the FastAPI endpoints.
All financial fields are Optional to handle partial extractions gracefully.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request Models ────────────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    """Payload for POST /v1/extract and /v1/extract-stream."""

    text: Annotated[
        str,
        Field(
            min_length=10,
            max_length=16_000,
            description="Raw SEC 10-K filing excerpt or financial report text",
            examples=["Apple Inc. reported revenue of $394.3 billion for fiscal 2023..."],
        ),
    ]
    temperature: Annotated[
        float,
        Field(
            default=0.1,
            ge=0.0,
            le=1.0,
            description="Sampling temperature. Lower = more deterministic JSON extraction.",
        ),
    ] = 0.1
    max_tokens: Annotated[
        int,
        Field(
            default=512,
            ge=64,
            le=2048,
            description="Maximum tokens to generate in the response.",
        ),
    ] = 512
    guided_json: Annotated[
        bool,
        Field(
            default=True,
            description="Enforce JSON schema at logit level via vLLM guided decoding.",
        ),
    ] = True


# ── Response Models ───────────────────────────────────────────────────────────

class FinancialExtraction(BaseModel):
    """Structured financial metrics extracted from a filing excerpt."""

    company_name: Optional[str] = Field(None, description="Company legal name")
    fiscal_year: Optional[int] = Field(None, ge=1900, le=2100, description="Fiscal year (e.g. 2023)")
    revenue_usd_millions: Optional[float] = Field(None, description="Total revenue in USD millions")
    net_income_usd_millions: Optional[float] = Field(None, description="Net income / loss in USD millions")
    operating_margin_pct: Optional[float] = Field(None, ge=-100.0, le=100.0, description="Operating margin %")
    eps_diluted: Optional[float] = Field(None, description="Diluted earnings per share (USD)")
    total_assets_usd_millions: Optional[float] = Field(None, description="Total assets in USD millions")
    total_debt_usd_millions: Optional[float] = Field(None, description="Total debt in USD millions")
    free_cash_flow_usd_millions: Optional[float] = Field(None, description="Free cash flow in USD millions")
    risk_factors: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Top material risk factors identified in the filing",
    )
    forward_guidance: Optional[str] = Field(None, description="Management forward guidance summary")

    @field_validator("risk_factors", mode="before")
    @classmethod
    def limit_risk_factors(cls, v: list[str]) -> list[str]:
        if isinstance(v, list):
            return v[:5]
        return v


class ExtractionResponse(BaseModel):
    """Full API response envelope for POST /v1/extract."""

    extraction: FinancialExtraction
    model_version: str = Field(default="FinAudit-8B-AWQ-v1.0")
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = Field(None, description="End-to-end inference latency in ms")


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(description="'ok' or 'degraded'")
    model_loaded: bool
    model_path: str
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_utilization_pct: Optional[float] = None


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: Optional[str] = None
    status_code: int
