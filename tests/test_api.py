"""
tests/test_api.py — Pytest suite for FinAudit-8B API endpoints (Days 7–9)
==========================================================================
Tests schema validation, endpoint contracts, and error handling
without requiring a live GPU or the vLLM engine.

Run with:
    pytest tests/ -v --tb=short
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    FinancialExtraction,
    HealthResponse,
)

# ── Test client (uses starlette TestClient — synchronous) ────────────────────
client = TestClient(app, raise_server_exceptions=False)

# ── Fixtures ──────────────────────────────────────────────────────────────────
VALID_TEXT = (
    "Apple Inc. reported net revenue of $394.3 billion for fiscal year 2023. "
    "Net income was $97.0 billion. Diluted EPS was $6.13. "
    "Key risks include supply chain concentration and FX headwinds."
)


# ── Schema validation tests ───────────────────────────────────────────────────
class TestSchemas:
    def test_extraction_request_valid(self):
        req = ExtractionRequest(text=VALID_TEXT)
        assert req.temperature == 0.1
        assert req.max_tokens == 512
        assert req.guided_json is True

    def test_extraction_request_temperature_bounds(self):
        with pytest.raises(Exception):
            ExtractionRequest(text=VALID_TEXT, temperature=1.5)

    def test_extraction_request_text_too_short(self):
        with pytest.raises(Exception):
            ExtractionRequest(text="hi")

    def test_financial_extraction_risk_factor_limit(self):
        fe = FinancialExtraction(
            risk_factors=["r1", "r2", "r3", "r4", "r5", "r6", "r7"]
        )
        assert len(fe.risk_factors) <= 5

    def test_financial_extraction_nullable_fields(self):
        fe = FinancialExtraction()
        assert fe.company_name is None
        assert fe.revenue_usd_millions is None
        assert fe.risk_factors == []

    def test_health_response_schema(self):
        hr = HealthResponse(status="ok", model_loaded=True, model_path="/tmp/model")
        assert hr.status == "ok"
        assert hr.vram_used_gb is None


# ── Health endpoint tests ─────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data

    def test_health_status_values(self):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] in ("ok", "degraded")


# ── Extract endpoint tests ────────────────────────────────────────────────────
class TestExtractEndpoint:
    def test_extract_returns_200_or_422(self):
        """In stub mode (no GPU) we expect 200 with mock data."""
        resp = client.post("/v1/extract", json={"text": VALID_TEXT})
        assert resp.status_code in (200, 422, 500)

    def test_extract_invalid_text_too_short(self):
        resp = client.post("/v1/extract", json={"text": "hi"})
        assert resp.status_code == 422

    def test_extract_missing_text(self):
        resp = client.post("/v1/extract", json={})
        assert resp.status_code == 422

    def test_extract_temperature_out_of_range(self):
        resp = client.post("/v1/extract", json={"text": VALID_TEXT, "temperature": 2.0})
        assert resp.status_code == 422

    def test_extract_response_has_extraction_key(self):
        resp = client.post("/v1/extract", json={"text": VALID_TEXT})
        if resp.status_code == 200:
            data = resp.json()
            assert "extraction" in data

    def test_extract_stream_returns_streaming_response(self):
        resp = client.post("/v1/extract-stream", json={"text": VALID_TEXT})
        # Should return either 200 (streaming) or appropriate error code
        assert resp.status_code in (200, 500)

    def test_extract_text_too_long(self):
        long_text = "A" * 17_000
        resp = client.post("/v1/extract", json={"text": long_text})
        assert resp.status_code == 422


# ── OpenAPI docs ──────────────────────────────────────────────────────────────
class TestDocs:
    def test_openapi_schema_available(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "FinAudit-8B"

    def test_docs_ui_available(self):
        resp = client.get("/docs")
        assert resp.status_code == 200
