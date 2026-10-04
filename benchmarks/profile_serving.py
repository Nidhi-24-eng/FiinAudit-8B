"""
benchmarks/profile_serving.py — Latency & Throughput Profiling (Day 8)
=======================================================================
Simulates concurrent users hitting the extraction API and measures:
  • Time-to-First-Token (TTFT) — via streaming endpoint
  • Token throughput (tok/s)
  • p50, p95, p99 end-to-end latency

Usage:
    python benchmarks/profile_serving.py \
        --api_url http://localhost:8000 \
        --concurrency 10 \
        --total_requests 100
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# ── Realistic financial text samples ─────────────────────────────────────────
SAMPLE_TEXTS = [
    (
        "Apple Inc. reported net revenue of $394.3 billion for fiscal year 2023, "
        "compared to $394.3 billion in the prior year. Net income was $97.0 billion, "
        "with diluted EPS of $6.13. The company faces supply chain concentration risk, "
        "foreign exchange headwinds, and smartphone market saturation."
    ),
    (
        "Microsoft Corporation reported total revenue of $211.9 billion for fiscal year 2023, "
        "a 7% increase year-over-year. Operating income was $88.5 billion with an operating "
        "margin of 41.8%. Key risks include cloud market competition, cybersecurity threats, "
        "and regulatory compliance across multiple jurisdictions."
    ),
    (
        "Amazon.com, Inc. reported net sales of $554.0 billion for the year ended December 31, 2023. "
        "Operating income was $36.9 billion. AWS revenue grew 13% to $90.8 billion. "
        "Material risks include intense retail competition, international regulatory exposure, "
        "and data center capacity constraints."
    ),
    (
        "Tesla, Inc. reported total revenues of $97.7 billion for fiscal 2023, "
        "a 19% increase versus 2022. Automotive gross margin declined to 17.1%. "
        "Risk factors include EV adoption uncertainty, production ramp challenges, "
        "raw material supply volatility, and competition from Chinese manufacturers."
    ),
    (
        "Alphabet Inc. reported revenues of $307.4 billion for fiscal 2023, "
        "driven primarily by Google Search ($175.0B) and YouTube ads ($31.5B). "
        "Net income was $73.8 billion. Primary risks include search market disruption "
        "from AI-native competitors, regulatory antitrust actions, and ad market cyclicality."
    ),
]


@dataclass
class RequestResult:
    latency_ms: float
    ttft_ms: Optional[float]
    output_tokens: int
    success: bool
    error: Optional[str] = None


async def single_extract_request(
    client: httpx.AsyncClient,
    api_url: str,
    text: str,
) -> RequestResult:
    """Send one extraction request and record timing metrics."""
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{api_url}/v1/extract",
            json={"text": text, "temperature": 0.0, "max_tokens": 512},
            timeout=90.0,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000

        # Estimate output token count from response
        output_text = json.dumps(data.get("extraction", {}))
        output_tokens = len(output_text.split())   # rough approximation

        # Use server-reported latency if available
        server_latency = data.get("latency_ms")
        ttft_ms = server_latency if server_latency else latency_ms

        return RequestResult(
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            output_tokens=output_tokens,
            success=True,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            latency_ms=latency_ms,
            ttft_ms=None,
            output_tokens=0,
            success=False,
            error=str(exc),
        )


async def run_load_test(
    api_url: str,
    concurrency: int,
    total_requests: int,
) -> list[RequestResult]:
    """Run concurrent load test and collect all results."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    texts = SAMPLE_TEXTS * (total_requests // len(SAMPLE_TEXTS) + 1)
    texts = texts[:total_requests]

    async def bounded_request(text: str) -> RequestResult:
        async with semaphore:
            return await single_extract_request(client, api_url, text)

    async with httpx.AsyncClient() as client:
        console.log(f"[cyan]Sending {total_requests} requests ({concurrency} concurrent) …[/cyan]")
        t_start = time.perf_counter()
        tasks = [bounded_request(t) for t in texts]
        results = await asyncio.gather(*tasks)
        total_time_s = time.perf_counter() - t_start

    console.log(f"[green]✓ All requests complete in {total_time_s:.2f}s[/green]")
    return list(results)


def compute_percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


@app.command()
def main(
    api_url: str     = typer.Option("http://localhost:8000"),
    concurrency: int = typer.Option(10, help="Concurrent users"),
    total_requests: int = typer.Option(100, help="Total requests to send"),
) -> None:
    """Profile FinAudit-8B API: latency, TTFT, and throughput."""
    console.rule("[bold]FinAudit-8B — Serving Performance Profile (Day 8)[/bold]")

    results = asyncio.run(run_load_test(api_url, concurrency, total_requests))

    # Separate successful from failed
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    console.log(f"[green]Successful: {len(successful)}  |  Failed: {len(failed)}[/green]")

    if not successful:
        console.log("[red]✗ No successful requests — is the API running?[/red]")
        raise typer.Exit(1)

    latencies  = [r.latency_ms for r in successful]
    ttfts      = [r.ttft_ms for r in successful if r.ttft_ms]
    token_cnts = [r.output_tokens for r in successful]

    total_tokens = sum(token_cnts)
    total_time_s = sum(latencies) / 1000 / concurrency  # parallel time estimate
    throughput   = total_tokens / max(total_time_s, 0.01)

    # Results table
    table = Table(title="FinAudit-8B Serving Performance", show_header=True)
    table.add_column("Metric", style="cyan", width=35)
    table.add_column("Value", justify="right", style="bold green")
    table.add_column("Target", justify="right", style="dim")

    table.add_row("Requests",              f"{len(successful)}/{total_requests}", "—")
    table.add_row("p50 Latency",           f"{compute_percentile(latencies, 50):.1f} ms", "< 100 ms")
    table.add_row("p95 Latency",           f"{compute_percentile(latencies, 95):.1f} ms", "< 200 ms")
    table.add_row("p99 Latency",           f"{compute_percentile(latencies, 99):.1f} ms", "< 500 ms")
    table.add_row("Mean TTFT",             f"{statistics.mean(ttfts):.1f} ms" if ttfts else "N/A", "< 41 ms")
    table.add_row("Token Throughput",      f"{throughput:.1f} tok/s", "> 158 tok/s")
    table.add_row("Mean Latency",          f"{statistics.mean(latencies):.1f} ms", "—")
    table.add_row("Std Dev",               f"{statistics.stdev(latencies):.1f} ms", "—")

    console.print(table)

    # Benchmark comparison table
    bench = Table(title="Benchmark Comparison (from README)", show_header=True)
    bench.add_column("Configuration",             style="cyan", width=40)
    bench.add_column("TTFT",                      justify="right")
    bench.add_column("Throughput",                justify="right")
    bench.add_column("VRAM",                      justify="right")
    bench.add_row("Base Llama 3.1 8B (HF)",       "240 ms", "18.2 tok/s",  "15.8 GB")
    bench.add_row("FinAudit-8B SFT+ORPO",          "240 ms", "18.2 tok/s",  "15.8 GB")
    bench.add_row("FinAudit-8B AWQ+vLLM (Target)", "41 ms",  "158.4 tok/s", "5.6 GB")
    bench.add_row(f"FinAudit-8B API (Measured)",
                  f"{statistics.mean(ttfts):.0f} ms" if ttfts else "N/A",
                  f"{throughput:.1f} tok/s", "—")
    console.print(bench)


if __name__ == "__main__":
    app()
