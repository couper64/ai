"""Benchmark the served image classifier API."""

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark the image classifier FastAPI service.")
    parser.add_argument("image_path", type=Path, help="Image file to upload to the prediction endpoint.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="Prediction endpoint URL.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to request.")
    parser.add_argument("--requests", type=int, default=50, help="Measured requests to send.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests to send before measuring.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--metrics-output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a percentile from a non-empty sorted sample."""
    if not values:
        return 0.0
    index = round((len(values) - 1) * percentile_value)
    return sorted(values)[index]


def post_prediction(client: httpx.Client, url: str, image_path: Path, image_bytes: bytes, top_k: int) -> httpx.Response:
    """Send one prediction request."""
    files = {"file": (image_path.name, image_bytes, "application/octet-stream")}
    return client.post(url, params={"top_k": top_k}, files=files)


def run_benchmark(
    *,
    url: str,
    image_path: Path,
    top_k: int,
    request_count: int,
    warmup_count: int,
    timeout: float,
) -> dict[str, Any]:
    """Run sequential HTTP prediction requests and return latency metrics."""
    if request_count <= 0:
        raise ValueError("--requests must be greater than 0.")
    if warmup_count < 0:
        raise ValueError("--warmup must be greater than or equal to 0.")
    if top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bytes = image_path.read_bytes()
    latencies_ms: list[float] = []
    failures = 0
    last_response: dict[str, Any] | None = None

    with httpx.Client(timeout=timeout) as client:
        for _ in range(warmup_count):
            response = post_prediction(client=client, url=url, image_path=image_path, image_bytes=image_bytes, top_k=top_k)
            response.raise_for_status()

        started_at = time.perf_counter()
        for _ in range(request_count):
            request_started_at = time.perf_counter()
            try:
                response = post_prediction(
                    client=client,
                    url=url,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    top_k=top_k,
                )
                response.raise_for_status()
                last_response = response.json()
            except httpx.HTTPError:
                failures += 1
            latencies_ms.append((time.perf_counter() - request_started_at) * 1000)
        total_seconds = time.perf_counter() - started_at

    successes = request_count - failures
    return {
        "url": url,
        "image_path": str(image_path),
        "top_k": top_k,
        "requests": request_count,
        "warmup": warmup_count,
        "successes": successes,
        "failures": failures,
        "total_seconds": total_seconds,
        "requests_per_second": request_count / total_seconds if total_seconds else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "p90": percentile(latencies_ms, 0.90),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
        "last_response": last_response,
    }


def print_report(metrics: dict[str, Any]) -> None:
    """Print a compact human-readable benchmark report."""
    latency = metrics["latency_ms"]
    print(f"Benchmarking {metrics['url']}")
    print(f"Image: {metrics['image_path']}")
    print(f"Requests: {metrics['requests']} measured, {metrics['warmup']} warmup")
    print(f"Successes: {metrics['successes']}")
    print(f"Failures: {metrics['failures']}")
    print(f"Throughput: {metrics['requests_per_second']:.2f} requests/sec")
    print(
        "Latency (ms): "
        f"mean={latency['mean']:.2f} "
        f"median={latency['median']:.2f} "
        f"p90={latency['p90']:.2f} "
        f"p95={latency['p95']:.2f} "
        f"p99={latency['p99']:.2f} "
        f"min={latency['min']:.2f} "
        f"max={latency['max']:.2f}"
    )


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()
    metrics = run_benchmark(
        url=args.url,
        image_path=args.image_path,
        top_k=args.top_k,
        request_count=args.requests,
        warmup_count=args.warmup,
        timeout=args.timeout,
    )
    print_report(metrics)

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
