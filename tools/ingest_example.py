#!/usr/bin/env python3
"""
Quick extractor: P95 latency and cost per trace from an NDJSON log.

Usage:
    python tools/ingest_example.py \
        --log logs/Investigation_20260310_145903.json \
        --trace 901a0e31-bf80-4fae-9806-fba4b537c37d \
        --print p95_latency,cost_per_trace
"""
import json
import sys
import argparse


def main():
    """Parse an NDJSON log, filter by trace_id, and print P95 latency + total cost."""
    ap = argparse.ArgumentParser(description="Extract P95 latency and cost from an NDJSON log")
    ap.add_argument("--log", required=True, help="Path to the NDJSON log file")
    ap.add_argument("--trace", required=True, help="Trace ID to filter on")
    ap.add_argument("--print", dest="fields", default="p95_latency,cost_per_trace",
                    help="Comma-separated fields to print (p95_latency, cost_per_trace)")
    args = ap.parse_args()

    latencies, cost = [], 0.0
    try:
        with open(args.log, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("trace_id") != args.trace:
                    continue
                d = e.get("details", {})
                if "latency" in d and "end_to_end_ms" in d["latency"]:
                    latencies.append(d["latency"]["end_to_end_ms"])
                if "cost_estimate_usd" in d:
                    cost += d["cost_estimate_usd"]
    except FileNotFoundError:
        sys.exit(f"File not found: {args.log}")

    latencies.sort()
    n = len(latencies)
    p95 = latencies[int(n * 0.95)] if n > 0 else 0
    for field in args.fields.split(","):
        field = field.strip()
        if field == "p95_latency":
            print(f"p95_latency={p95:.2f}ms")
        elif field == "cost_per_trace":
            print(f"cost_per_trace=${cost:.5f}")


if __name__ == "__main__":
    main()
