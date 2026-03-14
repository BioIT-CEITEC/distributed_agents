#!/usr/bin/env python3
"""
Flask backend for the AI Orchestration Metrics Dashboard.

Dynamically loads log files from logs/ directory, computes per-request and
aggregate metrics, then serves them via REST APIs for the Highcharts frontend.
"""

import json
import re
import os
import glob
from collections import defaultdict
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request as flask_request

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Label simplification map
# Converts internal component/node IDs into readable dashboard labels
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "Orchestrator": "Orchestrator",
    "NodeAgent-node2": "Node 2",
    "NodeAgent-node3": "Node 3",
    "NodeAgent-node4": "Node 4",
    "node2": "Node 2",
    "node3": "Node 3",
    "node4": "Node 4",
}


def simplify_label(component, message=""):
    """Convert internal component/node IDs to human-readable labels.
    Orchestrator spans that mention 'home' are attributed to 'Home Node'.
    """
    if component == "Orchestrator" and "home" in message.lower():
        return "Home Node"
    return LABEL_MAP.get(component, component)


# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------

PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    re.compile(r"\b(?<!\.)(?<!\d)\d{10,15}(?!\d)(?!\.)\b"),
    re.compile(r"\bnode\d+_P\d+\b", re.IGNORECASE),
]

INJECTION_PATTERNS = [
    re.compile(r"\{\{"),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.system\b", re.IGNORECASE),
    re.compile(r"rm\s+-rf"),
    re.compile(r";\s*\w"),
    re.compile(r"`[^`]*(curl|wget|bash)[^`]*`", re.IGNORECASE),
]

EXPECTED_PHASE_ORDER = ["PLANNING", "ACTING", "TELEMETRY", "SYNTHESIS"]

# ---------------------------------------------------------------------------
# Log file discovery
# ---------------------------------------------------------------------------


def list_log_files():
    """Scan logs/ directory and return list of .json log filenames,
    sorted by modification time (most recent first)."""
    try:
        files = [
            f for f in os.listdir(LOGS_DIR)
            if f.endswith(".json") and os.path.isfile(os.path.join(LOGS_DIR, f))
        ]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(LOGS_DIR, f)), reverse=True)
        return files
    except FileNotFoundError:
        print(f"WARNING: Logs directory not found: {LOGS_DIR}")
        return []


def get_latest_log_file():
    """Return the most recent log file path, or None."""
    files = list_log_files()
    return os.path.join(LOGS_DIR, files[0]) if files else None


# ---------------------------------------------------------------------------
# Log loading & grouping
# ---------------------------------------------------------------------------


def load_log(path: str) -> list[dict]:
    """Load NDJSON log file into a list of dicts."""
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"WARNING: Log file not found: {path}")
    return entries


def group_by_request(entries: list[dict]) -> dict[str, list[dict]]:
    """Group spans by request_id."""
    groups = defaultdict(list)
    for e in entries:
        rid = e.get("request_id")
        if rid:
            groups[rid].append(e)
    return dict(groups)


# ---------------------------------------------------------------------------
# Query & intent extraction
# ---------------------------------------------------------------------------


def extract_queries(entries: list[dict]) -> list[dict]:
    """Extract unique queries with their intents from Orchestrator
    Routing Decision / Goal Parsing spans. Returns one entry per request_id."""
    queries = {}
    for e in entries:
        if (e.get("component") == "Orchestrator"
                and e.get("phase") == "PLANNING"
                and e.get("message") in ("Routing Decision", "Goal Parsing")):
            rid = e.get("request_id", "")
            d = e.get("details", {})
            q = d.get("query", "")
            intent = d.get("intent", "")
            if rid and rid not in queries and q:
                queries[rid] = {
                    "request_id": rid,
                    "query": q.strip(),
                    "intent": intent,
                }
    return list(queries.values())


# ---------------------------------------------------------------------------
# Per-request metric computation (with simplified labels & home node)
# ---------------------------------------------------------------------------


def compute_e2e_ms(spans):
    """max(details.latency.end_to_end_ms) across spans."""
    vals = [
        s["details"]["latency"]["end_to_end_ms"]
        for s in spans
        if isinstance(s.get("details", {}).get("latency"), dict)
        and "end_to_end_ms" in s["details"]["latency"]
    ]
    return round(max(vals), 2) if vals else None


def compute_cost(spans):
    """sum(details.cost_estimate_usd) across spans."""
    vals = [
        s["details"]["cost_estimate_usd"]
        for s in spans
        if "cost_estimate_usd" in s.get("details", {})
    ]
    return round(sum(vals), 6) if vals else None


def compute_tokens(spans):
    """Aggregate token counts from details.usage."""
    prompt, completion, total = 0, 0, 0
    for s in spans:
        u = s.get("details", {}).get("usage", {})
        if u:
            prompt += u.get("prompt_tokens", 0)
            completion += u.get("completion_tokens", 0)
            total += u.get("total_tokens", 0)
    return {"prompt": prompt, "completion": completion, "total": total}


def compute_agent_success(spans):
    """Success rate of NodeAgent SYNTHESIS spans."""
    total, success = 0, 0
    for s in spans:
        if s.get("component", "").startswith("NodeAgent") and s.get("phase") == "SYNTHESIS":
            total += 1
            d = s.get("details", {})
            if d.get("has_data") is True and d.get("status") == "SUCCESS":
                success += 1
    return {"total": total, "successful": success,
            "rate": round(success / total, 4) if total else None}


def compute_nodes_ratio(spans):
    """nodes_with_data / nodes_queried from outputs."""
    for s in reversed(spans):
        o = s.get("details", {}).get("outputs", {})
        if isinstance(o, dict) and o.get("nodes_queried"):
            return {
                "queried": o["nodes_queried"],
                "with_data": o.get("nodes_with_data", 0),
                "ratio": round(o.get("nodes_with_data", 0) / o["nodes_queried"], 4),
            }
    return None


def compute_pii(spans):
    """Count PII regex matches across message + details."""
    count = 0
    for s in spans:
        text = s.get("message", "") + " " + json.dumps(s.get("details", {}))
        for pat in PII_PATTERNS:
            count += len(pat.findall(text))
    return count


def compute_injection(spans):
    """Prompt injection suspicion score."""
    total_words, suspicious = 0, 0
    for s in spans:
        text = s.get("message", "") + " " + json.dumps(s.get("details", {}))
        total_words += len(text.split())
        for pat in INJECTION_PATTERNS:
            suspicious += len(pat.findall(text))
    return round(suspicious / total_words, 6) if total_words else 0.0


def compute_step_latencies(spans):
    """Per-span latency breakdown with simplified labels, sorted descending."""
    results = []
    for s in spans:
        lat = s.get("details", {}).get("latency")
        if isinstance(lat, dict) and "end_to_end_ms" in lat:
            comp = s.get("component", "")
            msg = s.get("message", "")
            results.append({
                "component": simplify_label(comp, msg),
                "span_id": s.get("span_id", "")[:8],
                "e2e_ms": round(lat["end_to_end_ms"], 2),
                "phase": s.get("phase", ""),
            })
    results.sort(key=lambda x: x["e2e_ms"], reverse=True)
    return results


def compute_cost_by_component(spans):
    """Cost broken down by simplified component label, with home node split."""
    costs = defaultdict(float)
    for s in spans:
        c = s.get("details", {}).get("cost_estimate_usd")
        if c is not None:
            comp = s.get("component", "unknown")
            msg = s.get("message", "")
            label = simplify_label(comp, msg)
            costs[label] += c
    return {k: round(v, 6) for k, v in costs.items()}


def compute_tokens_by_component(spans):
    """Token usage by simplified component label, with home node split."""
    data = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
    for s in spans:
        u = s.get("details", {}).get("usage", {})
        if u:
            comp = s.get("component", "unknown")
            msg = s.get("message", "")
            label = simplify_label(comp, msg)
            data[label]["prompt"] += u.get("prompt_tokens", 0)
            data[label]["completion"] += u.get("completion_tokens", 0)
            data[label]["total"] += u.get("total_tokens", 0)
    return dict(data)


def compute_error_count(spans):
    """Count ERROR-level spans."""
    return sum(1 for s in spans if s.get("level") == "ERROR")


def compute_task_completion(spans):
    """1 if any SYNTHESIS phase span exists."""
    return 1 if any(s.get("phase") == "SYNTHESIS" for s in spans) else 0


def compute_network_latencies(spans):
    """Network latencies with simplified node labels."""
    results = []
    for s in spans:
        nl = s.get("details", {}).get("network_latency_ms")
        if nl is not None:
            raw_id = s.get("details", {}).get("node_id", "unknown")
            results.append({
                "node_id": simplify_label(raw_id),
                "latency_ms": round(nl, 2),
                "timestamp": s.get("timestamp", ""),
            })
    return results


def get_latest_ts(spans):
    """Latest timestamp from spans."""
    ts = [s.get("timestamp", "") for s in spans if s.get("timestamp")]
    return max(ts) if ts else None


# ---------------------------------------------------------------------------
# Scope helpers — group raw entries by request_id, trace_id, or span_id
# ---------------------------------------------------------------------------

def group_by_trace(entries: list[dict]) -> dict[str, list[dict]]:
    """Group spans by trace_id."""
    groups = defaultdict(list)
    for e in entries:
        tid = e.get("trace_id")
        if tid:
            groups[tid].append(e)
    return dict(groups)


def group_by_span(entries: list[dict]) -> list[dict]:
    """Return each span as its own 'group' (list of one) keyed by span_id."""
    groups = {}
    for e in entries:
        sid = e.get("span_id")
        if sid:
            groups[sid] = [e]
    return groups


def get_scoped_groups(entries, scope):
    """Return groups dict based on scope: requests | traces | queries."""
    if scope == "traces":
        return group_by_trace(entries)
    elif scope == "queries":
        return group_by_span(entries)
    else:
        return group_by_request(entries)


# ---------------------------------------------------------------------------
# Build full metrics payload
# ---------------------------------------------------------------------------


def build_all_metrics(entries, groups):
    """Compute metrics for all requests and aggregate."""
    per_request = []
    all_step_latencies = []
    all_network_latencies = []
    all_cost_by_comp = defaultdict(float)
    all_tokens_by_comp = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})

    for rid, spans in groups.items():
        tids = list({s.get("trace_id") for s in spans if s.get("trace_id")})
        e2e = compute_e2e_ms(spans)
        cost = compute_cost(spans)
        tokens = compute_tokens(spans)
        agent = compute_agent_success(spans)
        nodes = compute_nodes_ratio(spans)
        pii = compute_pii(spans)
        injection = compute_injection(spans)
        steps = compute_step_latencies(spans)
        cost_comp = compute_cost_by_component(spans)
        tok_comp = compute_tokens_by_component(spans)
        errors = compute_error_count(spans)
        task_comp = compute_task_completion(spans)
        net_lats = compute_network_latencies(spans)

        # Accumulate component-level data
        for k, v in cost_comp.items():
            all_cost_by_comp[k] += v
        for k, v in tok_comp.items():
            all_tokens_by_comp[k]["prompt"] += v["prompt"]
            all_tokens_by_comp[k]["completion"] += v["completion"]
            all_tokens_by_comp[k]["total"] += v["total"]

        for sl in steps:
            sl["request_id"] = rid[:12]
            all_step_latencies.append(sl)

        for nl in net_lats:
            nl["request_id"] = rid[:12]
            all_network_latencies.append(nl)

        per_request.append({
            "request_id": rid,
            "trace_id": tids[0] if tids else None,
            "timestamp": get_latest_ts(spans),
            "span_count": len(spans),
            "e2e_ms": e2e,
            "cost_usd": cost,
            "tokens": tokens,
            "agent_success": agent,
            "nodes_ratio": nodes,
            "pii_count": pii,
            "injection_score": injection,
            "error_count": errors,
            "task_completion": task_comp,
            "step_latencies": steps,
            "cost_by_component": cost_comp,
            "network_latencies": net_lats,
        })

    # Compute aggregate KPIs
    timestamps = []
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass

    window_sec = (
        (max(timestamps) - min(timestamps)).total_seconds()
        if len(timestamps) >= 2 else 60
    )
    throughput = round(len(groups) / window_sec, 6) if window_sec > 0 else 0

    e2e_vals = [r["e2e_ms"] for r in per_request if r["e2e_ms"] is not None]
    cost_vals = [r["cost_usd"] for r in per_request if r["cost_usd"] is not None]
    success_vals = [
        r["agent_success"]["rate"]
        for r in per_request
        if r["agent_success"]["rate"] is not None
    ]

    kpis = {
        "total_requests": len(groups),
        "total_spans": len(entries),
        "throughput_rps": throughput,
        "window_seconds": round(window_sec, 2),
        "avg_e2e_ms": round(sum(e2e_vals) / len(e2e_vals), 2) if e2e_vals else None,
        "max_e2e_ms": round(max(e2e_vals), 2) if e2e_vals else None,
        "total_cost_usd": round(sum(cost_vals), 6) if cost_vals else None,
        "avg_cost_usd": round(sum(cost_vals) / len(cost_vals), 6) if cost_vals else None,
        "avg_agent_success": (
            round(sum(success_vals) / len(success_vals), 4) if success_vals else None
        ),
        "total_pii": sum(r["pii_count"] for r in per_request),
        "total_errors": sum(r["error_count"] for r in per_request),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "kpis": kpis,
        "per_request": per_request,
        "all_step_latencies": all_step_latencies,
        "all_network_latencies": all_network_latencies,
        "cost_by_component": {k: round(v, 6) for k, v in all_cost_by_comp.items()},
        "tokens_by_component": dict(all_tokens_by_comp),
    }


# ---------------------------------------------------------------------------
# Global state — loaded on startup, reloadable via /api/load-log
# ---------------------------------------------------------------------------

CURRENT_LOG = None
ENTRIES = []
GROUPS = {}
METRICS = {}
QUERIES = []


def reload_data(log_path):
    """Reload all global data from the given log file path."""
    global CURRENT_LOG, ENTRIES, GROUPS, METRICS, QUERIES
    CURRENT_LOG = log_path
    ENTRIES = load_log(log_path)
    GROUPS = group_by_request(ENTRIES)
    METRICS = build_all_metrics(ENTRIES, GROUPS)
    QUERIES = extract_queries(ENTRIES)
    filename = os.path.basename(log_path)
    print(f"  Loaded {filename}: {len(ENTRIES)} spans, {len(GROUPS)} request(s), {len(QUERIES)} query(ies).")


# Auto-load the most recent log file on startup
_startup_log = get_latest_log_file()
if _startup_log:
    print(f"Auto-detected latest log: {os.path.basename(_startup_log)}")
    reload_data(_startup_log)
else:
    print("WARNING: No log files found in logs/ directory.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def dashboard():
    """Serve the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/api/logs")
def api_logs():
    """List all available log files in the logs/ directory."""
    files = list_log_files()
    current = os.path.basename(CURRENT_LOG) if CURRENT_LOG else None
    return jsonify({"files": files, "current": current})


@app.route("/api/load-log", methods=["POST"])
def api_load_log():
    """Hot-reload a different log file without restarting the server."""
    data = flask_request.get_json(silent=True) or {}
    filename = data.get("filename", "")
    # Security: only allow filenames (no path traversal)
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    path = os.path.join(LOGS_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"error": "File not found"}), 404
    reload_data(path)
    return jsonify({"status": "ok", "filename": filename,
                    "spans": len(ENTRIES), "requests": len(GROUPS),
                    "queries": len(QUERIES)})


@app.route("/api/queries-list")
def api_queries_list():
    """Return extracted queries with intents for the current log."""
    return jsonify({"queries": QUERIES})


@app.route("/api/metrics")
def api_metrics():
    """Overview KPIs + request list for filtering."""
    scope = flask_request.args.get("scope", "requests")
    req_id = flask_request.args.get("request_id")

    scoped = get_scoped_groups(ENTRIES, scope)
    scoped_metrics = build_all_metrics(ENTRIES, scoped)

    if req_id and req_id != "all":
        filtered = [r for r in scoped_metrics["per_request"] if r["request_id"] == req_id]
    else:
        filtered = scoped_metrics["per_request"]

    return jsonify({
        "kpis": scoped_metrics["kpis"],
        "requests": [
            {"request_id": r["request_id"], "trace_id": r.get("trace_id")}
            for r in scoped_metrics["per_request"]
        ],
    })


@app.route("/api/latency")
def api_latency():
    """Latency metrics: E2E per request, step breakdown, network latencies."""
    scope = flask_request.args.get("scope", "requests")
    req_id = flask_request.args.get("request_id")
    scoped = get_scoped_groups(ENTRIES, scope)
    scoped_metrics = build_all_metrics(ENTRIES, scoped)

    if req_id and req_id != "all":
        reqs = [r for r in scoped_metrics["per_request"] if r["request_id"] == req_id]
        steps = [s for s in scoped_metrics["all_step_latencies"]
                 if s["request_id"] == req_id[:12]]
        nets = [n for n in scoped_metrics["all_network_latencies"]
                if n["request_id"] == req_id[:12]]
    else:
        reqs = scoped_metrics["per_request"]
        steps = scoped_metrics["all_step_latencies"]
        nets = scoped_metrics["all_network_latencies"]

    return jsonify({
        "per_request": [
            {"request_id": r["request_id"][:12], "e2e_ms": r["e2e_ms"],
             "timestamp": r["timestamp"]}
            for r in reqs
        ],
        "step_latencies": steps[:20],
        "network_latencies": nets,
    })


@app.route("/api/cost")
def api_cost():
    """Cost metrics: per request, per component."""
    scope = flask_request.args.get("scope", "requests")
    req_id = flask_request.args.get("request_id")
    scoped = get_scoped_groups(ENTRIES, scope)
    scoped_metrics = build_all_metrics(ENTRIES, scoped)

    if req_id and req_id != "all":
        reqs = [r for r in scoped_metrics["per_request"] if r["request_id"] == req_id]
        comp = reqs[0]["cost_by_component"] if reqs else {}
    else:
        reqs = scoped_metrics["per_request"]
        comp = scoped_metrics["cost_by_component"]

    return jsonify({
        "per_request": [
            {"request_id": r["request_id"][:12], "cost_usd": r["cost_usd"]}
            for r in reqs
        ],
        "by_component": comp,
        "total": round(sum(r["cost_usd"] or 0 for r in reqs), 6),
    })


@app.route("/api/agents")
def api_agents():
    """Agent performance: success rate, nodes ratio, task completion."""
    scope = flask_request.args.get("scope", "requests")
    req_id = flask_request.args.get("request_id")
    scoped = get_scoped_groups(ENTRIES, scope)
    scoped_metrics = build_all_metrics(ENTRIES, scoped)

    if req_id and req_id != "all":
        reqs = [r for r in scoped_metrics["per_request"] if r["request_id"] == req_id]
    else:
        reqs = scoped_metrics["per_request"]

    total_agents = sum(r["agent_success"]["total"] for r in reqs)
    success_agents = sum(r["agent_success"]["successful"] for r in reqs)

    return jsonify({
        "overall": {
            "total": total_agents,
            "successful": success_agents,
            "rate": round(success_agents / total_agents, 4) if total_agents else 0,
        },
        "per_request": [
            {
                "request_id": r["request_id"][:12],
                "success_rate": r["agent_success"]["rate"],
                "nodes_queried": r["nodes_ratio"]["queried"] if r["nodes_ratio"] else 0,
                "nodes_with_data": r["nodes_ratio"]["with_data"] if r["nodes_ratio"] else 0,
                "task_completion": r["task_completion"],
            }
            for r in reqs
        ],
        "task_completion": {
            "completed": sum(r["task_completion"] for r in reqs),
            "total": len(reqs),
            "rate": round(
                sum(r["task_completion"] for r in reqs) / len(reqs), 4
            ) if reqs else 0,
        },
    })


@app.route("/api/tokens")
def api_tokens():
    """Token usage: per request, per component."""
    scope = flask_request.args.get("scope", "requests")
    req_id = flask_request.args.get("request_id")
    scoped = get_scoped_groups(ENTRIES, scope)
    scoped_metrics = build_all_metrics(ENTRIES, scoped)

    if req_id and req_id != "all":
        reqs = [r for r in scoped_metrics["per_request"] if r["request_id"] == req_id]
    else:
        reqs = scoped_metrics["per_request"]

    return jsonify({
        "per_request": [
            {
                "request_id": r["request_id"][:12],
                "prompt": r["tokens"]["prompt"],
                "completion": r["tokens"]["completion"],
                "total": r["tokens"]["total"],
            }
            for r in reqs
        ],
        "by_component": scoped_metrics["tokens_by_component"],
    })


@app.route("/api/safety")
def api_safety():
    """Safety metrics: PII, injection, errors."""
    scope = flask_request.args.get("scope", "requests")
    req_id = flask_request.args.get("request_id")
    scoped = get_scoped_groups(ENTRIES, scope)
    scoped_metrics = build_all_metrics(ENTRIES, scoped)

    if req_id and req_id != "all":
        reqs = [r for r in scoped_metrics["per_request"] if r["request_id"] == req_id]
    else:
        reqs = scoped_metrics["per_request"]

    return jsonify({
        "pii": [
            {"request_id": r["request_id"][:12], "count": r["pii_count"],
             "flag": 1 if r["pii_count"] > 0 else 0}
            for r in reqs
        ],
        "injection": [
            {"request_id": r["request_id"][:12], "score": r["injection_score"]}
            for r in reqs
        ],
        "errors": [
            {"request_id": r["request_id"][:12], "count": r["error_count"]}
            for r in reqs
        ],
        "totals": {
            "pii": sum(r["pii_count"] for r in reqs),
            "errors": sum(r["error_count"] for r in reqs),
            "requests_with_pii": sum(1 for r in reqs if r["pii_count"] > 0),
        },
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
