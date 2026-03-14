#!/usr/bin/env python3
"""
Test suite for the Consolidated Structured JSON Logging system.
Validates JSON format, trace/request fields, buffer mode, remote ingestion,
telemetry entries, and PII redaction.
"""

import json
import os
import sys
import tempfile
import uuid

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.structured_logger import (
    StructuredLogger,
    trace_id_var,
    span_id_var,
    parent_span_id_var,
    request_id_var,
)

PASS = 0
FAIL = 0

def check(description, condition):
    """Simple assertion helper with pass/fail tracking."""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {description}")
    else:
        FAIL += 1
        print(f"  ❌ {description}")


def test_mandatory_fields_and_request_id():
    """Verify every log line has trace_id, span_id, request_id, and component."""
    print("\n📋 Test 1: Mandatory Fields + request_id")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        log_file = f.name

    try:
        logger = StructuredLogger(component="TestComponent", log_file=log_file)
        trace = logger.init_trace()
        span = logger.start_span("test_operation")
        logger.log_planning("Test planning step", {"goal": "validate format"})
        logger.log_acting("Test action", {"tool": "test_tool"})
        logger.log_synthesis("Test synthesis", {"status": "SUCCESS"})

        with open(log_file, 'r') as f:
            lines = f.readlines()

        check("Log file has entries", len(lines) >= 4)

        for i, line in enumerate(lines):
            entry = json.loads(line.strip())
            check(f"Line {i+1} is valid JSON", True)
            check(f"Line {i+1} has 'trace_id'", "trace_id" in entry)
            check(f"Line {i+1} has 'span_id'", "span_id" in entry)
            check(f"Line {i+1} has 'request_id'", "request_id" in entry)
            check(f"Line {i+1} has 'component'", entry.get("component") == "TestComponent")
            check(f"Line {i+1} 'request_id' is not None", entry["request_id"] is not None)
    finally:
        os.unlink(log_file)


def test_buffer_mode():
    """Verify buffer mode stores entries in memory and flush_buffer returns them."""
    print("\n📋 Test 2: Buffer Mode")

    logger = StructuredLogger(component="BufferTest", buffer=True)
    logger.init_trace()
    logger.start_span("buffer_op")
    logger.log_planning("Buffered planning", {"step": 1})
    logger.log_acting("Buffered action", {"tool": "test"})
    logger.log_synthesis("Buffered synthesis", {"status": "OK"})

    # Flush should return all entries
    entries = logger.flush_buffer()
    check("Buffer has 4 entries (span_start + 3 events)", len(entries) == 4)
    check("All entries are dicts", all(isinstance(e, dict) for e in entries))
    check("First entry is SPAN_START", entries[0].get("phase") == "SPAN_START")

    # After flush, buffer should be empty
    empty = logger.flush_buffer()
    check("Buffer is empty after flush", len(empty) == 0)


def test_ingest_remote_logs():
    """Verify ingest_remote_logs merges node entries into the consolidated log."""
    print("\n📋 Test 3: Remote Log Ingestion")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        log_file = f.name

    try:
        # Orchestrator logger (file mode)
        orch_logger = StructuredLogger(component="Orchestrator", log_file=log_file)
        orch_logger.init_trace()
        orch_logger.start_span("orchestrator_root")
        orch_logger.log_planning("Dispatching to nodes", {"count": 3})

        # Simulate node buffer
        node_logs = [
            {"timestamp": "2026-01-01T00:00:00Z", "trace_id": trace_id_var.get(), "request_id": request_id_var.get(),
             "span_id": "node-span-1", "parent_span_id": None, "component": "NodeAgent-node2",
             "level": "INFO", "phase": "PLANNING", "message": "Received Query"},
            {"timestamp": "2026-01-01T00:00:01Z", "trace_id": trace_id_var.get(), "request_id": request_id_var.get(),
             "span_id": "node-span-1", "parent_span_id": None, "component": "NodeAgent-node2",
             "level": "INFO", "phase": "SYNTHESIS", "message": "Node query successful"},
        ]

        orch_logger.ingest_remote_logs(node_logs)
        orch_logger.log_synthesis("Investigation complete", {"status": "SUCCESS"})

        with open(log_file, 'r') as f:
            lines = f.readlines()

        entries = [json.loads(line.strip()) for line in lines]

        # Should have: span_start + planning + 2 node logs + synthesis = 5
        check("Consolidated log has 5 entries", len(entries) == 5)

        components = [e["component"] for e in entries]
        check("Contains Orchestrator entries", "Orchestrator" in components)
        check("Contains NodeAgent-node2 entries", "NodeAgent-node2" in components)

        # All should share the same trace_id
        trace_ids = set(e["trace_id"] for e in entries)
        check("All entries share one trace_id", len(trace_ids) == 1)

    finally:
        os.unlink(log_file)


def test_telemetry_fields():
    """Verify telemetry entries contain usage, cost, and latency."""
    print("\n📋 Test 4: Telemetry Fields")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        log_file = f.name

    try:
        logger = StructuredLogger(component="TelemetryTest", log_file=log_file)
        logger.init_trace()
        logger.start_span("telemetry_test")
        logger.log_telemetry(model="openai:gpt-4o", prompt_tokens=500, completion_tokens=200,
                             duration_ms=1234.5, ttft_ms=150.0)

        with open(log_file, 'r') as f:
            lines = f.readlines()

        telemetry_entries = [json.loads(l.strip()) for l in lines if '"TELEMETRY"' in l]
        check("Telemetry entry found", len(telemetry_entries) >= 1)

        if telemetry_entries:
            details = telemetry_entries[0].get("details", {})
            check("Has model field", details.get("model") == "openai:gpt-4o")
            check("Has prompt_tokens", details.get("usage", {}).get("prompt_tokens") == 500)
            check("Has completion_tokens", details.get("usage", {}).get("completion_tokens") == 200)
            check("Has total_tokens", details.get("usage", {}).get("total_tokens") == 700)
            check("Has cost_estimate_usd", isinstance(details.get("cost_estimate_usd"), float))
            check("Has end_to_end_ms", details.get("latency", {}).get("end_to_end_ms") == 1234.5)
            check("Has ttft_ms", details.get("latency", {}).get("ttft_ms") == 150.0)
    finally:
        os.unlink(log_file)


def test_pii_redaction():
    """Verify SSNs, emails, and API keys are redacted."""
    print("\n📋 Test 5: PII Redaction")

    logger = StructuredLogger(component="RedactionTest", buffer=True)
    logger.init_trace()
    logger.start_span("redaction_test")

    logger.log_event("INFO", "TEST", "Patient SSN is 123-45-6789")
    logger.log_event("INFO", "TEST", "Contact user@example.com for info")
    logger.log_event("INFO", "TEST", "api_key: sk_abcdefghijklmnopqrstuvwxyz1234")
    logger.log_event("INFO", "TEST", "password= SuperSecret123!")

    entries = logger.flush_buffer()
    # Skip span_start entry
    data_entries = entries[1:]

    messages = [e.get("message", "") for e in data_entries]
    all_text = " ".join(messages)

    check("No raw SSN", "123-45-6789" not in all_text)
    check("No raw email", "user@example.com" not in all_text)
    check("SSN replaced", "[REDACTED_SSN]" in all_text)
    check("Email replaced", "[REDACTED_EMAIL]" in all_text)


def test_lifecycle_phases():
    """Verify PLANNING, ACTING, SYNTHESIS phases are recorded."""
    print("\n📋 Test 6: Lifecycle Phases")

    logger = StructuredLogger(component="LifecycleTest", buffer=True)
    logger.init_trace()
    logger.start_span("lifecycle")
    logger.log_planning("Goal parsed", {"query": "test"})
    logger.log_acting("Tool called", {"tool": "fisher_test"})
    logger.log_synthesis("Result ready", {"status": "SUCCESS"})

    entries = logger.flush_buffer()
    phases = [e["phase"] for e in entries]

    check("SPAN_START present", "SPAN_START" in phases)
    check("PLANNING present", "PLANNING" in phases)
    check("ACTING present", "ACTING" in phases)
    check("SYNTHESIS present", "SYNTHESIS" in phases)


def test_no_file_io_in_buffer_mode():
    """Verify buffer mode does not create any files."""
    print("\n📋 Test 7: No File I/O in Buffer Mode")

    logger = StructuredLogger(component="NoFileTest", buffer=True)
    logger.init_trace()
    logger.start_span("no_file")
    logger.log_event("INFO", "TEST", "this should not create a file")

    check("log_file is None", logger.log_file is None)
    check("Buffer has entries", len(logger._buffer) > 0)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Consolidated Logging Test Suite")
    print("=" * 60)

    test_mandatory_fields_and_request_id()
    test_buffer_mode()
    test_ingest_remote_logs()
    test_telemetry_fields()
    test_pii_redaction()
    test_lifecycle_phases()
    test_no_file_io_in_buffer_mode()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    else:
        print("\n  🎉 All tests passed!")
        sys.exit(0)
