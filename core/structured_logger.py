import json
import logging
import contextvars
import uuid
import re
import time
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime

# ============================================================================
# Context variables for distributed tracing
# ============================================================================
trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("trace_id", default=None)
span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("span_id", default=None)
parent_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("parent_span_id", default=None)
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)

# ============================================================================
# Regular expressions for PII / secret redaction
# ============================================================================
REDACTION_PATTERNS = [
    # API Keys / Secrets
    (re.compile(r'(?i)(api[_-]?key[\s:=]+[\'\"]?)([a-zA-Z0-9_\-]{20,})([\'\"]?)'), r'\1[REDACTED_API_KEY]\3'),
    (re.compile(r'(?i)(secret[\s:=]+[\'\"]?)([a-zA-Z0-9_\-]{20,})([\'\"]?)'), r'\1[REDACTED_SECRET]\3'),
    (re.compile(r'(?i)(token[\s:=]+[\'\"]?)([a-zA-Z0-9_\-\.]{20,})([\'\"]?)'), r'\1[REDACTED_TOKEN]\3'),
    (re.compile(r'(?i)(password[\s:=]+[\'\"]?)([^"\'\s]+)([\'\"]?)'), r'\1[REDACTED_PASSWORD]\3'),
    # PII (SSN, Email)
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[REDACTED_SSN]'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[REDACTED_EMAIL]'),
]


class StructuredLogger:
    """
    Consolidated JSON structured logger with two operating modes:
    
    1. File mode (orchestrator): Writes newline-delimited JSON to a single log file.
       All node logs are ingested into this file via ingest_remote_logs().
    
    2. Buffer mode (node agents): Stores log entries in an in-memory list.
       The buffer is flushed and returned to the orchestrator via flush_buffer().
    
    Every entry includes trace_id, span_id, parent_span_id, request_id, and component
    for strict correlation across concurrent users.
    """
    def __init__(self, component: str, log_file: Optional[str] = None, buffer: bool = False):
        """
        Args:
            component: Identifier (e.g. 'Orchestrator', 'NodeAgent-node2')
            log_file: Path to the consolidated log file. If set, entries are written to disk.
            buffer: If True, entries are buffered in-memory (for node agents).
        """
        self.component = component
        self.log_file = log_file
        self.buffer_mode = buffer
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._file_lock = threading.Lock()
        self.logger = logging.getLogger(f"StructuredLogger-{component}")

    # ========================================================================
    # Redaction
    # ========================================================================
    def _redact_string(self, text: str) -> str:
        """Apply all redaction patterns to a single string."""
        for pattern, replacement in REDACTION_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _redact(self, obj: Any) -> Any:
        """Recursively redact strings within dicts, lists, or plain strings."""
        if isinstance(obj, str):
            return self._redact_string(obj)
        elif isinstance(obj, dict):
            return {k: self._redact(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._redact(v) for v in obj]
        return obj

    # ========================================================================
    # Context
    # ========================================================================
    def _get_context(self) -> Dict[str, Any]:
        """Build the mandatory header for every log entry."""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id_var.get(),
            "request_id": request_id_var.get(),
            "span_id": span_id_var.get(),
            "parent_span_id": parent_span_id_var.get(),
            "component": self.component,
        }

    # ========================================================================
    # Trace / Span management
    # ========================================================================
    def init_trace(self, existing_trace_id: Optional[str] = None) -> str:
        """Initialize or continue a trace. Sets trace_id_var and request_id_var."""
        trace_id = existing_trace_id or str(uuid.uuid4())
        trace_id_var.set(trace_id)
        if not existing_trace_id:
            parent_span_id_var.set(None)
        # Generate a new request_id for this invocation
        request_id_var.set(str(uuid.uuid4()))
        return trace_id

    def start_span(self, name: str) -> str:
        """Start a new span within the current trace context."""
        new_span_id = str(uuid.uuid4())
        current_span = span_id_var.get()
        parent_span_id_var.set(current_span)
        span_id_var.set(new_span_id)
        self.log_event("INFO", "SPAN_START", f"Starting span: {name}", span_name=name)
        return new_span_id

    def end_span(self, previous_span_id: Optional[str], previous_parent_id: Optional[str]):
        """Pop the span context back (call in a finally block)."""
        span_id_var.set(previous_span_id)
        parent_span_id_var.set(previous_parent_id)

    # ========================================================================
    # Core logging
    # ========================================================================
    def log_event(self, level: str, phase: str, message: str, **kwargs):
        """
        Log a structured JSON event.
        Writes to file (orchestrator) or appends to buffer (node).
        """
        entry = self._get_context()
        entry.update({
            "level": level.upper(),
            "phase": phase,
            "message": self._redact(message)
        })

        if kwargs:
            entry["details"] = self._redact(kwargs)

        # Route to the appropriate sink
        if self.buffer_mode:
            with self._buffer_lock:
                self._buffer.append(entry)
        else:
            self._write_entry(entry)

    def _write_entry(self, entry: Dict[str, Any]):
        """Write a single JSON entry to file and Python logger."""
        log_json = json.dumps(entry, default=str)

        # Console output via Python logger
        level = entry.get("level", "INFO")
        if level == "DEBUG":
            self.logger.debug(log_json)
        elif level in ("WARN", "WARNING"):
            self.logger.warning(log_json)
        elif level == "ERROR":
            self.logger.error(log_json)
        else:
            self.logger.info(log_json)

        # Append to consolidated log file (thread-safe)
        if self.log_file:
            with self._file_lock:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(log_json + "\n")
                except Exception as e:
                    self.logger.error(f"Failed to write to log file: {e}")

    # ========================================================================
    # Buffer mode operations (for node agents)
    # ========================================================================
    def flush_buffer(self) -> List[Dict[str, Any]]:
        """Return all buffered log entries and clear the buffer."""
        with self._buffer_lock:
            entries = list(self._buffer)
            self._buffer.clear()
        return entries

    # ========================================================================
    # Remote log ingestion (for orchestrator)
    # ========================================================================
    def ingest_remote_logs(self, logs: List[Dict[str, Any]]):
        """
        Merge log entries from a remote node into the consolidated log file.
        Each entry is written as-is (already has trace_id, request_id, etc).
        """
        for entry in logs:
            self._write_entry(entry)

    # ========================================================================
    # Convenience wrappers
    # ========================================================================
    def log_planning(self, action: str, details: Dict[str, Any]):
        """Log a PLANNING phase event."""
        self.log_event("INFO", "PLANNING", action, **details)

    def log_acting(self, action: str, details: Dict[str, Any]):
        """Log an ACTING phase event."""
        self.log_event("INFO", "ACTING", action, **details)

    def log_synthesis(self, action: str, details: Dict[str, Any]):
        """Log a SYNTHESIS phase event."""
        self.log_event("INFO", "SYNTHESIS", action, **details)

    def log_telemetry(self, model: str, prompt_tokens: int, completion_tokens: int, duration_ms: float, ttft_ms: Optional[float] = None):
        """
        Log standardized cost/performance telemetry.
        Pricing: $5/1M input, $15/1M output (gpt-4o default).
        """
        cost_estimate = (prompt_tokens / 1_000_000) * 5.0 + (completion_tokens / 1_000_000) * 15.0

        details = {
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "cost_estimate_usd": cost_estimate,
            "latency": {
                "end_to_end_ms": duration_ms
            }
        }
        if ttft_ms is not None:
            details["latency"]["ttft_ms"] = ttft_ms

        self.log_event("INFO", "TELEMETRY", "Agent execution metrics", **details)
