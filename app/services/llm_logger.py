
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = Path(
    os.getenv("LLM_LOG_FILE", str(PROJECT_ROOT / "llm_logs.jsonl"))
)

_file_lock = threading.Lock()


def log_llm_event(
    *,
    request_id: str,
    model: str,
    prompt_version: str,
    latency_seconds: float,
    success: bool,
    operation: str,
    error_type: str | None = None,
    retrieved_chunk_count: int | None = None,
    validation_errors: list[str] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Append one structured LLM event to a JSON Lines log."""

    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model": model,
        "prompt_version": prompt_version,
        "operation": operation,
        "latency_seconds": round(max(0.0, latency_seconds), 4),
        "success": success,
    }

    optional_fields = {
        "error_type": error_type,
        "retrieved_chunk_count": retrieved_chunk_count,
        "validation_errors": validation_errors,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    event.update({
        key: value
        for key, value in optional_fields.items()
        if value is not None
    })

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with _file_lock:
            with LOG_FILE.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(event, ensure_ascii=False) + "\n"
                )
    except OSError:
        # Logging failure should not expose sensitive data or
        # crash an otherwise successful API operation.
        logger.exception("Could not write the LLM event log")
