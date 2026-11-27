"""Structured service events without copying customer input into logs."""
from datetime import datetime, timezone
import json
import logging
import sys


class PredictionJsonFormatter(logging.Formatter):
    """Serialize only operational fields, including a UTC event timestamp."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "event": record.getMessage(),
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
        }
        for key in ("status", "status_code", "duration_ms", "error_type"):
            if hasattr(record, key):
                event[key] = getattr(record, key)
        return json.dumps(event, allow_nan=False)


def prediction_logger() -> logging.Logger:
    """Configure one stdout handler without modifying Uvicorn/root logging."""
    logger = logging.getLogger("mp299.predictions")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(PredictionJsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
