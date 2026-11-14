"""Structured (JSON) logging.

One JSON object per line is easy to read for humans with ``jq`` and easy to ship to a log
system later. Extra context goes in the ``extra`` argument:

    log.info("stage finished", extra={"stage": "ingest", "rows": 7043})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else was passed through ``extra``.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format a log record as one line of JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Return the record as a JSON string (with the traceback if there is one)."""
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


QUIET_LOGGERS = ("httpx", "httpcore")


def setup_logging(level: str | None = None) -> None:
    """Send all logs to stderr as JSON. Safe to call more than once.

    Args:
        level: "DEBUG", "INFO", ...; defaults to the LOG_LEVEL environment variable or INFO.
    """
    chosen = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(chosen)
    for handler in list(root.handlers):
        if getattr(handler, "_churnsense", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler._churnsense = True  # type: ignore[attr-defined]  # marker so we do not add twice
    root.addHandler(handler)
    for noisy in QUIET_LOGGERS:  # one line per HTTP call from the client libraries is noise
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger; call ``setup_logging`` once at program start."""
    return logging.getLogger(name)
