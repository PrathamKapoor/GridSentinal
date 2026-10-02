"""Structured logging foundation.

Phase 1 logging answers, for every record:

* what happened  -> ``message`` / ``event``
* when          -> UTC ISO-8601 ``timestamp``
* which component -> ``logger`` (module path under ``energy_intelligence``)
* which experiment -> ``experiment``
* which configuration -> ``config_id`` (see :meth:`AppConfig.fingerprint`)

Two sinks are configured:

* **console** - human-readable, for interactive development.
* **file** - JSON Lines, one object per line, for later ingestion and for
  correlating a run with the artifacts it produced.

Only the ``energy_intelligence`` logger is configured. The root logger is left
untouched so that importing this package never changes a host application's
logging behaviour.

Example:
    >>> from energy_intelligence.config import load_config
    >>> from energy_intelligence.logging_setup import initialize_logging, get_logger
    >>> config = load_config()                      # doctest: +SKIP
    >>> initialize_logging(config)                  # doctest: +SKIP
    >>> get_logger(__name__).info("pipeline stage", extra={"event": "stage_start"})
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import AppConfig

__all__ = [
    "ROOT_LOGGER_NAME",
    "ConsoleFormatter",
    "JsonLinesFormatter",
    "RunContextFilter",
    "get_logger",
    "initialize_logging",
    "reset_logging",
    "shutdown_logging",
]

ROOT_LOGGER_NAME = "energy_intelligence"

#: Marks handlers this module owns. The project logger can legitimately carry
#: handlers attached by a host (pytest's caplog, a Jupyter kernel, an embedding
#: application). Counting those as "already initialised" would make
#: initialize_logging() silently do nothing, so ownership is tracked explicitly.
_MANAGED_ATTR = "_energy_intelligence_managed"

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
_ISO_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Process-wide run identity, populated by initialize_logging().
_RUN_ID: str | None = None
_EXPERIMENT: str | None = None
_CONFIG_ID: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunContextFilter(logging.Filter):
    """Attach run identity to every record.

    Using a filter (rather than a formatter) means the context is available to
    handlers, to sibling filters and to application code that inspects records,
    regardless of which sink finally renders them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID or "unset"
        record.experiment = _EXPERIMENT or "unset"
        record.config_id = _CONFIG_ID or "unset"
        return True


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for interactive use."""

    def __init__(self) -> None:
        super().__init__(fmt=_CONSOLE_FORMAT, datefmt=_ISO_FORMAT)

    def formatTime(  # noqa: N802 - logging API name
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
            timespec="milliseconds"
        )


class JsonLinesFormatter(logging.Formatter):
    """One JSON object per line, suitable for machine consumption."""

    #: Attributes present on every :class:`logging.LogRecord`. Anything else set
    #: by application code is treated as an extra structured field.
    _RESERVED = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message", "module",
            "msecs", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _utc_now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "experiment": getattr(record, "experiment", None),
            "config_id": getattr(record, "config_id", None),
            "run_id": getattr(record, "run_id", None),
        }

        for key, value in record.__dict__.items():
            if key in self._RESERVED or key in payload or key.startswith("_"):
                continue
            payload[key] = value if _is_json_safe(value) else repr(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=False)


def _is_json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the project namespace.

    Args:
        name: Usually ``__name__``. A fully-qualified ``energy_intelligence.*``
            name is returned as-is; anything else is nested beneath the package
            so third-party or stdlib loggers are never captured by our handlers.
    """
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def _managed(handler: logging.Handler) -> logging.Handler:
    """Tag a handler as owned by this module."""
    setattr(handler, _MANAGED_ATTR, True)
    return handler


def _managed_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, _MANAGED_ATTR, False)]


def _detach_managed_handlers(logger: logging.Logger) -> None:
    """Remove only our own handlers, leaving host handlers untouched."""
    for handler in _managed_handlers(logger):
        logger.removeHandler(handler)
        handler.close()


def initialize_logging(
    config: AppConfig,
    *,
    console: bool = True,
    to_file: bool = True,
    force: bool = False,
) -> Path | None:
    """Configure structured logging for the project.

    Idempotent: calling it again without ``force=True`` returns the existing log
    file path without adding duplicate handlers, so importing and calling this
    from several entry points cannot double-log.

    Args:
        config: Validated application config. Supplies log level, experiment
            name and the log directory.
        console: Attach a human-readable stdout handler.
        to_file: Attach a JSON Lines file handler.
        force: Tear down and rebuild handlers even if already configured.

    Returns:
        Absolute path of the log file, or ``None`` if file logging is disabled or
        the log directory could not be created.
    """
    global _RUN_ID, _EXPERIMENT, _CONFIG_ID

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    existing = _managed_handlers(logger)
    if existing and not force:
        return _active_log_file()

    if existing:
        _detach_managed_handlers(logger)

    _RUN_ID = uuid.uuid4().hex[:8]
    _EXPERIMENT = config.experiment_name
    _CONFIG_ID = config.fingerprint()

    logger.setLevel(getattr(logging, config.log_level))
    logger.propagate = False

    context_filter = RunContextFilter()
    logger.addFilter(context_filter)

    if console:
        stream_handler = _managed(logging.StreamHandler(stream=sys.stdout))
        stream_handler.setFormatter(ConsoleFormatter())
        stream_handler.addFilter(context_filter)
        logger.addHandler(stream_handler)

    log_file: Path | None = None
    if to_file:
        log_file = config.log_dir / f"{config.experiment_name}.jsonl"
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = _managed(
                logging.FileHandler(log_file, mode="a", encoding="utf-8")
            )
            file_handler.setFormatter(JsonLinesFormatter())
            file_handler.addFilter(context_filter)
            logger.addHandler(file_handler)
        except OSError as exc:
            log_file = None
            logger.warning("File logging disabled: could not open %s (%s)", config.log_dir, exc)

    # INFO, not DEBUG: log files are opened in append mode, so a visible
    # run-start marker is what lets a reader tell where one run ends and the
    # next begins in a long-lived log file.
    logger.info(
        "logging initialised",
        extra={"event": "logging_initialised", "log_file": str(log_file) if log_file else None},
    )
    return log_file


def _active_log_file() -> Path | None:
    for handler in _managed_handlers(logging.getLogger(ROOT_LOGGER_NAME)):
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def reset_logging() -> None:
    """Remove this module's handlers and clear run identity.

    Handlers attached by a host application are deliberately left alone: this
    module owns only what it installed. Used by tests to guarantee isolation
    between cases.
    """
    global _RUN_ID, _EXPERIMENT, _CONFIG_ID

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    _detach_managed_handlers(logger)
    logger.filters.clear()
    _RUN_ID = None
    _EXPERIMENT = None
    _CONFIG_ID = None


def shutdown_logging() -> None:
    """Flush and detach handlers. Call before process exit."""
    _detach_managed_handlers(logging.getLogger(ROOT_LOGGER_NAME))
    logging.shutdown()
