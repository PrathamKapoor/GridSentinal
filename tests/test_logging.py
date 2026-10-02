"""Structured logging tests.

Covers the Phase 1 requirement that logs record what happened, when, which
component, which experiment and which configuration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from energy_intelligence.logging_setup import (
    ROOT_LOGGER_NAME,
    ConsoleFormatter,
    JsonLinesFormatter,
    get_logger,
    initialize_logging,
    reset_logging,
)


def _records(log_file: Path) -> list[dict]:
    lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def _managed_handlers() -> list[logging.Handler]:
    """Handlers this project installed, ignoring any a host (e.g. pytest) added."""
    return [
        handler
        for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers
        if getattr(handler, "_energy_intelligence_managed", False)
    ]


def test_logger_is_namespaced_under_the_project() -> None:
    assert get_logger("energy_intelligence.health").name == "energy_intelligence.health"
    assert get_logger(__name__).name.startswith(f"{ROOT_LOGGER_NAME}.")
    assert get_logger("some_third_party").name == "energy_intelligence.some_third_party"
    assert get_logger().name == ROOT_LOGGER_NAME


def test_initialize_writes_json_lines(make_config, tmp_paths) -> None:
    config = make_config(experiment_name="log-test")
    log_file = initialize_logging(config, console=False, to_file=True)
    assert log_file is not None
    assert log_file.parent == config.log_dir
    assert log_file.name == "log-test.jsonl"

    get_logger(__name__).info("hello", extra={"event": "greeting"})
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    records = _records(log_file)
    assert len(records) == 2  # run-start marker + greeting

    marker, greeting = records
    assert marker["event"] == "logging_initialised"
    assert marker["level"] == "INFO"
    assert marker["log_file"] == str(log_file)

    assert greeting["message"] == "hello"
    assert greeting["level"] == "INFO"
    assert greeting["event"] == "greeting"
    assert greeting["logger"] == get_logger(__name__).name


def test_every_record_carries_run_context(make_config) -> None:
    config = make_config(experiment_name="ctx-test")
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    get_logger(__name__).warning("with context")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    record = _records(log_file)[-1]
    assert record["experiment"] == "ctx-test"
    assert record["config_id"] == config.fingerprint()
    assert record["run_id"] not in (None, "", "unset")


def test_timestamp_is_iso8601_utc(make_config) -> None:
    config = make_config()
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    get_logger(__name__).info("timestamped")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    from datetime import datetime

    timestamp = _records(log_file)[-1]["timestamp"]
    assert timestamp.endswith("+00:00")
    assert datetime.fromisoformat(timestamp).tzinfo is not None


def test_level_is_applied_from_config(make_config) -> None:
    config = make_config(log_level="ERROR")
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    get_logger(__name__).info("should be filtered")
    get_logger(__name__).error("should appear")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    messages = [record["message"] for record in _records(log_file)]
    assert "should be filtered" not in messages
    assert "should appear" in messages


def test_extra_fields_become_top_level_keys(make_config) -> None:
    config = make_config()
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    get_logger(__name__).info(
        "structured", extra={"event": "forecast", "horizon_hours": 24, "regime": "peak"}
    )
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    record = _records(log_file)[-1]
    assert record["event"] == "forecast"
    assert record["horizon_hours"] == 24
    assert record["regime"] == "peak"


def test_non_serialisable_extra_is_stringified(make_config) -> None:
    config = make_config()
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    get_logger(__name__).info("odd", extra={"payload": object()})
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    record = _records(log_file)[-1]
    assert isinstance(record["payload"], str)


def test_exceptions_are_serialised(make_config) -> None:
    config = make_config()
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger(__name__).error("failed", exc_info=True)
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    record = _records(log_file)[-1]
    assert "ValueError: boom" in record["exception"]


def test_initialize_is_idempotent(make_config) -> None:
    config = make_config()
    first = initialize_logging(config, console=False)
    handler_count = len(_managed_handlers())

    second = initialize_logging(config, console=False)
    assert second == first
    assert len(_managed_handlers()) == handler_count


def test_host_handlers_do_not_disable_initialisation(make_config) -> None:
    """A handler attached by a host must not make initialize_logging a no-op.

    Regression guard: the idempotency check once counted *any* handler, so a
    single pre-existing handler (pytest's caplog, a Jupyter kernel) silently
    suppressed all logging setup.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    host_handler = logging.NullHandler()
    logger.addHandler(host_handler)
    try:
        log_file = initialize_logging(make_config(), console=False)
        assert log_file is not None
        assert log_file.exists()
        assert len(_managed_handlers()) == 1
    finally:
        logger.removeHandler(host_handler)


def test_reset_logging_leaves_host_handlers_alone(make_config) -> None:
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    host_handler = logging.NullHandler()
    logger.addHandler(host_handler)
    try:
        initialize_logging(make_config(), console=False)
        reset_logging()
        assert _managed_handlers() == []
        assert host_handler in logger.handlers
    finally:
        logger.removeHandler(host_handler)


def test_force_rebuilds_handlers(make_config) -> None:
    config = make_config()
    initialize_logging(config, console=False)
    initialize_logging(config, console=False, force=True)
    assert len(_managed_handlers()) == 1


def test_file_logging_can_be_disabled(make_config) -> None:
    config = make_config()
    assert initialize_logging(config, console=False, to_file=False) is None
    assert _managed_handlers() == []


def test_console_handler_writes_to_stdout(make_config, capsys) -> None:
    config = make_config(log_level="INFO")
    initialize_logging(config, console=True, to_file=False)

    get_logger("energy_intelligence.test").info("console message")
    captured = capsys.readouterr()
    assert "console message" in captured.out
    assert "energy_intelligence.test" in captured.out


def test_unwritable_log_dir_degrades_instead_of_crashing(make_config, tmp_path) -> None:
    """A bad log path must not take the run down; it must report clearly."""
    blocker = tmp_path / "logs-blocker"
    blocker.write_text("i am a file, not a directory", encoding="utf-8")

    config = make_config(log_dir=str(blocker / "logs"))
    log_file = initialize_logging(config, console=False, to_file=True)

    assert log_file is None
    handler = logging.getLogger(ROOT_LOGGER_NAME).handlers[0]
    assert isinstance(handler, logging.StreamHandler)


def test_project_logger_does_not_propagate_to_root(make_config) -> None:
    initialize_logging(make_config(), console=False, to_file=False)
    assert logging.getLogger(ROOT_LOGGER_NAME).propagate is False


def test_reset_logging_clears_handlers_and_identity(make_config) -> None:
    from energy_intelligence.logging_setup import RunContextFilter

    config = make_config()
    log_file = initialize_logging(config, console=False)
    assert log_file is not None

    get_logger(__name__).info("before reset")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()
    assert _records(log_file)[-1]["experiment"] == config.experiment_name

    reset_logging()
    assert _managed_handlers() == []

    # After a reset the filter falls back to its placeholder, proving the run
    # identity was cleared. (The formatter deliberately reports None when the
    # filter has not run, so it does not depend on filter execution order.)
    orphan = logging.LogRecord("x", logging.INFO, __file__, 1, "orphan", None, None)
    assert RunContextFilter().filter(orphan) is True
    assert orphan.experiment == "unset"
    assert orphan.config_id == "unset"
    assert orphan.run_id == "unset"


def test_formatters_render_without_a_handler(make_config) -> None:
    """Formatters must not depend on the filter having run."""
    record = logging.LogRecord("energy_intelligence.x", logging.INFO, __file__, 1, "raw", None, None)
    assert "raw" in ConsoleFormatter().format(record)
    assert json.loads(JsonLinesFormatter().format(record))["message"] == "raw"


@pytest.mark.parametrize("name", [None, "", ROOT_LOGGER_NAME])
def test_get_logger_handles_empty_names(name) -> None:
    assert get_logger(name).name == ROOT_LOGGER_NAME
