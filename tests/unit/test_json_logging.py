import json
import logging
import sys

from rag_engine.observability.logging import JsonFormatter, configure_logging, request_id_var


def make_record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("rag_engine.test", logging.INFO, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formats_one_json_object_per_record() -> None:
    line = JsonFormatter().format(make_record("hello"))

    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "rag_engine.test"
    assert payload["timestamp"].endswith("+00:00")
    assert "\n" not in line


def test_includes_extra_fields() -> None:
    payload = json.loads(JsonFormatter().format(make_record(http_status=200, duration_ms=1.5)))

    assert payload["http_status"] == 200
    assert payload["duration_ms"] == 1.5


def test_drops_uvicorns_ansi_colored_message_copy() -> None:
    payload = json.loads(JsonFormatter().format(make_record(color_message="\x1b[36mhi\x1b[0m")))

    assert "color_message" not in payload


def test_includes_the_request_id_only_inside_a_request() -> None:
    assert "request_id" not in json.loads(JsonFormatter().format(make_record()))

    token = request_id_var.set("abc123")
    try:
        payload = json.loads(JsonFormatter().format(make_record()))
    finally:
        request_id_var.reset(token)

    assert payload["request_id"] == "abc123"


def test_configure_logging_is_idempotent_and_quiets_noisy_loggers() -> None:
    root = logging.getLogger()
    original_handlers, original_level = list(root.handlers), root.level
    httpx_level = logging.getLogger("httpx").level
    try:
        configure_logging("INFO")
        configure_logging("INFO")

        json_handlers = [h for h in root.handlers if isinstance(h.formatter, JsonFormatter)]
        assert len(json_handlers) == 1
        assert not logging.getLogger("uvicorn.access").hasHandlers()
        assert logging.getLogger("uvicorn.error").propagate
        assert logging.getLogger("httpx").level == logging.WARNING
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_level)
        logging.getLogger("httpx").setLevel(httpx_level)


def test_includes_exception_details() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = make_record("failed")
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]
