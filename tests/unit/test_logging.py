import json
import logging
import sys

from ai_pipeline.core.logging import (
    JsonFormatter,
    add_file_handler,
    configure_logging,
    get_logger,
    remove_handler,
)


def test_context_is_injected_and_bind_merges(capsys):
    configure_logging(verbosity=1)
    log = get_logger("test", project="demo").bind(operation="train.start", run_id=None)
    log.info("hello %s", "world")
    err = capsys.readouterr().err
    assert "hello world" in err
    assert "project=demo" in err and "operation=train.start" in err
    assert "run_id" not in err


def test_quiet_suppresses_info_and_json_lines_are_parseable(capsys):
    configure_logging(quiet=True, json_logs=True)
    log = get_logger("test", project="demo")
    log.info("hidden")
    log.error("shown", extra={"target": "ng4511-lab"})
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1
    rec = json.loads(err[0])
    assert rec["message"] == "shown" and rec["level"] == "ERROR"
    assert rec["project"] == "demo" and rec["target"] == "ng4511-lab"


def test_file_handler_receives_debug(tmp_path):
    configure_logging(verbosity=0)
    path = tmp_path / "logs" / "op.log"
    handler = add_file_handler(path)
    try:
        get_logger("test", operation="x").debug("fine detail")
    finally:
        remove_handler(handler)
    assert "fine detail" in path.read_text()
    assert handler not in logging.getLogger("ai_pipeline").handlers


def test_json_formatter_includes_exception():
    fmt = JsonFormatter()
    try:
        raise RuntimeError("bang")
    except RuntimeError:
        record = logging.LogRecord(
            "n", logging.ERROR, __file__, 1, "msg", (), exc_info=sys.exc_info()
        )
    out = json.loads(fmt.format(record))
    assert "RuntimeError: bang" in out["exception"]
