import logging

from graspgenx.utils.logging_config import setup_logging, get_logger


def test_setup_logging_initializes():
    setup_logging()
    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO


def test_setup_logging_idempotent():
    setup_logging()
    handler_count_1 = len(logging.getLogger().handlers)
    setup_logging()
    handler_count_2 = len(logging.getLogger().handlers)
    assert handler_count_1 == handler_count_2


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_get_logger_different_names():
    logger_a = get_logger("module_a")
    logger_b = get_logger("module_b")
    assert logger_a.name != logger_b.name


def test_logger_can_log(caplog):
    logger = get_logger("test_log_output")
    with caplog.at_level(logging.INFO, logger="test_log_output"):
        logger.info("test message")
    assert "test message" in caplog.text
