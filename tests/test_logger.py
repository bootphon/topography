"""Test of the logging utility."""
import logging
from tempfile import NamedTemporaryFile

import pytest

from topography.utils import get_logger


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
def test_logger(capsys, level):
    temp_file = NamedTemporaryFile()
    log = get_logger("test", temp_file.name, level=level)
    log.debug("DEBUG")
    log.info("INFO")
    log.warning("WARNING")
    log.error("ERROR")

    captured = capsys.readouterr()
    assert not captured.out
    if level is logging.ERROR:
        assert "ERROR" in captured.err
        assert "WARNING" not in captured.err
        assert "INFO" not in captured.err
        assert "DEBUG" not in captured.err
    if level is logging.WARNING:
        assert "ERROR" in captured.err
        assert "WARNING" in captured.err
        assert "INFO" not in captured.err
        assert "DEBUG" not in captured.err
    if level is logging.INFO:
        assert "ERROR" in captured.err
        assert "WARNING" in captured.err
        assert "INFO" in captured.err
        assert "DEBUG" not in captured.err
    if level is logging.DEBUG:
        assert "ERROR" in captured.err
        assert "WARNING" in captured.err
        assert "INFO" in captured.err
        assert "DEBUG" in captured.err


def test_logger_bad_level():
    temp_file = NamedTemporaryFile()
    with pytest.raises(ValueError) as err:
        get_logger("test", temp_file.name, level="bad")
    assert "Invalid logging level" in str(err.value)
