"""Test of the logging utility."""
import logging
from pathlib import Path

import pandas as pd
import pytest

from topography.training import Writer
from topography.utils import get_logger, tensorboard_to_dataframe


@pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
def test_logger(tmp_path, capsys, level):
    log = get_logger("test", tmp_path / f"{level}.log", level=level)
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


def test_logger_bad_level(tmp_path):
    with pytest.raises(ValueError) as err:
        get_logger("test", tmp_path / "bad.log", level="bad")
    assert "Invalid logging level" in str(err.value)


def test_tensorboard_to_dataframe(tmp_path):
    writer = Writer(tmp_path / "writer")
    writer.next_epoch("test")
    writer["loss"].update(1, 3)
    writer["loss"].update(0, 2)
    writer["acc"].update(0.5, 1)

    summary = writer.summary()
    assert summary == "test, epoch 1, loss 0.600, acc 0.500"

    path = list(Path(tmp_path).rglob("*.tfevents*"))
    assert len(path) == 1
    dataframe = tensorboard_to_dataframe(path[0])
    assert len(dataframe) == 2

    ref_dataframe = pd.DataFrame(
        {
            "metric": ["test/loss", "test/acc"],
            "value": [0.6, 0.5],
            "step": [1.0, 1.0],
        }
    )
    decimals = 6
    assert dataframe.round(decimals).equals(ref_dataframe)
