"""
Test module for the basic callback provided in this repo.
"""

from time import sleep
from typing import Generator

import pytest

from rkopenmdao.callback import IterationLogging, WallClockMeasurement


@pytest.mark.parametrize("function_name", ["foo", "bar", "compute"])
@pytest.mark.parametrize("iteration", [1, 100, 100000])
def test_iteration_logging(
    function_name: str,
    iteration: int,
    capsys: Generator[pytest.CaptureFixture[str], None, None],
):
    """
    Tests that the class `IterationLogging` prints out the correct messages.

    Parameters
    ----------
    function_name: str
        Name of the function that is logged.
    iteration: int
        Iteration which is tested.
    capsys: Generator[pytest.CaptureFixture[str], None, None]
        Pytest-fixture allowing the capture of stdout.
    """
    iter_logging = IterationLogging(function_name)
    iter_logging.before_iteration(iteration, None, None, None)
    captured = capsys.readouterr()
    assert captured.out == f"Starting step <{iteration}> of {function_name}.\n"
    iter_logging.after_iteration(iteration, None, None, None)
    captured = capsys.readouterr()
    assert captured.out == f"Finishing step <{iteration}> of {function_name}.\n"


@pytest.mark.parametrize("sleeptime", [0.1, 1.0, 2.0])
def test_wall_clock_measurement(
    sleeptime: float, capsys: Generator[pytest.CaptureFixture[str], None, None]
):
    """
    Tests that the class `WallClockMeasurement` prints out consistent messages.

    Parameters
    ----------
    sleeptime: float
        Time which is slept between `before`- and `after_iteration(...)` calls.
    capsys: Generator[pytest.CaptureFixture[str], None, None]
        Pytest-fixture allowing the capture of stdout.
    """
    wall_clock_measure = WallClockMeasurement()
    wall_clock_measure.before_iteration(None, None, None, None)
    sleep(sleeptime)
    wall_clock_measure.after_iteration(None, None, None, None)
    captured = capsys.readouterr()
    measured_time = float(captured.out[15:-10])
    assert measured_time >= sleeptime
