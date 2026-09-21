"""
Callback utilities for logging the step sizes of adaptive time integrations
and reading the corresponding reference data.
"""

# pylint: disable=unnecessary-lambda

from pathlib import Path

from rkopenmdao.callback import TimeStepsLog


def save_data(timesteps_log: TimeStepsLog, write_file: str):
    """
    Utility to save the data created by ``TimeStepsLog`` to a file
    """
    path = Path(write_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(write_file, "a", encoding="utf-8") as file:
        for step_size in timesteps_log.time_steps:
            file.write(f"{step_size}\n")


def read_data(read_file: str) -> list[float]:
    """
    Utility to read the data created by ``TimeStepsLog`` from a file
    """
    assert Path(read_file).exists()

    with open(read_file, "r", encoding="utf-8") as f:
        return [float(line.split()[0]) for line in f]
