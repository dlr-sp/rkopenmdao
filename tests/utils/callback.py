"""Callback utiliies for `test_component_test_adaptive.py`"""

# pylint: disable=unnecessary-lambda

from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

from rkopenmdao.callback import Callback
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
)
from rkopenmdao.time_integration_state import TimeIntegrationState


@dataclass
class TimeStepsLog(Callback):
    """
    Callback for saving and printing step sizes taken for each
    step of time integration.
    """

    q: list = field(default_factory=lambda: [])

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        step_size = time_integration_state.step_size_history[0]
        print(f"Step size: {step_size}")
        self.q.append(step_size)


def save_data(timestepslog: TimeStepsLog, write_file: str):
    """
    Utility to save the data created by ``TimeStepsLog`` by generating a file
    """
    path = Path(write_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(write_file, "a", encoding="utf-8") as file:
        for step_size in timestepslog.q:
            file.write(f"{step_size}\n")


def read_data(read_file: str) -> list[float]:
    """
    Utility to read the data created by ``TimeStepsLog`` by generating a file
    """
    assert Path(read_file).exists()

    with open(read_file, "r", encoding="utf-8") as f:
        return [float(line.split()[0]) for line in f]
