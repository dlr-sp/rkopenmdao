"""Callback utiliies for `test_component_test_adaptive.py`"""

# pylint: disable=unnecessary-lambda

from dataclasses import dataclass,field
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

    q:deque  = field(default_factory=lambda: deque())

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

@dataclass
class TimeStepsLogToFile(Callback):
    """
    Callback for saving into a file and printing step sizes taken for
    each step of time integration.
    """

    write_file: str = ""

    def __post_init__(self):
        path = Path(self.write_file)
        path.parent.mkdir(parents=True, exist_ok=True)

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        step_size = time_integration_state.step_size_history[0]
        print(f"Step size: {step_size}")
        with open(self.write_file, "a", encoding='utf-8') as file:
            file.write(f"{step_size}\n")
