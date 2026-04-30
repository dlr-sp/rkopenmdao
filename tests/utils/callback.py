from dataclasses import dataclass

from rkopenmdao.callback import Callback
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
)
from rkopenmdao.time_integration_state import TimeIntegrationState


@dataclass
class TimeStepsLog(Callback):
    """
    Callback for saving and printing step sizes taken for each step of time integration.
    """

    write_file: str = ""

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        step_size = time_integration_state.step_size_history[0]
        print(f"Step size: {step_size}")

        # Save Step size in a file
        with open(self.write_file, "a") as file:
            file.write(f"{step_size}\n")
