# pylint: disable=missing-module-docstring

from dataclasses import dataclass


@dataclass
class IntegrationControl:
    """
    Object for exchanging data between the Runge-Kutta integrator and the inner problems.
    """

    # Could solve this by putting things into subclasses, but I don't see the benefit here.
    # pylint: disable=too-many-instance-attributes

    # General information
    initial_time: float
    num_steps: int

    # We later probably want step size control, so then it would be step data
    delta_t: float

    # Step data
    step: int = 0
    step_time_old: float = 0.0
    step_time_new: float = 0.0

    # Stage data
    stage: int = 0
    stage_time: float = 0.0
    butcher_diagonal_element: float = 0.0

    def reset(self):
        """
        Returns the instance to its initial state.
        """
        self.step = 0
        self.step_time_old = self.initial_time
        self.step_time_new = self.initial_time

        self.stage = 0
        self.stage_time = self.initial_time
        self.butcher_diagonal_element = 0.0
