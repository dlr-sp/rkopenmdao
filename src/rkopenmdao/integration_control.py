# pylint: disable=missing-module-docstring

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TerminationCriterion:
    """
    Termination Criterion for Runge-Kutta integrator. Can either be by number of time steps or by end time.
    Parameters
    ---------
    criterion: str
        Stopping criterion to use in Runge-Kutta integrator. Can either be 'num_steps', to set a number of fixed steps,
        or 'end_time', to set the end time of the Runge-Kutta integrator.
    value: [int, float]
        Value of criterion to use in Runge-Kutta integrator; either number of time steps or end time.
    """
    criterion: str  # ['num_steps', 'end_time']
    value: [int, float]


@dataclass
class IntegrationControl:
    """
    Object for exchanging data between the Runge-Kutta integrator and the inner
    problems.
    """

    # Could solve this by putting things into subclasses, but I don't see the
    # benefit here.
    # pylint: disable=too-many-instance-attributes

    # General information
    initial_time: float
    termination_criterion: TerminationCriterion
    # We later probably want step size control, so then it would be step data
    initial_delta_t: float
    # Step data
    initial_step: int = 0
    _delta_t: float = field(init=False)
    smallest_delta_t: float = field(init=False)
    step: int = field(init=False)
    step_time: float = field(init=False)
    # Stage data
    stage: int = field(init=False)
    stage_time: float = field(init=False)
    butcher_diagonal_element: float = field(init=False)

    def __post_init__(self):
        self.step = self.initial_step
        self.step_time = self.initial_time
        self.smallest_delta_t = self.initial_delta_t
        self.delta_t = self.initial_delta_t
        self.delta_t_suggestion = self.initial_delta_t

        self.stage = 0
        self.stage_time = self.initial_time
        self.butcher_diagonal_element = 0.0

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, delta_t):
        self._delta_t = delta_t
        if self.smallest_delta_t > delta_t and delta_t > 0 :
            self.smallest_delta_t = delta_t

    def is_last_time_step(self, tol=1e-6):
        if self.termination_criterion.criterion == 'end_time':
            return self.step_time - self.termination_criterion.value <= tol
        else:
            return self.step == self.termination_criterion.value

    def remaining_time(self):
        if self.termination_criterion.criterion == 'end_time':
            return self.termination_criterion.value - self.step_time
        else:
            raise TypeError("Termination criteria must be end_time.")

    def termination_condition_status(self):
        if self.termination_criterion.criterion == 'num_steps':
            if self.step != self.termination_criterion.value:
                self.increment_step()
                return True
        elif self.termination_criterion.criterion == 'end_time':
            if (np.abs(self.remaining_time())
                    > min(1e-13, self.smallest_delta_t) and self.remaining_time() > 0.):
                self.increment_step()
                return True
        return False

    def reset(self):
        """
        Returns the instance to its initial state.
        """
        self.__post_init__()

    def increment_step(self):
        self.step += 1

    def decrement_step(self):
        self.step -= 1
