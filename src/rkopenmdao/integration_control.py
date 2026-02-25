# pylint: disable=missing-module-docstring
from abc import ABC, abstractmethod


class IntegrationControl(ABC):
    """
    Interface for exchanging data between the Runge-Kutta integrator and the inner
    problems.
    """

    # Could solve this by putting things into subclasses, but I don't see the
    # benefit here.
    # pylint: disable=too-many-instance-attributes
    def __init__(self, delta_t: float, initial_time: float = 0.0):
        # Time Data
        self.initial_time = initial_time
        self.initial_delta_t = delta_t
        self.smallest_delta_t = self.initial_delta_t
        self._delta_t = self.initial_delta_t
        self.delta_t_suggestion = self.initial_delta_t
        # Step data
        self.initial_step = 0
        self.step = self.initial_step
        self.step_time = self.initial_time
        # Stage data
        self.stage = 0
        self.stage_time = self.initial_time
        self.butcher_diagonal_element = 0.0

    @property
    def delta_t(self):
        """
        Returns
        ------
        float
            Time difference of the current time step.
        """
        return self._delta_t

    @delta_t.setter
    def delta_t(self, delta_t):
        self._delta_t = delta_t
        self.update_smallest_delta_t()

    def increment_step(self):
        """Increments step"""
        self.step += 1

    def decrement_step(self):
        """Decrements step"""
        self.step -= 1

    def update_smallest_delta_t(self):
        """Update the smallest delta_t in the run"""
        self.smallest_delta_t = min(self.smallest_delta_t, self.delta_t)

    @abstractmethod
    def is_last_time_step(self):
        """
        Returns
        ------
        bool
            True if last step.
        """

    @abstractmethod
    def remaining_time(self, current_time: float):
        """
        Returns
        ------
        float
            Remaining time.
        """

    def termination_condition_status(self, termination_step=0) -> bool:
        """
        A termination condition parameter for a while loop, that terminates when
        last time step is reached.
        Returns
        ------
        bool
            False if reached
        """
        if termination_step > 0:
            if self.step != termination_step:
                self.increment_step()
                return True
        elif not self.is_last_time_step():
            self.increment_step()
            return True
        return False

    def reset(self):
        """
        Returns the instance to its initial state.
        """
        self.step = self.initial_step
        self.step_time = self.initial_time
        self.smallest_delta_t = self.initial_delta_t
        self.delta_t = self.initial_delta_t
        self.delta_t_suggestion = self.initial_delta_t

        self.stage = 0
        self.stage_time = self.initial_time
        self.butcher_diagonal_element = 0.0


class TimeTerminationIntegrationControl(IntegrationControl):
    """
    Object for exchanging data between the Runge-Kutta integrator
    and the inner problems and sets a time objective as a criterion
    for the termination.
    """

    def __init__(
        self,
        delta_t: float,
        end_time: float,
        initial_time: float = 0.0,
        tol: float = 1e-7,
    ):
        self.end_time = end_time
        self.tol = tol
        super().__init__(delta_t, initial_time)

    # @property
    # def delta_t(self):
    #     """
    #     Returns
    #     ------
    #     float
    #         Time difference of the current time step.
    #     """
    #     return self._delta_t

    # @delta_t.setter
    # def delta_t(self, delta_t):
    #     self._delta_t = min(delta_t, self.remaining_time(self.step_time))
    #     self.update_smallest_delta_t()

    def is_last_time_step(self) -> bool:
        """
        Returns
        ------
        bool
            True if last step.
        """
        return self.remaining_time(self.step_time) <= self.tol

    def remaining_time(self, current_time: float) -> float:
        """
        Returns
        ------
        float
            Remaining time.
        """
        return self.end_time - current_time


class StepTerminationIntegrationControl(IntegrationControl):
    """
    Object for exchanging data between the Runge-Kutta integrator
    and the inner problems and sets a step objective as a criterion
    for the termination.
    """

    def __init__(self, delta_t: float, num_steps: int, initial_time: float = 0.0):
        super().__init__(delta_t, initial_time)
        self.num_steps = num_steps

    def is_last_time_step(self) -> bool:
        """
        Returns
        ------
        bool
            True if last step.
        """
        return self.step == self.num_steps

    def remaining_time(self, current_time) -> float:
        """
        Returns
        ------
        float
            Remaining time.
        """
        return self.num_steps * self.delta_t - current_time
