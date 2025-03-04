"""Contains the class for the error controller of the adaptive Runge-Kutta schemes """

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from rkopenmdao.error_estimator import ErrorEstimator
from rkopenmdao.errors import OuterErrorControllerError, InnerErrorControllerError


@dataclass
class LocalData:
    """Object containing 2 prior steps information, namely the time steps and norms,
    of the current step."""

    _tol: float  # Tolerance is required for initialization of the local error norm.
    _local_error_norms: list = field(init=False, repr=False)  # of [step-1, step-2]
    _delta_time_steps: list = field(init=False, repr=False)  # of [step-1, step-2]

    def __post_init__(self):
        self.local_error_norms: list = 2 * [self._tol]
        self.delta_time_steps: list = 2 * [None]

    @property
    def local_error_norms(self):
        """
        Returns
        ------
            List of the error norms of the 2 last time steps.
        """
        return self._local_error_norms

    @local_error_norms.setter
    def local_error_norms(self, local_error_norms):
        """
        Initializes the local error norms attribute with tolerance values if None,
        else sets the local error norms.
        """
        self._local_error_norms = local_error_norms

    @property
    def delta_time_steps(self):
        """
        Returns
        ------
            List of the time difference between the 2 last time steps
        """
        return self._delta_time_steps

    @delta_time_steps.setter
    def delta_time_steps(self, delta_time_steps):
        self._delta_time_steps = delta_time_steps

    def push_to_delta_time_steps(self, delta_t):
        """Pushes to the new time step to the list and dumps the old one"""
        self._delta_time_steps[:0] = [delta_t]
        self._delta_time_steps.pop()

    def push_to_local_error_norms(self, norm):
        """Pushes to the norm to the list and dumps the old one"""
        if norm == 0:
            self._local_error_norms[:0] = [self._local_error_norms[0]]
        else:
            self._local_error_norms[:0] = [norm]
        self._local_error_norms.pop()

    def reset(self):
        """Resets the data."""
        self.local_error_norms: list = 2 * [self._tol]
        self.delta_time_steps: list = 2 * [None]


class ErrorController:
    r"""
    Error controller that estimates the next time-difference jumps of a Runge-Kutta
    scheme in accordance to a specified tolerance and uses the following equation:

    (\Delta t)^{[n+1]} = \kappa (\Delta t)^{[n]}
    \left \{\frac{\varepsilon}{\left\Vert \delta^{[n+1]} \right\Vert} \right\}^\alpha
    \left \{\frac{\left\Vert  \delta ^{[n]} \right\Vert}{\varepsilon}\right\} ^\beta
    \left\{\frac{\varepsilon}{\left\Vert \delta^{[n-1]} \right\Vert}\right\}^\gamma
    \left\{\frac{(\Delta t)^{[n]}}{(\Delta t)^{[n-1]}}\right\}^a
    \left\{\frac{(\Delta t)^{[n-1]}}{(\Delta t)^{[n-2]}}\right\}^b

    Attributes
    ----------
    alpha : float
        The exponent constant to the tolerance by the current norm.
    beta : float, optional
        The exponent constant to the current norm by the last norm.
    gamma : float, optional
        The exponent constant to the last norm by the prior norm.
    a : float, optional
        The exponent constant to the current time-difference by the last.
        time-difference
    b : float, optional
        The exponent constant to the last time-difference by the prior one.
    tol : float, optional
        The tolerance of the error controller.
    safety_factor: float, optional
        Safety factor in the equation smaller than 1.
    name: str, optional
        Name of the error controller
    error_estimator: ErrorEstimator object, optional.
        The error estimator class
    _local_error_norms: List[float]
        List of the error norms of the 2 last time steps.
    _delta_time_steps: List[float]
        List of the time difference between the 2 last time steps.

    Methods
    -------
    __call__(solution, embedded_solution, delta_t)
        Estimates next possible step size for a given state and embedded solution
        and returns whether the next step size meets the tolerance.
    __str__()
        Prints the class data.
    """

    def __init__(
        self,
        alpha,
        error_estimator: ErrorEstimator = None,
        beta: float = 0,
        gamma: float = 0,
        a: float = 0,
        b: float = 0,
        tol: float = 1e-6,
        safety_factor: float = 0.95,
        name: str = "ErrorController",
    ):
        # Constant parameters for the error controller equation
        # -----------
        # 1. General parameters
        self.safety_factor = safety_factor
        self.tol = tol
        # 2. Exponents for norm sensitivities
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # 3. Exponents for delta_t Sensitivities
        self.a = a
        self.b = b
        # -----------
        self.local_data = LocalData(self.tol)  # Local step History object
        self.error_estimator = error_estimator  # Error estimator for the Norm
        self.inner_most = True  # Indicator if it is not associated
        self.name = "Outer " + name
        self.width = 0

    def __call__(
        self,
        solution: np.ndarray,
        embedded_solution: np.ndarray,
        delta_t: float,
    ) -> Tuple[float, bool]:
        """
        A call function which returns a delta_t suggestion and
        an acceptance status.
        """
        suggestion, accepted = self._run(solution, embedded_solution, delta_t)
        if not accepted:
            if suggestion > delta_t and self.inner_most:
                raise OuterErrorControllerError(
                    f"Suggested delta T {suggestion} is larger than delta t {delta_t} on failure"
                )
        return suggestion, accepted

    def _run(
        self,
        solution: np.ndarray,
        embedded_solution: np.ndarray,
        delta_t: float,
    ) -> Tuple[float, bool]:
        """
        Estimates next possible step size for a given state and embedded solution
        and returns whether the next step size meets the tolerance.

        Parameters
        ----------
        solution : np.ndarray
            The solution of the current time step.
        embedded_solution : np.ndarray
            The embedded solution of the current time step.
        delta_t:
            current time step size.

        Returns
        -------
        Tuple(float,bool)
            1. A suggestion to next time step size and
            2. True if for current step size the norm is in tolerance, otherwise False.
        """
        current_norm = self.error_estimator(solution, embedded_solution)
        if current_norm <= self.tol:
            if current_norm != 0:
                delta_t = self._estimate_next_step_function(current_norm, delta_t)
            self.local_data.push_to_local_error_norms(current_norm)
            self.local_data.push_to_delta_time_steps(delta_t)
            return delta_t, True
        delta_t = self._estimate_next_step_function(current_norm, delta_t)
        return delta_t, False

    def _estimate_next_step_function(self, current_norm, current_dtime):
        """
        Estimates the delta time of the current or next time step.
        """
        delta_time_new = self.safety_factor * current_dtime
        delta_time_new *= (self.tol / current_norm) ** self.alpha
        delta_time_new *= (self.local_data.local_error_norms[0] / self.tol) ** self.beta
        delta_time_new *= (
            self.tol / self.local_data.local_error_norms[1]
        ) ** self.gamma
        if self.local_data.delta_time_steps[0] is not None:
            delta_time_new *= (
                current_dtime / self.local_data.delta_time_steps[0]
            ) ** self.a
            if self.local_data.delta_time_steps[1] is not None:
                delta_time_new *= (
                    self.local_data.delta_time_steps[0]
                    / self.local_data.delta_time_steps[1]
                ) ** self.b
        return delta_time_new

    def __str__(self):
        """
        Prints the class data as following:
            ---------------------------
            |     ErrorController     |
            ---------------------------
            | alpha: 1.0              |
            | beta: 2.0               |
            | gamma: 3.0              |
            | a: 4.0                  |
            | b: 5.0                  |
            | tol: 0.001              |
            | safety_factor: 0.95     |
            | error_estimator: L_None |
            ---------------------------
        """
        _vars = vars(self).copy()
        name = _vars.pop("name")
        _vars.pop("local_data")
        _str_list = []
        for i, var in enumerate(_vars):
            if _vars[var] is not None and _vars[var] != 0:
                if var == "error_controller":
                    continue
                _str_list.append(f"{var}: {_vars[var]}")
        width = len(max(_str_list, key=len)) + 4
        self.width = width
        title = name.center(width - 2)
        _str = "-" * width + "\n" + f"|{title}" + "|\n" + "-" * width + "\n"
        for i in _str_list:
            _str += "| " + i + " " * (width - 3 - len(i)) + "|\n"
        _str += "-" * width
        return _str

    def reset(self):
        """Resets the error controller to the initial state."""
        self.local_data.reset()


@dataclass
class ErrorControllerDecorator(ErrorController):
    """Error Controller Decorator that wraps the base error controller with
    a different error controller in case of failure."""

    def __init__(
        self,
        alpha,
        error_controller: ErrorController,
        error_estimator: ErrorEstimator = None,
        beta: float = 0,
        gamma: float = 0,
        a: float = 0,
        b: float = 0,
        tol: float = 1e-6,
        safety_factor: float = 0.95,
        name: str = "ErrorController",
    ):
        self.error_controller = error_controller
        self.error_controller.inner_most = False
        self.error_controller.name = self.error_controller.name.replace(
            "Inner ", " ", 1
        )
        super().__init__(
            alpha,
            error_estimator,
            beta,
            gamma,
            a,
            b,
            tol,
            safety_factor,
            name,
        )
        self.is_not_inner = True
        self.name = "Inner " + name
        self.local_data = self.error_controller.local_data
        self.outer_counter = 0

    def __call__(
        self,
        solution: np.ndarray,
        embedded_solution: np.ndarray,
        delta_t: float,
    ) -> Tuple[float, bool]:
        """
        A call function which returns a delta_t suggestion and
        an acceptance status
        """
        if self.is_not_inner:
            try:
                suggestion, accepted = self.error_controller(
                    solution, embedded_solution, delta_t
                )
                if not accepted:
                    self.outer_counter += 1
                    if not (suggestion <= delta_t and self.outer_counter <= 3):
                        raise OuterErrorControllerError(
                            f"""Suggested delta T {suggestion} is larger than 
                            delta t {delta_t} on failure"""
                        )
                self.outer_counter = 0
                return suggestion, accepted
            except OuterErrorControllerError:
                return self._run(solution, embedded_solution, delta_t)
        else:
            return self._run(solution, embedded_solution, delta_t)

    def __str__(self):
        """Prints the class data"""
        current_str = super().__str__()
        return f"{self.error_controller.__str__()}\n{' '*int(self.width/2)}+\n{current_str}"

    def _run(
        self,
        solution: np.ndarray,
        embedded_solution: np.ndarray,
        delta_t: float,
    ) -> Tuple[float, bool]:
        """
        Estimates next possible step size for a given state and embedded solution
        and returns whether the next step size meets the tolerance.

        Parameters
        ----------
        solution : np.ndarray
            The solution of the current time step.
        embedded_solution : np.ndarray
            The embedded solution of the current time step.
        delta_t:
            current time step size.

        Returns
        -------
        Tuple(float,bool)
            1. A suggestion to next time step size and
            2. True if for current step size the norm is in tolerance, otherwise False.
        """

        self.is_not_inner = True
        suggestion, accepted = super()._run(solution, embedded_solution, delta_t)
        if not accepted:
            self.is_not_inner = False
            if suggestion > delta_t and self.inner_most:
                raise InnerErrorControllerError(
                    f"Suggested delta T {suggestion} is larger than delta t {delta_t} on failure"
                )
        return suggestion, accepted
