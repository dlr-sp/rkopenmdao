from typing import Tuple
from dataclasses import dataclass, field
from rkopenmdao.error_estimator import *
import numpy as np


@dataclass
class ErrorController:
    r"""
    Error controller that estimates the next time-difference jumps of a Runge-Kutta scheme in accordance to a specified
    tolerance and uses the following equation:

    (\Delta t)^{[n+1]} = \kappa (\Delta t)^{[n]}
    \left \{\frac{\varepsilon}{\left\Vert \delta^{[n+1]} \right\Vert} \right\}^\alpha
    \left \{\frac{\left\Vert  \delta ^{[n]} \right\Vert}{\varepsilon}\right\} ^\beta
    \left\{\frac{\varepsilon}{\left\Vert \delta^{[n-1]} \right\Vert}\right\}^\gamma
    \left\{\frac{(\Delta t)^{[n]}}{(\Delta t)^{[n-1]}}\right\}^a
    \left\{\frac{(\Delta t)^{[n-1]}}{(\Delta t)^{[n-2]}}\right\}^b

    Attributes
    ----------
    alpha : float
        The exponent constant to the tolerance by the current norm
    beta : float, optional
        The exponent constant to the current norm by the last norm
    gamma : float, optional
        The exponent constant to the last norm by the prior norm
    a : float, optional
        The exponent constant to the current time-difference by the last time-difference
    b : float, optional
        The exponent constant to the last time-difference by the prior one
    tol : float, optional
        The tolerance of the error controller
    safety_factor: float, optional
        Safety factor in the equation smaller than 1
    name: str, optional
        Name of the error controller
    error_estimator: ErrorEstimator object, optional
        The error estimator

    Methods
    -------
    __call__(solution, embedded_solution, delta_t)
        Estimates next possible step size for a given state and embedded solution
        and returns whether the next step size meets the tolerance.
    __str__()
        Prints the class data
    """
    alpha: float
    error_estimator: ErrorEstimator = None
    beta: float = 0
    gamma: float = 0
    a: float = 0
    b: float = 0
    tol: float = 1e-6
    safety_factor: float = 0.95
    name: str = "ErrorController"
    _local_error_norms: list = field(init=False, repr=False)
    _delta_time_steps: list = field(init=False, repr=False)

    def __post_init__(self):
        self.local_error_norms = None
        self.delta_time_steps = None

    def reset(self):
        self.__post_init__()

    @property
    def local_error_norms(self):
        return self._local_error_norms

    @local_error_norms.setter
    def local_error_norms(self, local_error_norms):
        if local_error_norms is None:
            self._local_error_norms = 2*[self.tol]
        else:
            self._local_error_norms = local_error_norms

    @property
    def delta_time_steps(self):
        return self._delta_time_steps

    @delta_time_steps.setter
    def delta_time_steps(self, delta_time_steps):
        if delta_time_steps is None:
            self._delta_time_steps = 2 * [None]
        else:
            self._delta_time_steps = delta_time_steps

    def _estimate_next_step_function(self, current_norm, current_time):
        """
        Solves the equation
        """
        if 0 in self.local_error_norms:
            return current_time
        delta_time_new = self.safety_factor * current_time
        delta_time_new *= (self.tol / current_norm) ** self.alpha
        delta_time_new *= (self.local_error_norms[0] / self.tol) ** self.beta
        delta_time_new *= (self.tol / self.local_error_norms[1]) ** self.gamma
        if self._delta_time_steps[0] is not None:
            delta_time_new *= (current_time / self.delta_time_steps[0]) ** self.a
            if self._delta_time_steps[1] is not None:
                delta_time_new *= (self.delta_time_steps[0] / self.delta_time_steps[1]) ** self.b
        return delta_time_new

    def __call__(self,
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
            2. True if for current step size the norm is in tolerance, otherwise False
        """
        current_norm = self.error_estimator(solution, embedded_solution)
        print(current_norm)
        delta_t_new = self._estimate_next_step_function(current_norm,  delta_t)
        print(delta_t_new)
        print(self.delta_time_steps)
        if current_norm == 0:
            self.delta_time_steps = [delta_t, self._delta_time_steps[0]]
            return delta_t, True
        elif current_norm <= self.tol:
            self.local_error_norms = [current_norm, self._local_error_norms[0]]
            self.delta_time_steps = [delta_t, self._delta_time_steps[0]]
            return delta_t_new, True
        return delta_t_new, False

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
        _vars = vars(self)
        name = _vars.pop("name")
        _vars.pop("local_error_norms")
        _vars.pop("delta_time_steps")
        _str_list = []
        for var in _vars:
            if _vars[var] is not None and _vars[var] != 0:
                _str_list.append(f"{var}: {_vars[var]}")
        width = len(max(_str_list, key=len)) + 4
        title = name.center(width - 2)
        _str = ("-" * width + "\n" +
                f"|{title}" + "|\n" + "-" * width + "\n")
        for i in _str_list:
            _str += "| " + i + " " * (width - 3 - len(i)) + "|\n"
        _str += "-" * width
        return _str
