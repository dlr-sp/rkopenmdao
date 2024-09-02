from dataclasses import dataclass

import numpy as np

from rkopenmdao.runge_kutta_scheme import EmbeddedRungeKuttaScheme


def _three_queue_controller(_list: list, val) -> list:
    if _list is None:
        _list = [val]
    elif len(_list) < 3:
        _list.append(val)
    else:
        _list.pop(0)
        _list.append(val)


@dataclass
class ErrorController:
    alpha: float
    beta: float = 0
    gamma: float = 0
    a: float = 0
    b: float = 0
    name: str = "ErrorController"
    local_error_norms: list = None
    delta_time_steps: list = None

    def _estimate_next_step_function(self,
                                     local_error_norms: list,
                                     delta_time_steps: list,
                                     tolerance: float = 1e-7,
                                     safety_factor: float = 0.95):
        delta_time_new = safety_factor * delta_time_steps[-1]
        delta_time_new *= (tolerance/local_error_norms[-1]) ** self.alpha
        if len(local_error_norms) >= 2:
            delta_time_new *= (local_error_norms[-2]/tolerance) ** self.beta
            if len(local_error_norms) >= 3:
                delta_time_new *= (tolerance/local_error_norms[-3]) ** self.gamma
        if len(delta_time_new) >= 2:
            delta_time_new *= (delta_time_steps[-1] / delta_time_steps[-2]) ** self.a
            if len(delta_time_new) >= 3:
                delta_time_new *= (delta_time_steps[-2] / delta_time_steps[-3]) ** self.b
        return delta_time_new

    def estimate_next_step_size(self,
                                new_state_solution,
                                new_state_embedded_solution,
                                delta_t: float,
                                tol: float = 1e-3,
                                safety_factor: float = 0.95,
                                ):
        """ Estimates next possible step size for a given state and embedded solution
        and returns whether the next step size meets the tolerance.
        """
        current_norm = np.linalg.norm(new_state_solution-new_state_embedded_solution, 2)
        _three_queue_controller(self.local_error_norms, current_norm)
        _three_queue_controller(self.delta_time_steps, delta_t)
        delta_t_new = self._estimate_next_step_function(self.local_error_norms, self.delta_time_steps,
                                                        tol, safety_factor)
        if current_norm > tol:
            return False, delta_t_new
        return True, delta_t_new
