from dataclasses import dataclass

from rkopenmdao.error_estimator import ErrorEstimator


def _three_queue_controller(_list: list, val) -> list:
    if _list is None:
        _list = [val]
    elif len(_list) < 3:
        _list.append(val)
    else:
        _list.pop(0)
        _list.append(val)
    return _list


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
                                     tolerance: float = 1e-3,
                                     safety_factor: float = 0.95):

        delta_time_new = safety_factor * self.delta_time_steps[-1]
        if 0 in self.local_error_norms:
            return 1.0
        delta_time_new *= (tolerance / self.local_error_norms[-1]) ** self.alpha
        if len(self.local_error_norms) >= 2:
            delta_time_new *= (self.local_error_norms[-2] / tolerance) ** self.beta
            if len(self.local_error_norms) >= 3:
                delta_time_new *= (tolerance / self.local_error_norms[-3]) ** self.gamma
        if len(self.delta_time_steps) >= 2:
            delta_time_new *= (self.delta_time_steps[-1] / self.delta_time_steps[-2]) ** self.a
            if len(self.delta_time_steps) >= 3:
                delta_time_new *= (self.delta_time_steps[-2] / self.delta_time_steps[-3]) ** self.b
        return delta_time_new

    def __call__(self,
                 current_norm: float,
                 delta_t: float,
                 tol: float = 1e-3,
                 safety_factor: float = 0.95,
                 ):
        """ Estimates next possible step size for a given state and embedded solution
        and returns whether the next step size meets the tolerance.
        """
        self.local_error_norms = _three_queue_controller(self.local_error_norms, current_norm)
        self.delta_time_steps = _three_queue_controller(self.delta_time_steps, delta_t)
        delta_t_new = self._estimate_next_step_function(tol, safety_factor)
        return delta_t_new

    def __str__(self):
        _vars = vars(self)
        name = _vars.pop("name")
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
