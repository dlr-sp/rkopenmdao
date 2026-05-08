import numpy as np
import pytest


def assert_function_convergence_rate(
    error_function: callable,
    base_step_size: float,
    other_function_arguments: dict,
    expected_rate: float,
    num_calculations: int = 5,
):
    errors = np.zeros(num_calculations)
    for i in range(num_calculations):
        step_size = base_step_size * 0.5**i
        errors[i] = error_function(step_size=step_size, **other_function_arguments)
    if errors == pytest.approx(0.0, abs=1e-9):
        # Test is allowed to pass if machine accuracy is already achieved
        assert True
    else:
        rates = errors[:-1] / errors[1:]
        # Let the tests pass with if the expected order is reached
        #       - minus some tolerance
        #       - for the majority of runs
        assert np.sum(rates > expected_rate - 0.3) >= num_calculations // 2
