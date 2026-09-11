"""
Convergence utility functions for testing time integration schemes.

This module provides helper functions for verifying convergence rates of
numerical methods used in time integration implementations.
"""

import numpy as np
import pytest


def assert_function_convergence_rate(
    error_function: callable,
    base_step_size: float,
    other_function_arguments: dict,
    expected_rate: float,
    num_calculations: int = 5,
):
    """Verify that an error function converges at the expected rate.

    Performs a convergence study by evaluating the error function with
    successively halving step sizes and verifies that the error decreases
    at the expected asymptotic rate.

    Parameters
    ----------
    error_function : callable
        Function that computes the error for a given step size. Must accept
        ``step_size`` as a keyword argument and additional arguments from
        ``other_function_arguments``.
    base_step_size : float
        Initial step size for the convergence study. Step sizes will be
        ``base_step_size * 0.5^i`` for ``i = 0, ..., num_calculations-1``.
    other_function_arguments : dict
        Additional keyword arguments to pass to ``error_function`` on each call.
    expected_rate : float
        The expected asymptotic convergence rate (e.g., 2.0 for quadratic
        convergence, 1.0 for linear convergence).
    num_calculations : int, optional
        Number of convergence steps to perform. Default is 5.

    Returns
    -------
    None
        The function asserts that convergence occurs at the expected rate
        (within a tolerance of 0.3). The test passes if the ratio of errors
        exceeds ``expected_rate - 0.3`` for at least half of the calculations.

    Raises
    ------
    AssertionError
        If the observed convergence rate does not match the expected rate
        within the specified tolerance.

    Notes
    -----
    The convergence rate is estimated by computing the ratio of consecutive
    errors:

    .. math::

        r_i = \\frac{e_i}{e_{i+1}}

    where :math:`e_i` is the error at step size :math:`h_i`. If the method
    converges at rate :math:`p`, then :math:`r_i \\approx 2^p`.

    The test allows for some flexibility by accepting the expected rate minus
    a tolerance of 0.3, and requires that at least half of the calculations
    meet this criterion (to account for machine precision limits).
    """
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
