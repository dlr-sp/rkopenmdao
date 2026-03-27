"""
Object intended to allow the exchange of data between OpenMDAO components and
time integration.
"""

from dataclasses import dataclass


@dataclass
class OMDataExchange:
    """
    Object intended to allow the exchange of data between OpenMDAO components and
    time integration.

    Parameters
    ----------
    step_size: float
        Step size to use for current calculations.
    stage_factor: float
        Time discretization specific factor applied during calculation (e.g. a diagonal
        element of a butcher tableau).
    """

    step_size: float = 1.0e-3
    stage_factor: float = 1.0
