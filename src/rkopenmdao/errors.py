"""Error classes for the Runge-Kutta part of the code."""


class RungeKuttaError(Exception):
    """Base class for exceptions in the Runge-Kutta part of the code"""


class SetupError(RungeKuttaError, ValueError):
    """Exception for the case when something goes wrong in the setup of the RK-integrator"""


class PostprocessingError(RungeKuttaError, AssertionError):
    """Exception for the case when something goes wrong in the postprocessing of the RK-integrator"""


class TimeStageError(RungeKuttaError, AssertionError):
    """Exception for the case when something goes wrong in the time stage computation of the RK-integrator"""
