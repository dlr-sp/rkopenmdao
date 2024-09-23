"""Error classes for the Runge-Kutta part of the code."""


class RungeKuttaError(Exception):
    """Base class for exceptions in the Runge-Kutta part of the code"""


class SetupError(RungeKuttaError, ValueError):
    """Exception for the case when something goes wrong in the setup of the
    RungeKuttaIntegrator"""


class PostprocessingError(RungeKuttaError, AssertionError):
    """Exception for the case when something goes wrong in the postprocessing of the
    RungeKuttaIntegrator"""


class TimeStageError(RungeKuttaError, AssertionError):
    """Exception for the case when something goes wrong in the time stage computation
    of the RungeKuttaIntegrator"""
