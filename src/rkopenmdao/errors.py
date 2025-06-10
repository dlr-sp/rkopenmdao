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


class ErrorControllerError(Exception):
    """Base class for exceptions in the Error-Controller part of the code"""


class OuterErrorControllerError(ErrorControllerError, Exception):
    """Exception for the case that the outer ErrorController suggests
    larger delta T on failure.
    """


class InnerErrorControllerError(ErrorControllerError, Exception):
    """Exception for the case that the inner ErrorController suggests
    larger delta T on failure.
    """
