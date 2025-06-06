"""Collection of error controllers. Please refer to Diagonally Implicit Runge-Kutta
Methods for Ordinary DifferentialEquations. A Review by Kennedy, Christopher A. and
Mark H. Carpenter.
"""

# No need to write this into doc again.
# pylint: disable=missing-function-docstring
import numpy as np

from rkopenmdao.error_controller import ErrorController, ErrorControllerDecorator
from rkopenmdao.error_estimator import ErrorEstimator


def pseudo(
    p,
    error_estimator: ErrorEstimator,
    tol=np.inf,
    safety_factor=1.0,
    name="pseudo-Controller",
    lower_bound=0,
    upper_bound=np.inf,
    base: ErrorController = None,
    max_iter=1000,
):
    if base:
        return ErrorControllerDecorator(
            0,
            base,
            name=name,
        )
    return ErrorController(
        0,
        tol=tol,
        safety_factor=safety_factor,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def integral(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="I-Controller",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (1 + p)
    if base:
        return ErrorControllerDecorator(
            alpha,
            base,
            name=name,
        )
    return ErrorController(
        alpha,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h0_110(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H0_110",
    base: ErrorController = None,
    max_iter=5,
):

    return integral(
        p,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        base=base,
        max_iter=max_iter,
    )


def h_211(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H_211",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (4 * p)
    beta = -1 / (4 * p)
    a = -1 / 4
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            a=a,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        a=a,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h0_211(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H0_211",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (2 * p)
    beta = -1 / (2 * p)
    a = -1 / 2
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            a=a,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        a=a,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def pc(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="PC",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 2 / p
    beta = 1 / p
    a = 1
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            a=a,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        a=a,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h0_220(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H0_220",
    base: ErrorController = None,
    max_iter=5,
):
    return pc(
        p,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        base=base,
        max_iter=max_iter,
    )


def pid(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="PID",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (18 * p)
    beta = -1 / (9 * p)
    gamma = alpha
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h_312(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H_312",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (8 * p)
    beta = -1 / (4 * p)
    gamma = alpha
    a = -3 / 8
    b = -1 / 8
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h0_312(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H0_312",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (4 * p)
    beta = -1 / (2 * p)
    gamma = alpha
    a = -3 / 4
    b = -1 / 4
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h_312_general(
    p,
    var_alpha,
    a,
    b,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H_312_general",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = var_alpha / p
    beta = -2 * var_alpha / p
    gamma = alpha
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def ppid(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0,
    upper_bound=np.inf,
    name="PPID",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 6 / (20 * p)
    beta = -1 / (20 * p)
    gamma = -5 / (20 * p)
    a = 1
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h_321(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H_321",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (3 * p)
    beta = -1 / (18 * p)
    gamma = -5 / (18 * p)
    a = 5 / 6
    b = 1 / 6
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h0_321(
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H0_321",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 1 / (3 * p)
    beta = -1 / (2 * p)
    gamma = -3 / (4 * p)
    a = 1 / 4
    b = 3 / 4
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h_321_general(
    p,
    var_alpha,
    var_beta,
    a,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H_321_general",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = var_alpha / p
    beta = var_beta / p
    gamma = -(var_alpha + var_beta) / p
    b = 1 - a
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h0_330(  # delivers very small suggestions
    p,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H0_330",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = 3 / p
    beta = 3 / p
    gamma = 1 / p
    a = 2
    b = -1
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )


def h_330_general(
    p,
    var_alpha,
    var_beta,
    var_gamma,
    error_estimator: ErrorEstimator,
    tol=1e-6,
    safety_factor=0.95, 
    lower_bound=0, 
    upper_bound=np.inf,
    name="H_330_general",
    base: ErrorController = None,
    max_iter=5,
):
    alpha = var_alpha / p
    beta = var_beta / p
    gamma = var_gamma / p
    a = 2
    b = -1
    if base:
        return ErrorControllerDecorator(
            alpha,
            beta=beta,
            gamma=gamma,
            a=a,
            b=b,
            name=name,
            error_controller=base,
        )
    return ErrorController(
        alpha,
        beta=beta,
        gamma=gamma,
        a=a,
        b=b,
        tol=tol,
        safety_factor=safety_factor,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        name=name,
        error_estimator=error_estimator,
        max_iter=max_iter,
    )
