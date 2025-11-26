"""Collection of error controllers. Please refer to Diagonally Implicit Runge-Kutta
Methods for Ordinary DifferentialEquations. A Review by Kennedy, Christopher A. and
Mark H. Carpenter.
"""

# No need to write this into doc again.
# pylint: disable=missing-function-docstring
import numpy as np

from rkopenmdao.error_controller import (
    ErrorController,
    ErrorControllerDecorator,
    ErrorControllerConfig,
)


def pseudo(
    p,  # pylint: disable=unused-argument
    config=ErrorControllerConfig(np.inf, 0, np.inf, 1.0, 1000),
    name="pseudo-Controller",
    base: ErrorController = None,
):
    if base:
        return ErrorControllerDecorator(
            0,
            base,
            name=name,
        )
    return ErrorController(0, config=config)


def integral(
    p,
    config=ErrorControllerConfig(),
    name="I-Controller",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h0_110(
    p,
    config=ErrorControllerConfig(),
    name="H0_110",
    base: ErrorController = None,
):
    return integral(
        p,
        config=config,
        name=name,
        base=base,
    )


def h_211(
    p,
    config=ErrorControllerConfig(),
    name="H_211",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h0_211(
    p,
    config=ErrorControllerConfig(),
    name="H0_211",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def pc(
    p,
    config=ErrorControllerConfig(),
    name="PC",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h0_220(
    p,
    config=ErrorControllerConfig(),
    name="H0_220",
    base: ErrorController = None,
):
    return pc(
        p,
        config=config,
        name=name,
        base=base,
    )


def pid(
    p,
    config=ErrorControllerConfig(),
    name="PID",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h_312(
    p,
    config=ErrorControllerConfig(),
    name="H_312",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h0_312(
    p,
    config=ErrorControllerConfig(),
    name="H0_312",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h_312_general(
    p,
    var_alpha,
    a,
    b,
    config=ErrorControllerConfig(),
    name="H_312_general",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def ppid(
    p,
    config=ErrorControllerConfig(),
    name="PPID",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h_321(
    p,
    config=ErrorControllerConfig(),
    name="H_321",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h0_321(
    p,
    config=ErrorControllerConfig(),
    name="H0_321",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h_321_general(
    p,
    var_alpha,
    var_beta,
    a,
    config=ErrorControllerConfig(),
    name="H_321_general",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h0_330(  # delivers very small suggestions
    p,
    config=ErrorControllerConfig(),
    name="H0_330",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )


def h_330_general(
    p,
    var_alpha,
    var_beta,
    var_gamma,
    config=ErrorControllerConfig(),
    name="H_330_general",
    base: ErrorController = None,
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
        config=config,
        name=name,
    )
