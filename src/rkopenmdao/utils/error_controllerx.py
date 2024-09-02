from error_controller import ErrorController


__all__ = [
    "Integral",
    "H0_110",
    "H_211",
    "H0_211",
    "PC",
    "H0_220",
    "PID",
    "H_312",
    "H0_312",
    "H_312_general",
    "PPID",
    "H_321",
    "H0_321",
    "H_321_general",
    "H0_330",
    "H_330_general",
]


def Integral(p, name="I-Controller"):
    alpha = 1/(1+p)
    return ErrorController(alpha, name=name)


def H0_110(p, name="H0_110"):
    return Integral(p, name=name)


def H_211(p, name="H_211"):
    alpha = 1/(4*p)
    beta = -1/(4*p)
    a = -1/4
    return ErrorController(alpha, beta=beta, a=a, name=name)


def H0_211(p, name="H0_211"):
    alpha = 1/(2*p)
    beta = -1/(2*p)
    a = -1/2
    return ErrorController(alpha, beta=beta, a=a, name=name)


def PC(p, name="PC"):
    alpha = 2/p
    beta = 1/p
    a = 1
    return ErrorController(alpha, beta=beta, a=a, name=name)


def H0_220(p, name="H0_220"):
    return PC(p, name=name)


def PID(p, name="PID"):
    alpha = 1/(18*p)
    beta = -1/(9*p)
    gamma = alpha
    return ErrorController(alpha, beta=beta, gamma=gamma, name=name)


def H_312(p, name="H_312"):
    alpha = 1/(8*p)
    beta = -1/(4*p)
    gamma = alpha
    a = -3/8
    b = -1/8
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def H0_312(p, name="H0_312"):
    alpha = 1/(4*p)
    beta = -1/(2*p)
    gamma = alpha
    a = -3/4
    b = -1/4
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def H_312_general(p, var_alpha, a, b, name="H_312_general"):
    alpha = var_alpha/p
    beta = -2*var_alpha/p
    gamma = alpha
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def PPID(p, name="PPID"):
    alpha = 6/(20*p)
    beta = -1/(20*p)
    gamma = -5/(20*p)
    a = 1
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, name=name)


def H_321(p, name="H_321"):
    alpha = 1/(3*p)
    beta = -1/(18*p)
    gamma = -5/(18*p)
    a = 5/6
    b = 1/6
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def H0_321(p, name="H0_321"):
    alpha = 1/(3*p)
    beta = -1/(2*p)
    gamma = -3/(4*p)
    a = 1/4
    b = 3/4
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def H_321_general(p, var_alpha, var_beta, a, name="H_321_general"):
    alpha = var_alpha/p
    beta = var_beta/p
    gamma = -(var_alpha+var_beta)/p
    b = 1 - a
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def H0_330(p, name="H0_330"):
    alpha = 3/p
    beta = 3/p
    gamma = 1/p
    a = 2
    b = -1
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)


def H_330_general(p, var_alpha, var_beta, var_gamma, name="H_330_general"):
    alpha = var_alpha/p
    beta = var_beta/p
    gamma = var_gamma/p
    a = 2
    b = -1
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, name=name)
