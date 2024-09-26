from rkopenmdao.error_controller import ErrorController


def Integral(p, tol=1e-3, safety_factor=0.95, name="I-Controller"):
    alpha = 1/(1+p)
    return ErrorController(alpha, tol=tol, safety_factor=safety_factor, name=name)


def H0_110(p, tol=1e-3, safety_factor=0.95, name="H0_110"):
    return Integral(p, tol=tol, safety_factor=safety_factor, name=name)


def H_211(p, tol=1e-3, safety_factor=0.95, name="H_211"):
    alpha = 1/(4*p)
    beta = -1/(4*p)
    a = -1/4
    return ErrorController(alpha, beta=beta, a=a, tol=tol, safety_factor=safety_factor, name=name)


def H0_211(p, tol=1e-3, safety_factor=0.95, name="H0_211"):
    alpha = 1/(2*p)
    beta = -1/(2*p)
    a = -1/2
    return ErrorController(alpha, beta=beta, a=a, tol=tol, safety_factor=safety_factor, name=name)


def PC(p, tol=1e-3, safety_factor=0.95, name="PC"):
    alpha = 2/p
    beta = 1/p
    a = 1
    return ErrorController(alpha, beta=beta, a=a, tol=tol, safety_factor=safety_factor, name=name)


def H0_220(p, tol=1e-3, safety_factor=0.95, name="H0_220"):
    return PC(p, tol=tol, safety_factor=safety_factor, name=name)


def PID(p, tol=1e-3, safety_factor=0.95, name="PID"):
    alpha = 1/(18*p)
    beta = -1/(9*p)
    gamma = alpha
    return ErrorController(alpha, beta=beta, gamma=gamma, tol=tol, safety_factor=safety_factor, name=name)


def H_312(p, tol=1e-3, safety_factor=0.95, name="H_312"):
    alpha = 1/(8*p)
    beta = -1/(4*p)
    gamma = alpha
    a = -3/8
    b = -1/8
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def H0_312(p, tol=1e-3, safety_factor=0.95, name="H0_312"):
    alpha = 1/(4*p)
    beta = -1/(2*p)
    gamma = alpha
    a = -3/4
    b = -1/4
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def H_312_general(p, var_alpha, a, b, tol=1e-3, safety_factor=0.95, name="H_312_general"):
    alpha = var_alpha/p
    beta = -2*var_alpha/p
    gamma = alpha
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def PPID(p, tol=1e-3, safety_factor=0.95, name="PPID"):
    alpha = 6/(20*p)
    beta = -1/(20*p)
    gamma = -5/(20*p)
    a = 1
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, tol=tol, safety_factor=safety_factor, name=name)


def H_321(p, tol=1e-3, safety_factor=0.95, name="H_321"):
    alpha = 1/(3*p)
    beta = -1/(18*p)
    gamma = -5/(18*p)
    a = 5/6
    b = 1/6
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def H0_321(p, tol=1e-3, safety_factor=0.95, name="H0_321"):
    alpha = 1/(3*p)
    beta = -1/(2*p)
    gamma = -3/(4*p)
    a = 1/4
    b = 3/4
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def H_321_general(p, var_alpha, var_beta, a, tol=1e-3, safety_factor=0.95, name="H_321_general"):
    alpha = var_alpha/p
    beta = var_beta/p
    gamma = -(var_alpha+var_beta)/p
    b = 1 - a
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def H0_330(p, tol=1e-3, safety_factor=0.95, name="H0_330"):
    alpha = 3/p
    beta = 3/p
    gamma = 1/p
    a = 2
    b = -1
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)


def H_330_general(p, var_alpha, var_beta, var_gamma, tol=1e-3, safety_factor=0.95, name="H_330_general"):
    alpha = var_alpha/p
    beta = var_beta/p
    gamma = var_gamma/p
    a = 2
    b = -1
    return ErrorController(alpha, beta=beta, gamma=gamma, a=a, b=b, tol=tol, safety_factor=safety_factor, name=name)
