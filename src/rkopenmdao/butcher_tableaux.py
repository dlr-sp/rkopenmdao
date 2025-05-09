"""
A collection of butcher tableaux. Most of them are given by
Kennedy, Christopher A. and Mark H. Carpenter. “Diagonally
Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review.” (2016).
Exceptions are either from well known methods (like explicit/implicit Euler),
or their source is given at the tableau.

Acronyms:
DIRK = Diagonally Implicit Runge-Kutta
Prefixes:
    S = Singly
    E = Explicit first stage
    Q = Quasi
"""

# All functions in here just create the corresponding butcher tableau.
# No need to write this into doc again.
# pylint: disable=missing-function-docstring

import numpy as np

from rkopenmdao.butcher_tableau import ButcherTableau, EmbeddedButcherTableau

__all__ = [
    "explicit_euler",
    "implicit_euler",
    "implicit_midpoint",
    "third_order_two_stage_sdirk",
    "third_order_three_stage_sdirk",
    "third_order_three_stage_esdirk",
    "runge_kutta_four",
    "third_order_four_stage_sdirk",
    "third_order_second_weak_stage_order_four_stage_dirk",
    "third_order_third_weak_stage_order_four_stage_dirk",
    "fourth_order_five_stage_sdirk",
    "fifth_order_five_stage_sdirk",
    "fourth_order_six_stage_esdirk",
    "fourth_order_third_weak_stage_order_six_stage_dirk",
    "fifth_order_six_stage_esdirk",
    "embedded_heun_euler",
    "embedded_second_order_two_stage_sdirk",
    "embedded_second_order_three_stage_esdirk",
    "embedded_third_order_three_stage_sdirk",
    "embedded_third_order_four_stage_esdirk",
    "embedded_third_order_five_stage_esdirk",
    "embedded_fourth_order_five_stage_esdirk",
    "embedded_runge_kutta_fehlberg",
    "embedded_fifth_order_six_stage_esdirk",
]

# one stage methods
explicit_euler = ButcherTableau(
    np.array([[0.0]]), np.array([1.0]), np.array([0.0]), p=1, name="Explicit Euler"
)

implicit_euler = ButcherTableau(
    np.array([[1.0]]), np.array([1.0]), np.array([1.0]), p=1, name="Implicit Euler"
)

implicit_midpoint = ButcherTableau(
    np.array([[0.5]]), np.array([1.0]), np.array([0.5]), p=2, name="Implicit Midpoint"
)


# two stage methods


def create_third_order_two_stage_sdirk():
    gamma = 0.5 + np.sqrt(3.0) / 6
    c2 = 0.5 - np.sqrt(3.0) / 6
    tableau = ButcherTableau(
        np.array([[gamma, 0.0], [c2 - gamma, gamma]]),
        np.array([0.5, 0.5]),
        np.array([gamma, c2]),
        p=3,
        name="SDIRK 2-stage, 3rd order",
    )
    return tableau


third_order_two_stage_sdirk = create_third_order_two_stage_sdirk()


# three stage methods


def create_third_order_three_stage_esdirk():
    gamma = (3.0 + np.sqrt(3.0)) / 6.0
    c3 = 2 * gamma - np.sqrt((2 + np.sqrt(3.0)) / 3)
    a32 = c3 * (c3 - 2 * gamma) / (4 * gamma)
    b2 = (-2.0 + 3.0 * c3) / (12.0 * (c3 - 2 * gamma) * gamma)
    b3 = (1 - 3.0 * gamma) / (3 * c3 * (c3 - 2 * gamma))
    tableau = ButcherTableau(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [gamma, gamma, 0.0],
                [c3 - a32 - gamma, a32, gamma],
            ]
        ),
        np.array([1 - b2 - b3, b2, b3]),
        np.array([0.0, 2 * gamma, c3]),
        p=2,
        name="ESDIRK 3-stage, 3rd order",
    )
    return tableau


third_order_three_stage_esdirk = create_third_order_three_stage_esdirk()


def create_third_order_three_stage_sdirk(): #TODO: Embedded 
    gamma = 0.43586652150845899941601945
    alpha = 1 - 4 * gamma + 2 * gamma**2
    beta = -1 + 6 * gamma - 9 * gamma**2 + 3 * gamma**3
    b2 = -3 * alpha**2 / (4 * beta)
    c2 = (2 - 9 * gamma + 6 * gamma**2) / (3 * alpha)
    tableau = ButcherTableau(
        np.array(
            [
                [gamma, 0.0, 0.0],
                [c2 - gamma, gamma, 0.0],
                [1 - b2 - gamma, b2, gamma],
            ]
        ),
        np.array([1 - b2 - gamma, b2, gamma]),
        np.array([gamma, c2, 1]),
        p=3,
        name="SDIRK 3-stage, 3rd order",
    )
    return tableau


third_order_three_stage_sdirk = create_third_order_three_stage_sdirk()


# four stage methods
def create_third_order_four_stage_sdirk():
    gamma = 9 / 40
    c2 = 7 / 13
    c3 = 11 / 15
    a32 = -(
        (c2 - c3) * (c3 - gamma) * (-1 + 9 * gamma - 18 * gamma**2 + 6 * gamma**3)
    ) / (
        (c2 - gamma)
        * (-2 + 3 * c2 + 9 * gamma - 12 * c2 * gamma - 6 * gamma**2 + 6 * c2 * gamma**2)
    )
    b2 = -(
        -2 + 3 * c3 + 9 * gamma - 12 * c3 * gamma - 6 * gamma**2 + 6 * c3 * gamma**2
    ) / (6 * (c2 - c3) * (c2 - gamma))
    b3 = (
        -2 + 3 * c2 + 9 * gamma - 12 * c2 * gamma - 6 * gamma**2 + 6 * c2 * gamma**2
    ) / (6 * (c2 - c3) * (c3 - gamma))

    tableau = ButcherTableau(
        np.array(
            [
                [gamma, 0.0, 0.0, 0.0],
                [c2 - gamma, gamma, 0.0, 0.0],
                [c3 - a32 - gamma, a32, gamma, 0.0],
                [1 - b2 - b3 - gamma, b2, b3, gamma],
            ]
        ),
        np.array([1 - b2 - b3 - gamma, b2, b3, gamma]),
        np.array([gamma, c2, c3, 1]),
        p=3,
        name="SDIRK 4-stage, 3rd order",
    )
    return tableau


third_order_four_stage_sdirk = create_third_order_four_stage_sdirk()

# see Ketcheson, David I. et al. “DIRK Schemes with High Weak Stage Order.”
# Lecture Notes in Computational Science and Engineering (2018): p. 5.
# https://arxiv.org/pdf/1811.01285.pdf
third_order_second_weak_stage_order_four_stage_dirk = ButcherTableau(
    np.array(
        [
            [0.01900072890, 0.0, 0.0, 0.0],
            [0.40434605601, 0.38435717512, 0.0, 0.0],
            [0.06487908412, -0.16389640295, 0.51545231222, 0.0],
            [0.02343549374, -0.41207877888, 0.96661161281, 0.42203167233],
        ]
    ),
    np.array([0.02343549374, -0.41207877888, 0.96661161281, 0.42203167233]),
    np.array([0.01900072890, 0.78870323114, 0.41643499339, 1.0]),
    p=3,
    name="2nd weak stage order 4-stage DIRK, 3rd order",
)

# see Ketcheson, David I. et al. “DIRK Schemes with High Weak Stage Order.”
# Lecture Notes in Computational Science and Engineering (2018): p. 6.
# https://arxiv.org/pdf/1811.01285.pdf
third_order_third_weak_stage_order_four_stage_dirk = ButcherTableau(
    np.array(
        [
            [0.13756543551, 0.0, 0.0, 0.0],
            [0.56695122794, 0.23483888782, 0.0, 0.0],
            [-1.08354072813, 2.96618223864, 0.44915521951, 0.0],
            [0.59761291500, -0.43420997584, -0.05305815322, 0.88965521406],
        ]
    ),
    np.array([0.59761291500, -0.43420997584, -0.05305815322, 0.88965521406]),
    np.array([0.13756543551, 0.80179011576, 2.33179673002, 1.0]),
    p=3,
    name="3rd weak stage order 4-stage DIRK, 3rd order",
)

runge_kutta_four = ButcherTableau(
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ),
    np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
    np.array([0.0, 0.5, 0.5, 1.0]),
    p=4,
    name="Runge-Kutta 4",
)

# five stage methods:


fourth_order_five_stage_sdirk = ButcherTableau(
    np.array(
        [
            [0.25, 0.0, 0.0, 0.0, 0.0],
            [(1 - 2**0.5) * 0.25, 0.25, 0.0, 0.0, 0.0],
            [
                (-1676 + 145 * 2**0.5) / 6724,
                3 * (709 + 389 * 2**0.5) / 6724,
                0.25,
                0.0,
                0.0,
            ],
            [
                (-371435 - 351111 * 2**0.5) / 470596,
                (98054928 + 73894543 * 2**0.5) / 112001848,
                (56061972 + 30241643 * 2**0.5) / 112001848,
                0.25,
                0.0,
            ],
            [
                0.0,
                4 * (74 + 273 * 2**0.5) / 5253,
                (19187 + 5031 * 2**0.5) / 55284,
                (116092 - 100113 * 2**0.5) / 334956,
                0.25,
            ],
        ]
    ),
    np.array(
        [
            0.0,
            4 * (74 + 273 * 2**0.5) / 5253,
            (19187 + 5031 * 2**0.5) / 55284,
            (116092 - 100113 * 2**0.5) / 334956,
            0.25,
        ]
    ),
    np.array(
        [
            0.25,
            (2 - 2**0.5) / 4,
            (13 + 8 * 2**0.5) / 41,
            (41 + 9 * 2**0.5) / 49,
            1.0,
        ]
    ),
    p=4,
    name="SDIRK 5-stage, 4th order",
)

fifth_order_five_stage_sdirk = ButcherTableau(
    np.array(
        [
            [4024571134387 / 14474071345096, 0.0, 0.0, 0.0, 0.0],
            [
                9365021263232 / 12572342979331,
                4024571134387 / 14474071345096,
                0.0,
                0.0,
                0.0,
            ],
            [
                2144716224527 / 9320917548702,
                -397905335951 / 4008788611757,
                4024571134387 / 14474071345096,
                0.0,
                0.0,
            ],
            [
                -291541413000 / 6267936762551,
                226761949132 / 4473940808273,
                -1282248297070 / 9697416712681,
                4024571134387 / 14474071345096,
                0.0,
            ],
            [
                -2481679516057 / 4626464057815,
                -197112422687 / 6604378783090,
                3952887910906 / 9713059315593,
                4906835613583 / 8134926921134,
                4024571134387 / 14474071345096,
            ],
        ]
    ),
    np.array(
        [
            -2522702558582 / 12162329469185,
            1018267903655 / 12907234417901,
            4542392826351 / 13702606430957,
            5001116467727 / 12224457745473,
            1509636094297 / 3891594770934,
        ]
    ),
    np.array(
        [
            4024571134387 / 14474071345096,
            5555633399575 / 5431021154178,
            5255299487392 / 12852514622453,
            3 / 20,
            10449500210709 / 14474071345096,
        ]
    ),
    p=5,
    name="SDIRK 5-stage, 5th order",
)

# six stage methods:

fourth_order_six_stage_esdirk = ButcherTableau(
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            [(1 - 2**0.5) / 8, (1 - 2**0.5) / 8, 0.25, 0.0, 0.0, 0.0],
            [
                (5 - 7 * 2**0.5) / 64,
                (5 - 7 * 2**0.5) / 64,
                7 * (1 + 2**0.5) / 32,
                0.25,
                0.0,
                0.0,
            ],
            [
                (-13796 - 54539 * 2**0.5) / 125000,
                (-13796 - 54539 * 2**0.5) / 125000,
                (506605 + 132109 * 2**0.5) / 437500,
                166 * (-97 + 376 * 2**0.5) / 109375,
                0.25,
                0.0,
            ],
            [
                (1181 - 987 * 2**0.5) / 13782,
                (1181 - 987 * 2**0.5) / 13782,
                47 * (-267 + 1783 * 2**0.5) / 273343,
                -16 * (-22922 + 3525 * 2**0.5) / 571953,
                -15625 * (97 + 376 * 2**0.5) / 90749876,
                0.25,
            ],
        ]
    ),
    np.array(
        [
            (1181 - 987 * 2**0.5) / 13782,
            (1181 - 987 * 2**0.5) / 13782,
            47 * (-267 + 1783 * 2**0.5) / 273343,
            -16 * (-22922 + 3525 * 2**0.5) / 571953,
            -15625 * (97 + 376 * 2**0.5) / 90749876,
            0.25,
        ]
    ),
    np.array([0.0, 0.5, (2 - 2**0.5) / 4, 5 / 8, 26 / 25, 1.0]),
    p=4,
)

fifth_order_six_stage_esdirk = ButcherTableau(
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                3282482714977 / 11805205429139,
                3282482714977 / 11805205429139,
                0,
                0,
                0,
                0,
            ],
            [
                606638434273 / 1934588254988,
                2719561380667 / 6223645057524,
                3282482714977 / 11805205429139,
                0,
                0,
                0,
            ],
            [
                -651839358321 / 6893317340882,
                -1510159624805 / 11312503783159,
                235043282255 / 4700683032009,
                3282482714977 / 11805205429139,
                0,
                0,
            ],
            [
                -5266892529762 / 23715740857879,
                -1007523679375 / 10375683364751,
                521543607658 / 16698046240053,
                514935039541 / 7366641897523,
                3282482714977 / 11805205429139,
                0,
            ],
            [
                -6225479754948 / 6925873918471,
                6894665360202 / 11185215031699,
                -2508324082331 / 20512393166649,
                -7289596211309 / 4653106810017,
                39811658682819 / 14781729060964,
                3282482714977 / 11805205429139,
            ],
        ]
    ),
    np.array(
        [
            -6225479754948 / 6925873918471,
            6894665360202 / 11185215031699,
            -2508324082331 / 20512393166649,
            -7289596211309 / 4653106810017,
            39811658682819 / 14781729060964,
            3282482714977 / 11805205429139,
        ],
    ),
    np.array(
        [
            0.0,
            4024571134387 / 7237035672548,
            14228244952610 / 13832614967709,
            0.1,
            3 / 50,
            1.0,
        ]
    ),
    p=5,
    name="ESDIRK 6-stage, 5th order",
)

# see Ketcheson, David I. et al. “DIRK Schemes with High Weak Stage Order.”
# Lecture Notes in Computational Science and Engineering (2018): p. 6.
# https://arxiv.org/pdf/1811.01285.pdf
fourth_order_third_weak_stage_order_six_stage_dirk = ButcherTableau(
    np.array(
        [
            [
                0.079672377876931,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.328355391763968,
                0.136009256546967,
                0,
                0,
                0,
                0,
            ],
            [
                -0.650772774016417,
                1.742859063495349,
                0.256472952467792,
                0,
                0,
                0,
            ],
            [
                -0.714580550967259,
                1.793745752775934,
                -0.078254785672497,
                0.311753794172585,
                0,
                0,
            ],
            [
                -1.120092779092918,
                1.983452339867353,
                3.117393885836001,
                -3.761930177913743,
                0.770646024799205,
                0,
            ],
            [
                0.214823667785537,
                0.536367363903245,
                0.154488125726409,
                -0.217748592703941,
                0.072226422925896,
                0.239843012362853,
            ],
        ]
    ),
    np.array(
        [
            0.214823667785537,
            0.536367363903245,
            0.154488125726409,
            -0.217748592703941,
            0.072226422925896,
            0.239843012362853,
        ],
    ),
    np.array(
        [
            0.079672377876931,
            0.464364648310935,
            1.348559241946724,
            1.312664210308764,
            0.989469293495897,
            1.0,
        ]
    ),
    p=4,
    name="3rd weak stage 6-stage DIRK, 4th order",
)

# Embedded Schemes
# ------------------

embedded_heun_euler = EmbeddedButcherTableau(
    np.array([[0.0, 0.0], [1.0, 0.0]]),
    np.array([0.5, 0.5]),
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    p=2,
    phat=1,
    name="Embedded Heun's Method with Euler method",
)


def create_second_order_two_stage_sdirk():
    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    tableau = EmbeddedButcherTableau(
        np.array([[gamma, 0.0], [1 - gamma, gamma]]),
        np.array([1 - gamma, gamma]),
        np.array([0.5, 0.5]),
        np.array([gamma, 1.0]),
        p=2,
        phat=1,
        name="SDIRK 2-stage, 2nd order",
    )
    return tableau


embedded_second_order_two_stage_sdirk = create_second_order_two_stage_sdirk()


def create_second_order_three_stage_esdirk():
    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    b2 = (1 - 2 * gamma) / (4 * gamma)
    b2_hat = (
        gamma * (-2 + 7 * gamma - 5 * gamma**2 + 4 * gamma**3) / (2 * (2 * gamma - 1))
    )
    b3_hat = -2 * gamma**2 * (1 - gamma + gamma**2) / (2 * gamma - 1)
    tableau = EmbeddedButcherTableau(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [gamma, gamma, 0.0],
                [1 - b2 - gamma, b2, gamma],
            ]
        ),
        np.array([1 - b2 - gamma, b2, gamma]),
        np.array([1 - b2_hat - b3_hat, b2_hat, b3_hat]),
        np.array([0.0, 2 * gamma, 1.0]),
        p=2,
        phat=1,
        name="ESDIRK 3-stage, 2nd order",
    )
    return tableau


embedded_second_order_three_stage_esdirk = create_second_order_three_stage_esdirk()


def create_third_order_four_stage_esdirk():
    gamma = 0.435866521508458999416019
    c3 = (3 - 20 * gamma + 24 * gamma**2) / (4 - 24 * gamma + 24 * gamma**2)
    a32 = c3 * (c3 - 2 * gamma) / (4 * gamma)
    b2 = (-2 + 3 * c3 + 6 * gamma * (1 - c3)) / (12 * gamma * (c3 - 2 * gamma))
    b3 = (1 - 6 * gamma + 6 * gamma**2) / (3 * c3 * (c3 - 2 * gamma))
    # for the adaptive part
    polynomial = 1 - 6 * gamma + 6 * gamma**2
    linear = 2 * gamma - c3
    b2_hat = c3 * (
        -1 + 6 * gamma - 24 * gamma**3 + 12 * gamma**4 - 6 * gamma**5
    ) / (4 * gamma * linear * polynomial) + (
        3 - 27 * gamma + 68 * gamma**2 - 55 * gamma**3 + 21 * gamma**4 - 6 * gamma**5
    ) / (
        2 * linear * polynomial
    )
    b3_hat = (
        -gamma
        * (
            -2
            + 21 * gamma
            - 68 * gamma**2
            + 79 * gamma**3
            - 33 * gamma**4
            + 12 * gamma**5
        )
        / (c3 * -linear * polynomial)
    )
    b4_hat = -3 * gamma**2 * (-1 + 4 * gamma - 2 * gamma**2 + gamma**3) / polynomial

    tableau = EmbeddedButcherTableau(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [gamma, gamma, 0.0, 0.0],
                [c3 - a32 - gamma, a32, gamma, 0.0],
                [1 - b2 - b3 - gamma, b2, b3, gamma],
            ]
        ),
        np.array([1 - b2 - b3 - gamma, b2, b3, gamma]),
        np.array([1 - b2_hat - b3_hat - b4_hat, b2_hat, b3_hat, b4_hat]),
        np.array([0.0, 2 * gamma, c3, 1]),
        p=3,
        phat=2,
        name="ESDIRK 4-stage, 3rd order",
    )
    return tableau

# https://doi.org/10.1007/BF01934920
embedded_third_order_three_stage_sdirk = EmbeddedButcherTableau(
    np.array(
        [
            [ 5/6, 0.0, 0.0],
            [-61/108, 5/6, 0.0],
            [-23/183, -33/61, 5/6],
        ]
    ),
    np.array(
        [25/61, 36/61, 0.0]
    ),
    np.array(
        [26/61, 324/671, 1/11]
    ),
    np.array(
        [5/6, 29/108, 1/6]
    ),
    p=2,
    phat=3,
    name="SDIRK 3-stage, 3rd order",
)

embedded_third_order_five_stage_esdirk = EmbeddedButcherTableau(
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [9 / 40, 9 / 40, 0.0, 0.0, 0.0],
            [9 * (1 + 2**0.5) / 80, 9 * (1 + 2**0.5) / 80, 9 / 40, 0.0, 0.0],
            [
                (22 + 15 * 2**0.5) / (80 * (1 + 2**0.5)),
                (22 + 15 * 2**0.5) / (80 * (1 + 2**0.5)),
                -7 / (40 * (1 + 2**0.5)),
                9 / 40,
                0.0,
            ],
            [
                (2398 + 1205 * 2**0.5) / (2835 * (4 + 3 * 2**0.5)),
                (2398 + 1205 * 2**0.5) / (2835 * (4 + 3 * 2**0.5)),
                (-2374 * (1 + 2 * 2**0.5)) / (2835 * (5 + 3 * 2**0.5)),
                5827 / 7560,
                9 / 40,
            ],
        ]
    ),
    np.array(
        [
            (2398 + 1205 * 2**0.5) / (2835 * (4 + 3 * 2**0.5)),
            (2398 + 1205 * 2**0.5) / (2835 * (4 + 3 * 2**0.5)),
            (-2374 * (1 + 2 * 2**0.5)) / (2835 * (5 + 3 * 2**0.5)),
            5827 / 7560,
            9 / 40,
        ]
    ),
    np.array(
        [
            4555948517383 / 24713416420891,
            4555948517383 / 24713416420891,
            -7107561914881 / 25547637784726,
            30698249 / 44052120,
            49563 / 233080,
        ]
    ),
    np.array([0.0, 9 / 20, 9 * (2 + 2**0.5) / 40, 0.8, 1]),
    p=3,
    phat=2,
    name="ESDIRK 5-stage, 3rd order",
)


embedded_third_order_four_stage_esdirk = create_third_order_four_stage_esdirk()


def create_fourth_order_five_stage_esdirk():  # STIFF METHOD
    gamma = 0.43586652150845899941601945
    c3 = (2 * gamma * (2 - 9 * gamma + 12 * gamma**2)) / (1 - 6 * gamma + 12 * gamma**2)
    c4 = 1.0
    phi1 = 1 - 6 * gamma + 6 * gamma**2
    phi2 = 3 - 20 * gamma + 24 * gamma**2
    phi3 = 5 - 36 * gamma + 48 * gamma**2
    phi4 = -1 + 12 * gamma - 36 * gamma**2 + 24 * gamma**3
    b2 = (
        3
        - 12 * gamma
        + 4 * c4 * (-1 + 3 * gamma)
        - 2 * c3 * (2 - 6 * gamma + c4 * (-3 + 6 * gamma))
    ) / (24 * gamma * (2 * gamma - c3) * (2 * gamma - c4))
    b3 = (phi2 - 4 * c4 * phi1) / (12 * c3 * (c3 - c4) * (c3 - 2 * gamma))
    b4 = (phi2 - 4 * c3 * phi1) / (12 * c4 * (c4 - c3) * (c4 - 2 * gamma))
    a32 = (c3 * (c3 - 2 * gamma)) / (4 * gamma)
    a42 = (
        c4
        * (c4 - 2 * gamma)
        * (-4 * c3**2 * phi1 - 2 * gamma * phi2 + c3 * phi3 + 2 * c4 * phi4)
        / (4 * gamma * (2 * gamma - c3) * (4 * c3 * phi1 - phi2))
    )
    a43 = ((c4 - c3) * c4 * (c4 - 2 * gamma) * phi4) / (
        c3 * (c3 - 2 * gamma) * (4 * c3 * phi1 - phi2)
    )
    # for the adaptive part
    phi1_hat = (
        2
        - 43 * gamma
        + 336 * gamma**2
        - 1194 * gamma**3
        + 1966 * gamma**4
        - 1336 * gamma**5
        + 168 * gamma**6
        + 96 * gamma**7
    )
    b5_hat = (
        4
        * gamma**2
        * (-1 + 9 * gamma - 18 * gamma**2 + 6 * gamma**3 - 2 * gamma**4)
        / (-1 + 12 * gamma - 36 * gamma**2 + 24 * gamma**3)
    )
    b4_hat = (
        -gamma
        * phi1_hat
        * (
            -3
            + 4 * c3
            + 20 * gamma
            - 24 * c3 * gamma
            - 24 * gamma**2
            + 24 * c3 * gamma**2
        )
        / (
            3
            * c4
            * (c4 - c3)
            * (c4 - 2 * gamma)
            * (-1 + 12 * gamma - 36 * gamma**2 + 24 * gamma**3) ** 2
        )
    )
    b3_hat = (
        1
        - 3 * b5_hat
        - 3 * b4_hat * c4**2
        - 3 * gamma
        + 6 * b5_hat * gamma
        + 6 * b4_hat * c4 * gamma
    ) / (3 * c3 * (c3 - 2 * gamma))
    b2_hat = (
        2
        - 6 * b5_hat
        - 3 * c3
        + 6 * b5_hat * c3
        + 6 * b4_hat * c3 * c4
        - 6 * b4_hat * c4**2
    ) / (12 * gamma * (2 * gamma - c3))

    tableau = EmbeddedButcherTableau(
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [gamma, gamma, 0.0, 0.0, 0.0],
                [c3 - a32 - gamma, a32, gamma, 0.0, 0.0],
                [c4 - a42 - a43 - gamma, a42, a43, gamma, 0.0],
                [1 - b2 - b3 - b4 - gamma, b2, b3, b4, gamma],
            ]
        ),
        np.array([1 - b2 - b3 - b4 - gamma, b2, b3, b4, gamma]),
        np.array(
            [1 - b5_hat - b4_hat - b3_hat - b2_hat, b2_hat, b3_hat, b4_hat, b5_hat]
        ),
        np.array([0.0, 2 * gamma, c3, c4, 1.0]),
        p=4,
        phat=3,
        name="ESDIRK 5-stage, 4th order",
    )

    return tableau


embedded_fourth_order_five_stage_esdirk = create_fourth_order_five_stage_esdirk()

# Source: https://doi.org/10.1016/j.aej.2015.12.026
embedded_runge_kutta_fehlberg = EmbeddedButcherTableau(
    np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3 / 32, 9 / 32, 0.0, 0.0, 0.0, 0.0],
            [1932 / 2197, -7200 / 2197, 7296 / 2197, 0.0, 0.0, 0.0],
            [439 / 216, -8.0, 3680 / 513, -845 / 4104, 0.0, 0.0],
            [-8 / 27, 2.0, -3544 / 2565, 1859 / 4014, -11 / 40, 0.0],
        ]
    ),
    np.array([16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]),
    np.array([25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0]),
    np.array([0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2]),
    p=5,
    phat=4,
    name="The Runge–Kutta–Fehlberg method",
)


def create_fifth_order_six_stage_esdirk():
    a = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                3282482714977 / 11805205429139,
                3282482714977 / 11805205429139,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                606638434273 / 1934588254988,
                2719561380667 / 6223645057524,
                3282482714977 / 11805205429139,
                0.0,
                0.0,
                0.0,
            ],
            [
                -651839358321 / 6893317340882,
                -1510159624805 / 11312503783159,
                235043282255 / 4700683032009,
                3282482714977 / 11805205429139,
                0.0,
                0.0,
            ],
            [
                -5266892529762 / 23715740857879,
                -1007523679375 / 1037568336475,
                521543607658 / 16698046240053,
                514935039541 / 7366641897523,
                3282482714977 / 11805205429139,
                0.0,
            ],
            [
                -6225479754948 / 6925873918471,
                6894665360202 / 11185215031699,
                -2508324082331 / 20512393166649,
                -7289596211309 / 4653106810017,
                39811658682819 / 14781729060964,
                3282482714977 / 11805205429139,
            ],
        ]
    )
    c = np.array(
        [
            0.0,
            4024571134387 / 7237035672548,
            14228244952610 / 13832614967709,
            0.1,
            3 / 50,
            1.0,
        ]
    )
    b = np.array(
        [
            -6225479754948 / 6925873918471,
            6894665360202 / 11185215031699,
            -2508324082331 / 20512393166649,
            -7289596211309 / 4653106810017,
            39811658682819 / 14781729060964,
            3282482714977 / 11805205429139,
        ]
    )
    bhat = np.array(
        [
            -2512930284403 / 5616797563683,
            5849584892053 / 8244045029872,
            -718651703996 / 6000050726475,
            -18982822128277 / 13735826808854,
            23127941173280 / 11608435116569,
            2847520232427 / 11515777524847,
        ]
    )
    return EmbeddedButcherTableau(
        a,
        b,
        bhat,
        c,
        p=5,
        phat=3,
        name="ESDIRK 6-stage, 5th order",
    )


embedded_fifth_order_six_stage_esdirk = create_fifth_order_six_stage_esdirk()
