from collections.abc import Callable

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as second_order_sdirk,
    embedded_second_order_three_stage_esdirk as second_order_esdirk,
    embedded_third_order_three_stage_sdirk as third_order_sdirk,
    embedded_third_order_four_stage_esdirk as third_order_esdirk,
    embedded_fourth_order_four_stage_sdirk as fourth_order_sdirk,
    embedded_fourth_order_five_stage_esdirk as fourth_order_esdirk,
)
from .problems import parse_problem

PROBLEM = parse_problem()

MARKER: list[str] = ["-o", "-X", "-P", "-D", "-v", "-H"]
COLORS: list[str] = [
    "indigo",
    "indianred",
    "seagreen",
    "slategray",
    "orange",
    "lightskyblue",
]
BUTCHER_NAMES: list[str] = [
    "SDIRK2",
    "ESDIRK2",
    "SDIRK3",
    "ESDIRK3",
    "SDIRK4",
    "ESDIRK4",
]
BUTCHER_TABLEAUX: list[Callable[[], ButcherTableau]] = [
    second_order_sdirk,
    second_order_esdirk,
    third_order_sdirk,
    third_order_esdirk,
    fourth_order_sdirk,
    fourth_order_esdirk,
]
