"""Constants for the convergence study."""

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
from rkopenmdao.utils.problems import parse_problem

PROBLEM = parse_problem()

BUTCHER_TABLEAUX: dict[str, Callable[[], ButcherTableau]] = {
    "SDIRK2": second_order_sdirk,
    "ESDIRK2": second_order_esdirk,
    "SDIRK3": third_order_sdirk,
    "ESDIRK3": third_order_esdirk,
    "SDIRK4": fourth_order_sdirk,
    "ESDIRK4": fourth_order_esdirk,
}
