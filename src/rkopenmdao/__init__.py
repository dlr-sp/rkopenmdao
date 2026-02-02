from . import butcher_tableau
from . import butcher_tableaux
from . import error_controller
from . import error_controllers
from . import error_measurer
from . import file_writer
from . import integration_control
from . import metadata_extractor
from . import runge_kutta_integrator
from . import runge_kutta_scheme


def __getattr__(attr):
    if attr == "odes":
        from rkopenmdao import odes

        return odes
    elif attr == "utils":
        from rkopenmdao import utils

        return utils
    elif attr == "checkpoint_interface":
        from rkopenmdao import checkpoint_interface

        return checkpoint_interface
    elif attr == "discretized_ode":
        from rkopenmdao import discretized_ode

        return discretized_ode
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")


__all__ = [
    "butcher_tableau",
    "butcher_tableaux",
    "error_controller",
    "error_controllers",
    "error_measurer",
    "file_writer",
    "integration_control",
    "metadata_extractor",
    "runge_kutta_integrator",
    "runge_kutta_scheme",
]
