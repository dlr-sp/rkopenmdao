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
        import rkopenmdao.odes as odes

        return odes
    elif attr == "utils":
        import rkopenmdao.utils as utils

        return utils
    elif attr == "checkpoint_interface":
        import rkopenmdao.checkpoint_interface as checkpoint_interface

        return checkpoint_interface
    elif attr == "discretized_ode":
        import rkopenmdao.discretized_ode as discretized_ode

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
