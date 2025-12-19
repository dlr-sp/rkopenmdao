from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.rk_setup import IntegrationConfig, rk_setup

if __name__ == "__main__":
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0.1, PROBLEM.time_objective, 0.0),
        [integral],
        SimpleErrorMeasurer(),
        "",
        {"config": ErrorControllerConfig(tol=1e-6)},
    )
    for butcher_tableau in BUTCHER_TABLEAUX:
        integration_config.write_file = PROBLEM.get_file_path(
            butcher_tableau.name, "adaptive"
        )[1]
        rk_setup(PROBLEM, butcher_tableau, integration_config)
