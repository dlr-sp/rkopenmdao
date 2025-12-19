from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.rk_setup import IntegrationConfig, rk_setup

integration_config = IntegrationConfig(
    TimeTerminationIntegrationControl(0, PROBLEM.time_objective, 0.0),
    [pseudo],
    SimpleErrorMeasurer(),
    "",
    {},
)

if __name__ == "__main__":
    for butcher_tableau in BUTCHER_TABLEAUX:
        for step_size in PROBLEM.delta_t:
            integration_config.integration_control = TimeTerminationIntegrationControl(
                step_size, PROBLEM.time_objective, 0.0
            )
            integration_config.write_file = (
                PROBLEM.folder_path
                / f"data_{step_size:.0E}_{butcher_tableau.name}.h5".replace(" ", "_")
                .replace(",", "")
                .lower()
            )
            rk_setup(PROBLEM, butcher_tableau, integration_config)
