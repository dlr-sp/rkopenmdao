import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

import pytest

from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import RungeKuttaIntegrator
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.butcher_tableaux import implicit_euler
from .test_components import TestComp6

revolver_set = {"SingleLevel", "MultiLevel", "Memory", "Disk", "Base"}


# TODO: tests with compression
@pytest.mark.rk
@pytest.mark.parametrize(
    "revolver_type, revolver_options",
    (
        ["SingleLevel", {}],
        ["SingleLevel", {"n_checkpoints": 20}],
        ["SingleLevel", {"n_checkpoints": 100}],
        ["SingleLevel", {"diskstorage": True}],
        ["SingleLevel", {"n_checkpoints": 20, "diskstorage": True}],
        ["SingleLevel", {"n_checkpoints": 100, "diskstorage": True}],
        # something is strange here. For certain checkpoint numbers, the MultiLevelRevolver works, but for others it doesn't
        # We skip this for now
        # TODO: investigate this
        # [
        #     "MultiLevel",
        #     {
        #         "storage_list": {
        #             "Numpy": {"n_ckp": 3, "dtype": float},
        #             "Disk": {"n_ckp": 5, "dtype": float},
        #         }
        #     },
        # ],
        # [
        #     "MultiLevel",
        #     {
        #         "storage_list": {
        #             "Numpy": {"n_ckp": 5, "dtype": float},
        #             "Disk": {"n_ckp": 5, "dtype": float},
        #         }
        #     },
        # ],
        ["Memory", {}],
        ["Memory", {"n_checkpoints": 20}],
        ["Memory", {"n_checkpoints": 100}],
        ["Disk", {}],
        ["Disk", {"n_checkpoints": 20}],
        ["Disk", {"n_checkpoints": 100}],
    ),
)
def test_rk_integrator_revolver_options(revolver_type, revolver_options):
    integration_control = IntegrationControl(0.0, 100, 0.01)

    inner_prob = om.Problem()
    inner_prob.model.add_subsystem(
        "test", TestComp6(integration_control=integration_control)
    )
    runge_kutta_prob = om.Problem()

    runge_kutta_prob.model.add_subsystem(
        "rk",
        RungeKuttaIntegrator(
            time_stage_problem=inner_prob,
            butcher_tableau=implicit_euler,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            revolver_type=revolver_type,
            revolver_options=revolver_options,
        ),
    )
    runge_kutta_prob.setup()
    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)
