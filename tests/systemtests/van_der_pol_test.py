"""Makes sure the optimization capabilities stay the same via the van-der-Pol
oscillator example."""

import numpy as np
import openmdao.api as om
import pytest

from rkopenmdao.butcher_tableaux import third_order_four_stage_esdirk
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator

from examples.van_der_pol_oscillator.van_der_pol_oscillator_computation import (
    VanDerPolFunctional,
    VanDerPolComponent1,
    VanDerPolComponent2,
)

SIMULATION_TIME = 10.0
INPUT_TO_RESULT = [
    (
        0.1,
        np.array(
            [
                0.3,
                0.3,
                0.3,
            ]
        ),
        None,
        None,
        np.array(
            [
                2.094268984096502084e00,
                -2.975703230699157653e-01,
                7.200293106932552367e-02,
            ]
        ),
        8.08395735,
    ),
    (
        0.1,
        np.array(
            [
                2.094268984096502084e00,
                2.094268984096502084e00,
                -2.975703230699157653e-01,
                -2.975703230699157653e-01,
                7.200293106932552367e-02,
                7.200293106932552367e-02,
            ]
        ),
        None,
        None,
        np.array(
            [
                1.167544319500043448e00,
                3.635223632827306428e00,
                -2.558782657369365610e-01,
                5.327215906350509789e-01,
                -2.808768373488292405e-02,
                2.203624622764142546e-02,
            ]
        ),
        7.83356891,
    ),
]


def create_van_Der_pol_integrator(
    integration_control: IntegrationControl,
    num_parameters: int,
    mode: str,
    y1_optimization: bool = None,
    y2_optimization: bool = None,
) -> om.Problem:
    """Convenience function to create a time integrator for the van-der-Pol
    oscillator."""
    vdp_1 = VanDerPolComponent1(integration_control=integration_control)
    vdp_2 = VanDerPolComponent2(
        integration_control=integration_control,
        num_parameters=num_parameters,
        simulation_time=SIMULATION_TIME,
    )
    vdp_functional = VanDerPolFunctional()

    vdp_inner_prob = om.Problem()
    vdp_inner_prob.model.add_subsystem("vdp1", vdp_1, promotes=["*"])
    vdp_inner_prob.model.add_subsystem("vdp2", vdp_2, promotes=["*"])
    vdp_inner_prob.model.add_subsystem("vdp_functional", vdp_functional, promotes=["*"])

    vdp_inner_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=False, iprint=-1, restart_from_successful=True, maxiter=1000
    )
    vdp_inner_prob.model.linear_solver = om.ScipyKrylov(iprint=-1)

    vdp_rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=vdp_inner_prob,
        time_integration_quantities=["y1", "y2", "J"],
        butcher_tableau=third_order_four_stage_esdirk,
        integration_control=integration_control,
        time_independent_input_quantities=["epsilon"],
        checkpointing_type=PyrevolveCheckpointer,
    )

    rk_prob = om.Problem()
    rk_prob.model.add_subsystem("rk_integration", vdp_rk_integrator, promotes=["*"])
    rk_prob.model.add_design_var("epsilon", lower=-10, upper=10)
    if y1_optimization:
        rk_prob.model.add_design_var("y1_initial", lower=-3, upper=3)
    if y2_optimization:
        rk_prob.model.add_design_var("y2_initial", lower=-3, upper=3)
    rk_prob.model.add_objective("J_final")
    rk_prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")

    rk_prob.setup(mode=mode)
    return rk_prob


@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize(
    "delta_t, initial_epsilons, y1_optimization, y2_optimization, expected_eps, expected_j",
    INPUT_TO_RESULT,
)
def test_van_der_pol_regression(
    mode,
    delta_t,
    initial_epsilons,
    y1_optimization,
    y2_optimization,
    expected_eps,
    expected_j,
):
    """Assures that code changes don't change the optimization results of the van-der-
    Pol example"""
    num_steps = int(SIMULATION_TIME / delta_t)
    integration_control = IntegrationControl(0.0, num_steps, delta_t)
    rk_prob = create_van_Der_pol_integrator(
        integration_control,
        initial_epsilons.size,
        mode,
        y1_optimization,
        y2_optimization,
    )
    rk_prob["y1_initial"] = 2.0
    rk_prob["y2_initial"] = 2.0

    rk_prob["epsilon"] = initial_epsilons

    rk_prob.run_driver()

    assert rk_prob["epsilon"] == pytest.approx(expected_eps)
    assert rk_prob["J_final"] == pytest.approx(expected_j)


def test_zero_optimality():
    """With an epsilon of zero and initial conditions on the unit circle, the functional
    of the van-der-Pol example should be 0. This tests that."""
    num_steps = int(SIMULATION_TIME / 0.1)
    integration_control = IntegrationControl(0.0, num_steps, 0.1)
    rk_prob = create_van_Der_pol_integrator(
        integration_control,
        1,
        "auto",
        None,
        None,
    )
    rk_prob["y1_initial"] = 1.0
    rk_prob["y2_initial"] = 0.0

    rk_prob["epsilon"].fill(0.0)

    rk_prob.run_model()

    assert rk_prob["J_final"] == pytest.approx(0.0, abs=1e-6)


def test_conditional_monotonity():
    """When first running an optimization with a lower number of parameters, then using
    these parameters as an initial point for the epsilons of an optimization with double
    the parameters (by repeating each epsilon once), the resulting optimization must
    produce an functional value that is equal or lower than the first one."""
    num_steps = int(SIMULATION_TIME / 0.1)
    integration_control = IntegrationControl(0.0, num_steps, 0.1)
    rk_prob = create_van_Der_pol_integrator(
        integration_control,
        2,
        "auto",
        None,
        None,
    )
    rk_prob["y1_initial"] = 2.0
    rk_prob["y2_initial"] = 2.0

    rk_prob["epsilon"].fill(0.3)

    rk_prob.run_driver()
    final_epsilon_coarse = rk_prob["epsilon"].copy()
    j_final_coarse = rk_prob["J_final"].copy()

    rk_prob = create_van_Der_pol_integrator(
        integration_control,
        4,
        "auto",
        None,
        None,
    )
    rk_prob["y1_initial"] = 2.0
    rk_prob["y2_initial"] = 2.0

    rk_prob["epsilon"][0] = final_epsilon_coarse[0]
    rk_prob["epsilon"][1] = final_epsilon_coarse[0]
    rk_prob["epsilon"][2] = final_epsilon_coarse[1]
    rk_prob["epsilon"][3] = final_epsilon_coarse[1]

    rk_prob.run_driver()

    assert rk_prob["J_final"] <= j_final_coarse
