"""Tests the adaptive time integration with the various components of
test_components.py"""

# pylint: disable=duplicate-code

import os

import openmdao.api as om
import pytest
import h5py
import numpy as np


from rkopenmdao.integration_control import TimeTerminationIntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    embedded_second_order_three_stage_esdirk as three_stage_esdirk,
    embedded_heun_euler as heun_euler,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)
from rkopenmdao.checkpoint_interface.no_checkpointer import NoCheckpointer
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.error_controllers import (
    integral,
    h_211,
    h0_211,
    pid,
    h_312,
    h0_312,
    ppid,
    h_321,
    h0_321,
)
from rkopenmdao.error_estimator import SimpleErrorEstimator, ImprovedErrorEstimator
from .test_components import (
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    Testcomp51,
    Testcomp52,
    TestComp6,
    TestComp6a,
    TestComp7,
)

from .test_components import (
    solution_test1,
    solution_test2,
    solution_test3,
    solution_test4,
    solution_test6,
    solution_test7,
)

test_comp_class_list = [
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    Testcomp51,
    Testcomp52,
    TestComp6,
    TestComp6a,
    TestComp7,
]

error_controller_list = [
    integral,
    h0_211,
    pid,
]

times = np.linspace(1.0, 9.0, 3)
butcher_diagonal_elements = np.linspace(0.0, 1.0, 3)

countupper=0
countlower=0


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "test_class, test_functor, initial_time, initial_values",
    (
        [TestComp1, solution_test1, 0.0, np.array([1.0])],
        [TestComp1, solution_test1, 1.0, np.array([1.0])],
        [TestComp2, solution_test2, 0.0, np.array([1.0])],
        [TestComp2, solution_test2, 1.0, np.array([1.0])],
        [TestComp3, solution_test3, 0.0, np.array([1.0])],
        [TestComp3, solution_test3, 1.0, np.array([1.0])],
        [TestComp4, solution_test4, 0.0, np.array([[1.0, 1.0]])],
        [TestComp4, solution_test4, 1.0, np.array([[1.0, 1.0]])],
        [TestComp6, solution_test6, 0.0, np.array([1.0])],
        [TestComp6, solution_test6, 1.0, np.array([1.0])],
        [TestComp6a, solution_test6, 0.0, np.array([1.0])],
        [TestComp6a, solution_test6, 1.0, np.array([1.0])],
        [TestComp7, solution_test7, 1.0, np.array([1.0])],
        [TestComp7, solution_test7, 2.0, np.array([1.0])],
    ),
)
@pytest.mark.parametrize(
    "butcher_tableau",
    [
        two_stage_dirk,
    ],
)
@pytest.mark.parametrize("quantities", [["x"]])
@pytest.mark.parametrize(
    "test_estimator", [SimpleErrorEstimator]
)
@pytest.mark.parametrize("test_controller", error_controller_list)
def test_upper_bound(
    test_class,
    test_functor,
    initial_time,
    initial_values,
    butcher_tableau,
    quantities,
    test_estimator,
    test_controller,
):
    """Tests the whether the upper bound of the error controller functions."""
    global countupper
    end_time=1.0
    integration_control = TimeTerminationIntegrationControl(
        0.01, initial_time+end_time, initial_time
    )
    upper_bound = 0.007
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )
    time_integration_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True
    )
    time_integration_prob.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob = om.Problem()
    write_file = f"test_upper_bound{countupper}.h5"
    countupper+=1
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller_options={'tol':1e-5,'upper_bound':upper_bound},
            error_controller=[test_controller, integral],
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
            write_file=write_file,
            write_out_distance=1,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]

    runge_kutta_prob.run_model()

    result = np.zeros_like(initial_values)
    for i, quantity in enumerate(quantities):
        result[i] = runge_kutta_prob[quantity + "_final"]

    # relatively coarse, but this isn't supposed to test the accuracy,
    # it's just to make sure the solution is in the right region
    assert test_functor(
        initial_time+end_time,
        initial_values,
        initial_time,
    ) == pytest.approx(result, rel=1e-4)
    time_list = {}
    with h5py.File(write_file, mode="r") as f:
        group = f["x"]
        for key in group.keys():
            time_list.update({int(key): group[key].attrs["time"]})

    time_list = dict(sorted(time_list.items()))

    delta_t = [0] * (len(time_list)-1)
    for i in range(len(time_list) - 1):
        delta_t[i] = time_list[i + 1] - time_list[i]
    assert np.abs(max(np.max(delta_t),upper_bound) - upper_bound) <= 1e-12
    if os.path.exists(write_file):
        os.remove(write_file)


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "test_class, test_functor, initial_time, initial_values",
    (
        [TestComp1, solution_test1, 0.0, np.array([1.0])],
        [TestComp1, solution_test1, 1.0, np.array([1.0])],
        [TestComp2, solution_test2, 0.0, np.array([1.0])],
        [TestComp2, solution_test2, 1.0, np.array([1.0])],
        [TestComp3, solution_test3, 0.0, np.array([1.0])],
        [TestComp3, solution_test3, 1.0, np.array([1.0])],
        [TestComp4, solution_test4, 0.0, np.array([[1.0, 1.0]])],
        [TestComp4, solution_test4, 1.0, np.array([[1.0, 1.0]])],
        [TestComp6, solution_test6, 0.0, np.array([1.0])],
        [TestComp6, solution_test6, 1.0, np.array([1.0])],
        [TestComp6a, solution_test6, 0.0, np.array([1.0])],
        [TestComp6a, solution_test6, 1.0, np.array([1.0])],
        [TestComp7, solution_test7, 1.0, np.array([1.0])],
        [TestComp7, solution_test7, 2.0, np.array([1.0])],
    ),
)
@pytest.mark.parametrize(
    "butcher_tableau",
    [
        two_stage_dirk,
    ],
)
@pytest.mark.parametrize("quantities", [["x"]])
@pytest.mark.parametrize(
    "test_estimator", [SimpleErrorEstimator]
)
@pytest.mark.parametrize("test_controller", error_controller_list)
def test_lower_bound(
    test_class,
    test_functor,
    initial_time,
    initial_values,
    butcher_tableau,
    quantities,
    test_estimator,
    test_controller,
):
    """Tests the whether the lower bound of the error controller functions."""
    global countlower
    end_time=1.0
    integration_control = TimeTerminationIntegrationControl(
        0.01, initial_time+end_time, initial_time
    )
    lower_bound = 0.005
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )
    time_integration_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True
    )
    time_integration_prob.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob = om.Problem()
    write_file = f"test_upper_bound{str(countlower)}.h5"
    countlower+=1
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller_options={'tol':1e-5,'lower_bound':lower_bound},
            error_controller=[test_controller, integral],
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
            write_file=write_file,
            write_out_distance=1,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]

    runge_kutta_prob.run_model()

    result = np.zeros_like(initial_values)
    for i, quantity in enumerate(quantities):
        result[i] = runge_kutta_prob[quantity + "_final"]

    # relatively coarse, but this isn't supposed to test the accuracy,
    # it's just to make sure the solution is in the right region
    assert test_functor(
        initial_time+end_time,
        initial_values,
        initial_time,
    ) == pytest.approx(result, rel=1e-4)
    time_list = {}
    with h5py.File(write_file, mode="r") as f:
        group = f["x"]
        for key in group.keys():
            time_list.update({int(key): group[key].attrs["time"]})

    time_list = dict(sorted(time_list.items()))

    delta_t = [0] * (len(time_list)-1)
    for i in range(len(time_list) - 1):
        delta_t[i] = time_list[i + 1] - time_list[i]
    assert np.abs(min(np.min(delta_t),lower_bound) - lower_bound) <= 1e-12
    if os.path.exists(write_file):
        os.remove(write_file)




