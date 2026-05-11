"""Compact tests for adaptive RK integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openmdao.api as om
import pytest
from mpi4py import MPI

from rkopenmdao.error_controller import ErrorControllerConfig
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
from rkopenmdao.error_measurer import ImprovedErrorMeasurer, SimpleErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.checkpoint_interface.no_checkpointer import NoCheckpointer
from rkopenmdao.termination_criterion import PredefinedFinalTime
from rkopenmdao.butcher_tableaux import (
    embedded_heun_euler as heun_euler,
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    embedded_second_order_three_stage_esdirk as three_stage_esdirk,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)

from .test_components import (
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    TestComp6,
    TestComp6a,
    TestComp7,
    Testcomp51,
    Testcomp52,
    solution_test1,
    solution_test2,
    solution_test3,
    solution_test4,
    solution_test6,
    solution_test7,
)
from .utils.callback import TimeStepsLog, TimeStepsLogToFile

COMPONENTS = [
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

CONTROLLERS = [
    integral,
    h_211,
    h0_211,
    pid,
    h_312,
    h0_312,
    ppid,
    h_321,
    h0_321,
]

TABLEAUX = [
    heun_euler,
    two_stage_dirk,
    three_stage_esdirk,
    four_stage_esdirk,
    five_stage_esdirk,
]

MEASURERS = [SimpleErrorMeasurer(), ImprovedErrorMeasurer()]
CHECKPOINTERS = [NoCheckpointer, AllCheckpointer]

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()


@pytest.fixture(name="integration_cfg", scope="function")
def integration_cfg_fixture(request):
    """
    Base configuration (adaptive, short horizon, tiny initial step).
    """
    init_time = 0.0  # default

    if "component_case" in request.fixturenames:
        _, _, init_time, _ = request.getfixturevalue("component_case")
    return IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(init_time + 0.01),
        initial_step_size=0.01,
    )


@pytest.fixture(name="time_stage_problem")
def time_stage_problem_fixture():
    """
    Factory that builds the “inner” problem holding the test component.
    """

    def _make(comp_cls):
        prob = om.Problem()
        prob.model.add_subsystem("test_comp", comp_cls())
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = om.ScipyKrylov()
        return prob

    return _make


@pytest.fixture(name="rk_problem")
def rk_problem_fixture(integration_cfg, time_stage_problem):
    """
    Factory that builds the outer RK problem.
    Parameters are supplied via ``indirect`` parametrisation.
    """

    def _make(
        *,
        comp_cls,
        tableau,
        quantities,
        controller,
        measurer,
        checkpoint_impl=NoCheckpointer,
        callbacks=None,
        init_time: float,
    ):
        #  Build inner problem (time‑stage)
        inner = time_stage_problem(comp_cls)

        #  Build outer RK problem
        rk = om.Problem()
        rk.model.add_subsystem(
            "rk_integrator",
            RungeKuttaIntegrator(
                time_stage_problem=inner,
                butcher_tableau=tableau,
                integration_config=integration_cfg,
                time_integration_quantities=quantities,
                error_controller=[controller, integral],
                error_controller_options={"config": ErrorControllerConfig(tol=1e-6)},
                error_measurer=measurer,
                checkpointing_type=checkpoint_impl,
                compute_callbacks=callbacks or [],
            ),
            promotes=["*"],
        )
        rk.setup()
        rk["time_initial"] = init_time
        return rk

    return _make


@pytest.fixture(name="time_step_log")
def time_step_log_fixture():
    """Create a fresh TimeStepsLog."""
    return TimeStepsLog()


@pytest.fixture(
    name="component_case",
    params=[
        (TestComp1, solution_test1, 0.0, np.array([1.0])),
        (TestComp1, solution_test1, 1.0, np.array([1.0])),
        (TestComp2, solution_test2, 0.0, np.array([1.0])),
        (TestComp2, solution_test2, 1.0, np.array([1.0])),
        (TestComp3, solution_test3, 0.0, np.array([1.0])),
        (TestComp3, solution_test3, 1.0, np.array([1.0])),
        (TestComp4, solution_test4, 0.0, np.array([[1.0, 1.0]])),
        (TestComp4, solution_test4, 1.0, np.array([[1.0, 1.0]])),
        (TestComp6, solution_test6, 0.0, np.array([1.0])),
        (TestComp6, solution_test6, 1.0, np.array([1.0])),
        (TestComp6a, solution_test6, 0.0, np.array([1.0])),
        (TestComp6a, solution_test6, 1.0, np.array([1.0])),
        (TestComp7, solution_test7, 1.0, np.array([1.0])),
        (TestComp7, solution_test7, 2.0, np.array([1.0])),
    ],
)
def component_case_fixture(request):
    return request.param


def read_time_steps(log: TimeStepsLogToFile) -> list[float]:
    """
    Utility used by ``test_if_adaptive`` – returns a list of the first column.
    """

    assert Path(log.write_file).exists()

    with open(log.write_file, "r", encoding="utf-8") as f:
        return [float(line.split()[0]) for line in f]


def run_and_compare(rk_prob, quantities, init_vals, functor, init_time):
    """
    Run the RK problem and assert the numerical result matches the analytic one.
    """
    rk_prob.run_model()
    result = np.zeros_like(init_vals)
    for i, quantity in enumerate(quantities):
        result[i] = rk_prob[quantity + "_final"][0]
    assert functor(init_time + 0.01, init_vals, init_time) == pytest.approx(
        result, rel=1e-4
    )


def run_and_check_partials(rk_prob, checkpoint_impl):
    """
    Run the RK problem and, depending on the checkpoint implementation, test partials.
    """
    rk_prob.run_model()
    if checkpoint_impl is NoCheckpointer:
        with pytest.raises(NotImplementedError):
            rk_prob.check_partials()
    else:
        data = rk_prob.check_partials()
        _compare_fwd_rev(data)


def _compare_fwd_rev(jac_data: dict, tol: float = 1e-6):
    """
    Utility used by ``run_and_check_partials`` – forward vs reverse Jacobians.
    """
    for pair in jac_data["rk_integrator"]:
        fwd = jac_data["rk_integrator"][pair]["J_fwd"]
        rev = jac_data["rk_integrator"][pair]["J_rev"]
        assert rev == pytest.approx(fwd, rel=tol, abs=tol)


@pytest.mark.parametrize("tableau", TABLEAUX)
@pytest.mark.parametrize("measurer", MEASURERS)
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_component_integration(
    rk_problem,
    component_case,
    tableau,
    measurer,
    controller,
):
    """
    Validate that the adaptive RK integrator reproduces the known analytic solution.
    """
    comp_cls, functor, init_time, init_vals = component_case
    rk = rk_problem(
        comp_cls=comp_cls,
        tableau=tableau,
        quantities=["x"],
        controller=controller,
        measurer=measurer,
        init_time=init_time,
        callbacks=None,
    )
    # Set the initial condition(s)
    rk["x_initial"] = init_vals[0] if init_vals.ndim == 1 else init_vals
    # Run and compare
    run_and_compare(rk, ["x"], init_vals, functor, init_time)


@pytest.mark.parametrize("tableau", TABLEAUX)
@pytest.mark.parametrize("ckpt_impl", CHECKPOINTERS)
def test_time_integration_partials(
    rk_problem,
    component_case,
    tableau,
    ckpt_impl,
):
    """
    Check that forward‑ and reverse‑mode Jacobians match
    (when checkpointing is available).
    """
    comp_cls, _, init_time, _ = component_case
    rk = rk_problem(
        comp_cls=comp_cls,
        tableau=tableau,
        quantities=["x"],
        controller=integral,
        measurer=SimpleErrorMeasurer(),
        checkpoint_impl=ckpt_impl,
        init_time=init_time,
    )
    run_and_check_partials(rk, ckpt_impl)


@pytest.mark.parametrize("tableau", TABLEAUX[0:3])
def test_if_adaptive(rk_problem, time_step_log, component_case, tableau):
    """
    Ensure that the integrator really changes its step size
    (i.e. logs >1 distinct values).
    """
    callbacks = [time_step_log] if COMM.Get_rank() == 0 else []
    comp_cls, _, init_time, _ = component_case
    rk = rk_problem(
        comp_cls=comp_cls,
        tableau=tableau,
        quantities=["x"],
        controller=integral,
        measurer=SimpleErrorMeasurer(),
        callbacks=callbacks,
        init_time=init_time,
    )
    rk.run_model()
    if RANK == 0:
        steps = list(time_step_log.q)
        assert len(steps) > 1, "Only one time step was recorded"
        assert len(set(steps)) > 1, "All recorded time steps are identical"


def test_new_old_adaptive_comparison(rk_problem, time_step_log):
    """
    Ensure that the integrator change steps size are identical to the one in './data.'
    """
    callbacks = [time_step_log] if RANK == 0 else []
    rk = rk_problem(
        comp_cls=TestComp1,
        tableau=heun_euler,
        quantities=["x"],
        controller=integral,
        measurer=SimpleErrorMeasurer(),
        callbacks=callbacks,
        init_time=0.0,
    )
    rk.run_model()
    if RANK == 0:
        steps_new = np.array(list(time_step_log.q))
        steps_old = np.array(
            read_time_steps(TimeStepsLogToFile(f"tests/data/time_step_{0}.txt"))
        )
        assert np.allclose(steps_new, steps_old)
    COMM.Barrier()
