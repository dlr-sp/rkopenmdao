"""Compact tests for adaptive RK integration (originally test_adaptive_time_integration.py)."""

from __future__ import annotations

from itertools import product

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
from .utils.callback import TimeStepsLog

# ----------------------------------------------------------------------
#  Parameter collections (kept unchanged)
# ----------------------------------------------------------------------
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

TABLEAUS = [
    heun_euler,
    two_stage_dirk,
    three_stage_esdirk,
    four_stage_esdirk,
    five_stage_esdirk,
]

MEASURERS = [SimpleErrorMeasurer(), ImprovedErrorMeasurer()]
CHECKPOINTERS = [NoCheckpointer, AllCheckpointer]


# ----------------------------------------------------------------------
#  Fixtures – the heavy lifting
# ----------------------------------------------------------------------
@pytest.fixture(scope="function")
def integration_cfg():
    """Base configuration (adaptive, short horizon, tiny initial step)."""
    return IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(0.01),  # dummy; overridden per test
        initial_step_size=0.01,
    )


@pytest.fixture
def time_stage_problem():
    """Factory that builds the “inner” problem holding the test component."""

    def _make(comp_cls):
        prob = om.Problem()
        prob.model.add_subsystem("test_comp", comp_cls())
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = om.ScipyKrylov()
        return prob

    return _make


@pytest.fixture
def rk_problem(integration_cfg, time_stage_problem):
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
        write_out_distance=0,
        init_time: float,
    ):
        # ------------------------------------------------------------------
        #  Build inner problem (time‑stage)
        # ------------------------------------------------------------------
        inner = time_stage_problem(comp_cls)

        # ------------------------------------------------------------------
        #  Build outer RK problem
        # ------------------------------------------------------------------
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
                callbacks=callbacks or [],
                write_out_distance=write_out_distance,
            ),
            promotes=["*"],
        )
        rk.setup()
        rk["time_initial"] = init_time
        return rk

    return _make


@pytest.fixture
def time_step_log(tmp_path_factory):
    """Create a fresh TimeStepsLog that writes into a temporary file."""
    tmp_file = tmp_path_factory.mktemp("timesteps") / "steps.txt"
    return TimeStepsLog(str(tmp_file))


def read_time_steps(log: TimeStepsLog) -> list[float]:
    """Utility used by ``test_if_adaptive`` – returns a list of the first column."""
    with open(log.write_file, "r") as f:
        return [float(line.split()[0]) for line in f]


# ----------------------------------------------------------------------
#  Helper utilities (run → compare / run → check partials)
# ----------------------------------------------------------------------
def run_and_compare(rk_prob, quantities, init_vals, functor, init_time):
    """Run the RK problem and assert the numerical result matches the analytic one."""
    rk_prob.run_model()
    result = np.array([rk_prob[q + "_final"][0] for q in quantities])
    assert functor(init_time + 0.01, init_vals, init_time) == pytest.approx(
        result, rel=1e-4
    )


def run_and_check_partials(rk_prob, checkpoint_impl):
    """Run the RK problem and, depending on the checkpoint implementation, test partials."""
    rk_prob.run_model()
    if checkpoint_impl is NoCheckpointer:
        with pytest.raises(NotImplementedError):
            rk_prob.check_partials()
    else:
        data = rk_prob.check_partials()
        _compare_fwd_rev(data)


def _compare_fwd_rev(jac_data: dict, tol: float = 1e-6):
    """Utility used by ``run_and_check_partials`` – forward vs reverse Jacobians."""
    for pair in jac_data["rk_integrator"]:
        fwd = jac_data["rk_integrator"][pair]["J_fwd"]
        rev = jac_data["rk_integrator"][pair]["J_rev"]
        assert rev == pytest.approx(fwd, rel=tol, abs=tol)


# ----------------------------------------------------------------------
#  Parametrised test cases (compact!)
# ----------------------------------------------------------------------
# ------------------------------------------------------------------
#  1  Component‑integration correctness
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "comp_cls, functor, init_time, init_vals",
    [
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
@pytest.mark.parametrize("tableau", TABLEAUS)
@pytest.mark.parametrize("measurer", MEASURERS)
@pytest.mark.parametrize("controller", CONTROLLERS)
def test_component_integration(
    rk_problem,
    comp_cls,
    functor,
    init_time,
    init_vals,
    tableau,
    measurer,
    controller,
):
    """Validate that the adaptive RK integrator reproduces the known analytic solution."""
    # Build a *single‑quantity* problem (the original tests always used ["x"])
    rk = rk_problem(
        comp_cls=comp_cls,
        tableau=tableau,
        quantities=["x"],
        controller=controller,
        measurer=measurer,
        init_time=init_time,
    )
    # Set the initial condition(s)
    rk["x_initial"] = init_vals[0] if init_vals.ndim == 1 else init_vals
    # Run and compare
    run_and_compare(rk, ["x"], init_vals, functor, init_time)


# ------------------------------------------------------------------
#  2  Partial‑derivative consistency (with/without checkpointing)
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "comp_cls, init_time",
    list(
        product(
            [TestComp1, TestComp2, TestComp3, TestComp4, TestComp6, TestComp6a],
            [0.0, 1.0],
        )
    )
    + [[TestComp7, 1.0]]
    + [[TestComp7, 2.0]],
)
@pytest.mark.parametrize("tableau", TABLEAUS)
@pytest.mark.parametrize("ckpt_impl", CHECKPOINTERS)
def test_time_integration_partials(
    rk_problem,
    comp_cls,
    init_time,
    tableau,
    ckpt_impl,
):
    """Check that forward‑ and reverse‑mode Jacobians match (when checkpointing is available)."""
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


# ------------------------------------------------------------------
#  3 Adaptive‑step logging sanity check
# ------------------------------------------------------------------
@pytest.mark.parametrize(
    "comp_cls, init_time",
    list(
        product(
            [TestComp1, TestComp2, TestComp3, TestComp4, TestComp6, TestComp6a],
            [0.0, 1.0],
        )
    )
    + [[TestComp7, 1.0]]
    + [[TestComp7, 2.0]],
)
@pytest.mark.parametrize("tableau", TABLEAUS)
def test_if_adaptive(rk_problem, time_step_log, comp_cls, init_time, tableau):
    """Ensure that the integrator really changes its step size (i.e. logs >1 distinct values)."""
    callbacks = [time_step_log] if MPI.COMM_WORLD.Get_rank() == 0 else []
    rk = rk_problem(
        comp_cls=comp_cls,
        tableau=tableau,
        quantities=["x"],
        controller=integral,
        measurer=SimpleErrorMeasurer(),
        callbacks=callbacks,
        write_out_distance=0,
        init_time=init_time,
    )
    rk.run_model()

    steps = read_time_steps(time_step_log)
    assert len(steps) > 1, "Only one time step was recorded"
    assert len(set(steps)) > 1, "All recorded time steps are identical"
