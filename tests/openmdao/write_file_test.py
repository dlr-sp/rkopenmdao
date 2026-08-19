"""Tests the correct writing into hdf5-files."""

from __future__ import annotations

from collections.abc import Callable
import pathlib


import h5py
import numpy as np
import openmdao.api as om
import pytest

from om_components import ODE2dSplit1, ODE2dSplit2, ode2d_analytical_solution

from rkopenmdao.butcher_tableaux import (
    embedded_third_order_four_stage_esdirk,
)
from rkopenmdao.checkpointed_time_integration.no_checkpoint_time_integration import (
    NoCheckpointTimeIntegration,
)
from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE
from rkopenmdao.error_controller import ErrorController
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.file_writer import (
    read_hdf5_file,
    OpenMDAOHDF5Callback,
)
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.openmdao_time_stepping import OpenMDAOTimeStepping
from rkopenmdao.termination_criterion import (
    PredefinedNumberOfSteps,
    PredefinedFinalTime,
)


from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
)

# ########################
# Helper funnctions
# ########################
WRITE_FILE = "rk_write_out.h5"


def _multi_prob() -> om.Problem:
    """A two‑component problem that produces ``x`` and ``y`` (promoted)."""
    prob = om.Problem()
    prob.model.add_subsystem(
        "test_comp_1",
        ODE2dSplit1(),
        promotes=["*"],
    )
    prob.model.add_subsystem(
        "test_comp_2",
        ODE2dSplit2(),
        promotes=["*"],
    )
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=-1)
    prob.model.linear_solver = om.ScipyKrylov()
    prob.setup()
    prob.final_setup()
    return prob


def _make_rk_problem(
    *,
    time_stage_problem: om.Problem,
    integration_config: IntegrationConfig,
    write_out_distance: int,
    quantities: list[str],
    error_controller: ErrorController | None = None,
) -> om.Problem:
    """
    Factory that builds a ``Problem`` containing a single ``RungeKuttaIntegrator``.
    """
    file_writer_callback = OpenMDAOHDF5Callback(
        filename=WRITE_FILE, write_out_period=write_out_distance
    )
    time_integration = NoCheckpointTimeIntegration(
        ode=OpenMDAOODE(
            time_stage_problem=time_stage_problem,
            time_integration_quantities=quantities,
        ),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(
            embedded_third_order_four_stage_esdirk
        ),
        time_integration_config=integration_config,
        integrate_callbacks=[file_writer_callback],
        error_controller=(
            error_controller if error_controller is not None else pseudo(1)
        ),
    )
    rk_prob = om.Problem()
    rk_prob.model.add_subsystem(
        "rk_integration",
        OpenMDAOTimeStepping(time_integrator=time_integration),
        promotes=["*"],
    )
    return rk_prob


# ########################
# Fixtures
# ########################


@pytest.fixture(name="initial_time")
def initial_time_fixture():
    """
    Fixture for the initial time of the problems.
    """
    return 1.0


@pytest.fixture(name="integration_config")
def integration_config_fixture():
    """Step‑termination control: dt=0.01, max 100 steps, start time = 1.0."""
    return IntegrationConfig(
        use_adaptive_time_stepping=False,
        termination_criterion=PredefinedNumberOfSteps(100),
        initial_step_size=0.01,
    )
    # return StepTerminationIntegrationControl(0.01, 100, 1.0)


@pytest.fixture(name="multi_problem")
def multidisciplinary_problem_fixture() -> om.Problem:
    """A two‑component problem that produces ``x`` and ``y`` (promoted)."""
    return _multi_prob()


@pytest.fixture(name="nd_array_problem")
def n_d_array_problem_fixture() -> tuple[om.Problem, list[int]]:
    """Problem that exposes a user-defined N-D array with tag ``time_int``."""
    prob = om.Problem()
    indep = om.IndepVarComp()
    indep.add_output(
        "time_int_test_output",
        shape=(2, 2),
        val=0.0,
        tags=["stage_output_var", "time_int"],
    )
    prob.model.add_subsystem("time_int_indep", indep, promotes=["*"])
    prob.setup()
    prob.final_setup()
    return prob, [2, 2]


@pytest.fixture(name="parallel_problem")
def parallel_problem_fixture(
    shape: tuple[int, ...],
) -> tuple[om.Problem, tuple[int, ...]]:
    """Problem for the MPI test - a distributed variable with the requested shape."""
    prob = om.Problem()
    indep = om.IndepVarComp()
    indep.add_output(
        "time_int_test_output",
        distributed=True,
        shape=shape,
        val=0.0,
        tags=["stage_output_var", "time_int"],
    )
    prob.model.add_subsystem("time_int_indep", indep)
    prob.setup()
    prob.final_setup()
    return prob, shape


@pytest.fixture(name="multi_h5")
def multidisciplinary_h5() -> (
    tuple[pathlib.Path, list[str], Callable[[float, float, float], np.ndarray]]
):
    """
    Run a multidisciplinary problem (Testcomp51 + Testcomp52 for quantities ``x``
    and ``y``) and write the results to a temporary HDF5 file.
    """

    _integration_con = IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(2.0),
        initial_step_size=0.01,
    )
    stage = _multi_prob()

    rk_prob = _make_rk_problem(
        time_stage_problem=stage,
        integration_config=_integration_con,
        write_out_distance=1,
        quantities=["x", "y"],
    )
    rk_prob.setup()
    rk_prob.final_setup()
    rk_prob["time_initial"] = 1.0
    rk_prob.run_model()
    return ["x", "y"], ode2d_analytical_solution


# ########################
# Tests
# ########################


@pytest.mark.parametrize("write_out_distance", [1, 10, 25])
def test_multidisciplinary(
    multi_problem,
    integration_config,
    initial_time: float,
    write_out_distance: int,
) -> None:
    """Write‑out for a multidisciplinary problem (quantities ``x`` and ``y``)."""
    rk_prob = _make_rk_problem(
        time_stage_problem=multi_problem,
        integration_config=integration_config,
        write_out_distance=write_out_distance,
        quantities=["x", "y"],
    )
    rk_prob.setup()
    rk_prob["time_initial"] = initial_time
    rk_prob.run_model()

    with h5py.File(WRITE_FILE, "r") as f:

        for step in range(0, 100, write_out_distance):
            assert str(step) in f.keys(), f"Step {step} missing from file."
            for q in ("x", "y"):
                assert q in f[str(step)], f"Missing group '{q}' in step {step}."
        for step in range(1, write_out_distance):
            assert str(step) not in f.keys(), f"Unexpected step {step} in file.'."

        # final values must match the model outputs
        np.testing.assert_array_equal(rk_prob["rk_integration.x_final"], f["100"]["x"])
        np.testing.assert_array_equal(rk_prob["rk_integration.y_final"], f["100"]["y"])


@pytest.mark.parametrize("write_out_distance", (1, 10, 25))
def test_n_d_array(
    nd_array_problem,
    integration_config,
    initial_time: float,
    write_out_distance: int,
) -> None:
    """Write‑out when the stored quantity has shape > 1 (2x2 array in this case)."""
    prob, shape = nd_array_problem

    rk_prob = _make_rk_problem(
        time_stage_problem=prob,
        integration_config=integration_config,
        write_out_distance=write_out_distance,
        quantities=["time_int"],
    )
    rk_prob.setup()
    # initialise the variable that the integrator will read
    rk_prob["time_int_initial"] = np.zeros(shape)
    rk_prob["time_initial"] = initial_time
    rk_prob.run_model()

    with h5py.File(WRITE_FILE, "r") as f:
        for step in range(0, 100, write_out_distance):
            assert str(step) in f.keys(), f"Step {step} missing."
            assert "time_int" in f[str(step)], "Missing dataset 'time_int'."

        for step in range(1, write_out_distance):
            assert str(step) not in f.keys(), f"Unexpected step {step} written."

        np.testing.assert_array_equal(
            rk_prob["rk_integration.time_int_final"],
            f["100"]["time_int"],
        )


@pytest.mark.mpi
@pytest.mark.parametrize("write_out_distance", [1, 10])
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_parallel_write_out(
    parallel_problem: tuple,
    integration_config: IntegrationConfig,
    initial_time: float,
    write_out_distance: int,
    shape: tuple[int, ...],
) -> None:
    """
    Parallel write‑out test (requires an MPI‑enabled build of h5py).

    * Each rank supplies a different initial value.
    * The final dataset is read back on each rank and compared to the local view
      of the distributed variable stored in the OpenMDAO problem.
    """

    prob, _ = parallel_problem
    rk_prob = _make_rk_problem(
        time_stage_problem=prob,
        integration_config=integration_config,
        write_out_distance=write_out_distance,
        quantities=["time_int"],
    )
    time_initial = om.IndepVarComp()
    time_initial.add_output("time_int_initial", shape=shape, distributed=True)
    rk_prob.model.add_subsystem("time_initial", time_initial, promotes=["*"])

    rk_prob.setup()
    rk_prob["time_initial"] = initial_time

    init_val = np.zeros(shape) if prob.comm.rank == 0 else np.ones(shape)
    rk_prob["time_int_initial"] = init_val

    rk_prob.run_model()

    with h5py.File(WRITE_FILE, "r", driver="mpio", comm=rk_prob.comm) as f:
        for step in range(0, 100, write_out_distance):
            assert str(step) in f.keys(), f"Step {step} missing."
            assert (
                "time_int" in f[str(step)].keys()
            ), "Missing group 'time_int' in parallel file."

        for step in range(1, write_out_distance):
            assert str(step) not in f.keys(), f"Unexpected step {step}."

        local_final = rk_prob.get_val(
            name="rk_integration.time_int_final", get_remote=False
        )
        slice_obj = slice(0, 2) if prob.comm.rank == 0 else slice(2, 4)
        h5_final = f["100"]["time_int"][slice_obj, ...]
        np.testing.assert_array_equal(local_final, h5_final)


# Tests for read_hdf_file


@pytest.mark.mpi
def test_read_hdf5_file_multidisciplinary_h5(multi_h5):
    """
    Read the quantity ``x``,``y`` of a multidisciplinary problem and assert for each
    step whether the function `read_hdf5_file` provides correct time, error and result
    parameters.
    """
    quantities, solution = multi_h5
    time_dict, error_dict, result_dict = read_hdf5_file(
        WRITE_FILE, quantities, solution
    )
    with h5py.File(WRITE_FILE, "r") as f:
        for step_str, group in f.items():
            step = int(step_str)
            assert time_dict[step] == group["time"]
            # stored results in an array
            for q in quantities:
                assert result_dict[q][step] == group[q]

    for step in result_dict[quantities[0]].keys():
        computed_error = np.abs(
            solution(
                time_dict[step],
                (result_dict[quantities[0]][0], result_dict[quantities[1]][0]),
                time_dict[0],
            )
            - np.array(
                [result_dict[quantities[0]][step], result_dict[quantities[1]][step]]
            )
        )
        for index, q in enumerate(quantities):
            assert error_dict[q][step] == computed_error[index]
