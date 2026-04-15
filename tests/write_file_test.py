"""Tests the correct writing into hdf5-files."""

from __future__ import annotations


import pathlib
from typing import List, Tuple, Callable


import h5py
import numpy as np
import openmdao.api as om
import pytest

from rkopenmdao.butcher_tableaux import (
    embedded_third_order_four_stage_esdirk,
)
from rkopenmdao.error_controllers import pseudo, integral
from rkopenmdao.file_writer import (
    read_hdf5_file,
    read_last_local_error,
    OpenMDAOHDF5Callback,
)
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.termination_criterion import (
    PredefinedNumberOfSteps,
    PredefinedFinalTime,
)

from .test_components import (
    TestComp1,
    Testcomp51,
    Testcomp52,
    solution_test1,
    solution_test5,
)


# ########################
# Helper funnctions
# ########################
WRITE_FILE = "rk_write_out.h5"


def _mono_prob() -> om.Problem:
    """A single‑component problem that only produces the quantity ``x``."""
    prob = om.Problem()
    prob.model.add_subsystem("test_comp", TestComp1())
    return prob


def _multi_prob() -> om.Problem:
    """A two‑component problem that produces ``x`` and ``y`` (promoted)."""
    prob = om.Problem()
    prob.model.add_subsystem(
        "test_comp_1",
        Testcomp51(),
        promotes=["*"],
    )
    prob.model.add_subsystem(
        "test_comp_2",
        Testcomp52(),
        promotes=["*"],
    )
    return prob


def _make_rk_problem(
    *,
    time_stage_problem: om.Problem,
    integration_config: IntegrationConfig,
    write_out_distance: int,
    quantities: List[str],
    error_controller: List = None,
) -> om.Problem:
    """
    Factory that builds a ``Problem`` containing a single ``RungeKuttaIntegrator``.
    """
    file_writer_callback = OpenMDAOHDF5Callback(
        filename=WRITE_FILE, write_out_period=write_out_distance
    )

    rk_prob = om.Problem()
    rk_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem,
            butcher_tableau=embedded_third_order_four_stage_esdirk,
            integration_config=integration_config,
            error_controller=error_controller or [integral],
            time_integration_quantities=quantities,
            compute_callbacks=[file_writer_callback],
        ),
        promotes=["*"],
    )
    return rk_prob


def _assert_time(
    h5_group: h5py.Group,
    *,
    step: int,
    dt: float,
    t0: float,
    atol: float = 1e-12,
) -> None:
    """Check the ``time`` attribute stored for a given step."""
    expected = pytest.approx(step * dt + t0, rel=0, abs=atol)
    assert h5_group["time"][0] == expected


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


@pytest.fixture(name="mono_problem")
def monodisciplinary_problem_fixture() -> om.Problem:
    """A single‑component problem that produces ``x``."""
    return _mono_prob()


@pytest.fixture(name="multi_problem")
def multidisciplinary_problem_fixture() -> om.Problem:
    """A two‑component problem that produces ``x`` and ``y`` (promoted)."""
    return _multi_prob()


@pytest.fixture(name="nd_array_problem")
def n_d_array_problem_fixture() -> Tuple[om.Problem, List[int]]:
    """Problem that exposes a user‑defined N‑D array with tag ``time_int``."""
    prob = om.Problem()
    indep = om.IndepVarComp()
    indep.add_output(
        "time_int_test_output",
        shape=(2, 2),
        val=0.0,
        tags=["stage_output_var", "time_int"],
    )
    prob.model.add_subsystem("time_int_indep", indep)
    return prob, [2, 2]


@pytest.fixture(name="parallel_problem")
def parallel_problem_fixture(
    shape: Tuple[int, ...],
) -> Tuple[om.Problem, Tuple[int, ...]]:
    """Problem for the MPI test – a distributed variable with the requested shape."""
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
    return prob, shape


@pytest.fixture(name="mono_h5")
def monodisciplinary_h5() -> (
    Tuple[pathlib.Path, List[str], Callable[[float, float, float, float], float]]
):
    """
    Run a monodisciplinary problem (TestComp1 for quantity ``x``) and write the
    results to a temporary HDF5 file.  The fixture returns the file path and the
    list of quantities that were stored.
    """
    # Build the “stage” problem that only contains TestComp1
    _integration_con = IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(2.0),
        initial_step_size=0.01,
    )
    stage = _mono_prob()
    # Wrap it with the RK integrator
    rk_prob = _make_rk_problem(
        time_stage_problem=stage,
        integration_config=_integration_con,
        write_out_distance=1,
        quantities=["x"],
        error_controller=[pseudo],
    )
    rk_prob.setup()
    rk_prob["time_initial"] = 1.0
    rk_prob.run_model()

    return ["x"], solution_test1


@pytest.fixture(name="multi_h5")
def multidisciplinary_h5() -> (
    Tuple[pathlib.Path, List[str], Callable[[float, float, float], np.ndarray]]
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
        error_controller=[pseudo],
    )
    rk_prob.setup()
    rk_prob["time_initial"] = 1.0
    rk_prob.run_model()
    return ["x", "y"], solution_test5


# ########################
# Tests
# ########################


@pytest.mark.parametrize("write_out_distance", [1, 10, 25])
def test_monodisciplinary(
    mono_problem, integration_config, initial_time: float, write_out_distance: int
) -> None:
    """Write‑out for a monodisciplinary problem (only ``x``)."""
    rk_prob = _make_rk_problem(
        time_stage_problem=mono_problem,
        integration_config=integration_config,
        write_out_distance=write_out_distance,
        quantities=["x"],
    )
    rk_prob.setup()
    rk_prob["time_initial"] = initial_time
    rk_prob.run_model()

    dt = integration_config.initial_step_size

    with h5py.File(WRITE_FILE, "r") as f:

        for step in range(0, 100, write_out_distance):
            assert str(step) in f.keys(), f"Step {step} missing from file."

            assert "x" in f[str(step)].keys(), "Missing dataset 'x' in HDF5 file."

            _assert_time(f[str(step)], step=step, dt=dt, t0=initial_time)

        # make sure *no* extra steps were written
        for step in range(1, write_out_distance):
            assert str(step) not in f.keys(), f"Unexpected step {step} written."

        # the final value stored in the file must match the model output
        np.testing.assert_array_equal(
            rk_prob["rk_integration.x_final"],
            f[str(100)]["x"][:],
            err_msg="Value of final step is wrong.",
        )


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
    """Write‑out when the stored quantity has shape > 1 (2×2 array in this case)."""
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
    parallel_problem: Tuple,
    integration_config: IntegrationConfig,
    initial_time: float,
    write_out_distance: int,
    shape: Tuple[int, ...],
) -> None:
    """
    Parallel write‑out test (requires an MPI‑enabled build of h5py).

    * Each rank supplies a different initial value.
    * The final dataset is read back on each rank and compared to the local view
      of the distributed variable stored in the OpenMDAO problem.
    """

    prob, _ = parallel_problem

    # Build the RK problem that writes to a temporary file.

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

    # Rank‑0 gets zeros, all other ranks get ones – this matches the original test.
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


def test_read_hdf5_file_monodisciplinary(mono_h5):
    """
    Read the quantity ``x`` of a monodisciplinary problem and assert for each step
    whether the function `read_hdf5_file` provides correct time, error
    and result parameters.
    """
    quantities, solution = mono_h5
    time_dict, error_dict, result_dict = read_hdf5_file(
        WRITE_FILE, quantities, solution
    )
    with h5py.File(WRITE_FILE, "r") as f:
        for step_str, group in f.items():
            step = int(step_str)
            assert time_dict[step] == group["time"]
            # stored results in an array
            np.testing.assert_array_equal(result_dict["x"][step], group["x"])

            computed_error = np.abs(
                solution(time_dict[step], result_dict["x"][0], time_dict[0])
                - result_dict["x"][step]
            )
            assert error_dict["x"][step] == computed_error


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


def test_read_last_local_error_exact_step(mono_h5):
    """Write an ``error_measure`` group with several steps and ask for the
    error belonging to the step that exactly matches ``time_objective/step_size``."""
    err = read_last_local_error(WRITE_FILE)
    with h5py.File(WRITE_FILE, "r") as f:
        assert (
            err == f["100"]["error_measure"][0]
        ), f"Error estimation of ``{mono_h5[0]}`` at step ``100`` is wrong."
