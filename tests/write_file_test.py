"""Tests the correct writing into hdf5-files."""

from __future__ import annotations


import pathlib
from typing import List, Tuple, Callable


import h5py
import numpy as np
import openmdao.api as om
import pytest

from rkopenmdao.integration_control import (
    StepTerminationIntegrationControl,
    TimeTerminationIntegrationControl,
)
from rkopenmdao.butcher_tableaux import (
    embedded_third_order_four_stage_esdirk,
)
from rkopenmdao.error_controllers import pseudo, integral
from rkopenmdao.file_writer import read_hdf5_file, read_last_local_error
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
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


def _mono_prob(
    integration_con,
) -> om.Problem:
    """A single‑component problem that only produces the quantity ``x``."""
    prob = om.Problem()
    prob.model.add_subsystem(
        "test_comp", TestComp1(integration_control=integration_con)
    )
    return prob


def _multi_prob(integration_con) -> om.Problem:
    """A two‑component problem that produces ``x`` and ``y`` (promoted)."""
    prob = om.Problem()
    prob.model.add_subsystem(
        "test_comp_1",
        Testcomp51(integration_control=integration_con),
        promotes=["*"],
    )
    prob.model.add_subsystem(
        "test_comp_2",
        Testcomp52(integration_control=integration_con),
        promotes=["*"],
    )
    return prob


def _make_rk_problem(
    *,
    time_stage_problem: om.Problem,
    integration_con,
    write_out_distance: int,
    quantities: List[str],
    adaptive: bool = False,
    error_controller: List = None,
) -> om.Problem:
    """
    Factory that builds a ``Problem`` containing a single ``RungeKuttaIntegrator``.
    """
    rk_prob = om.Problem()
    rk_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem,
            butcher_tableau=embedded_third_order_four_stage_esdirk,
            integration_control=integration_con,
            write_out_distance=write_out_distance,
            adaptive_time_stepping=adaptive,
            error_controller=error_controller or [integral],
            write_file=WRITE_FILE,
            time_integration_quantities=quantities,
        ),
        promotes=["*"],
    )
    return rk_prob


def _assert_time_attrs(
    h5_group: h5py.Group,
    *,
    step: int,
    dt: float,
    t0: float,
    atol: float = 1e-12,
) -> None:
    """Check the ``time`` attribute stored for a given step."""
    expected = pytest.approx(step * dt + t0, rel=0, abs=atol)
    assert h5_group.attrs["time"] == expected


# ########################
# Fixtures
# ########################


@pytest.fixture(name="integration_con")
def integration_control():
    """Step‑termination control: dt=0.01, max 100 steps, start time = 1.0."""
    return StepTerminationIntegrationControl(0.01, 100, 1.0)


@pytest.fixture(name="mono_problem")
def monodisciplinary_problem(
    integration_con,
) -> om.Problem:
    return _mono_prob(integration_con)


@pytest.fixture(name="multi_problem")
def multidisciplinary_problem(integration_con) -> om.Problem:
    """A two‑component problem that produces ``x`` and ``y`` (promoted)."""
    return _multi_prob(integration_con)


@pytest.fixture(name="nd_array_prob")
def n_d_array_problem() -> Tuple[om.Problem, List[int]]:
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


@pytest.fixture(name="parallel_prob")
def parallel_problem(shape: Tuple[int, ...]) -> Tuple[om.Problem, Tuple[int, ...]]:
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
    # TODO : Change code here.
    _integration_con = TimeTerminationIntegrationControl(0.01, 2.0, 1.0)
    stage = _mono_prob(_integration_con)
    # Wrap it with the RK integrator
    rk_prob = _make_rk_problem(
        time_stage_problem=stage,
        integration_con=_integration_con,
        write_out_distance=1,
        quantities=["x"],
        adaptive=True,
        error_controller=[pseudo],
    )
    rk_prob.setup()
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

    _integration_con = TimeTerminationIntegrationControl(0.01, 2.0, 1.0)
    stage = _multi_prob(_integration_con)

    rk_prob = _make_rk_problem(
        time_stage_problem=stage,
        integration_con=_integration_con,
        write_out_distance=1,
        quantities=["x", "y"],
        adaptive=True,
        error_controller=[pseudo],
    )
    rk_prob.setup()
    rk_prob.run_model()
    return ["x", "y"], solution_test5


# ########################
# Tests
# ########################


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10, 20, 30])
def test_monodisciplinary(
    mono_problem, integration_con, write_out_distance: int
) -> None:
    """Write‑out for a monodisciplinary problem (only ``x``)."""
    rk_prob = _make_rk_problem(
        time_stage_problem=mono_problem,
        integration_con=integration_con,
        write_out_distance=write_out_distance,
        quantities=["x"],
    )
    rk_prob.setup()
    rk_prob.run_model()

    with h5py.File(WRITE_FILE, "r") as f:
        assert "x" in f.keys(), "Missing group 'x' in HDF5 file."

        for step in range(0, 100, write_out_distance):
            assert str(step) in f["x"].keys(), f"Step {step} missing from 'x'."

        # make sure *no* extra steps were written
        for step in range(1, write_out_distance):
            assert str(step) not in f["x"].keys(), f"Unexpected step {step} written."

        # the final step must always be present
        assert "100" in f["x"].keys(), "Final step (100) missing."

        # the final value stored in the file must match the model output
        np.testing.assert_array_equal(
            rk_prob["rk_integration.x_final"], f["x"][str(100)][:]
        )


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10, 20, 30])
def test_time_attribute(
    mono_problem,
    integration_con,
    write_out_distance: int,
) -> None:
    """Validate that the ``time`` attribute of each stored step is correct."""
    rk_prob = _make_rk_problem(
        time_stage_problem=mono_problem,
        integration_con=integration_con,
        write_out_distance=write_out_distance,
        quantities=["x"],
    )
    rk_prob.setup()
    rk_prob.run_model()

    dt = integration_con.initial_delta_t
    t0 = integration_con.initial_time

    with h5py.File(WRITE_FILE, "r") as f:
        for step in range(0, 100, write_out_distance):
            _assert_time_attrs(f["x"][str(step)], step=step, dt=dt, t0=t0)


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10, 20, 30])
def test_multidisciplinary(
    multi_problem,
    integration_con,
    write_out_distance: int,
) -> None:
    """Write‑out for a multidisciplinary problem (quantities ``x`` and ``y``)."""
    rk_prob = _make_rk_problem(
        time_stage_problem=multi_problem,
        integration_con=integration_con,
        write_out_distance=write_out_distance,
        quantities=["x", "y"],
    )
    rk_prob.setup()
    rk_prob.run_model()

    with h5py.File(WRITE_FILE, "r") as f:
        for q in ("x", "y"):
            assert q in f, f"Missing group '{q}'."
            for step in range(0, 100, write_out_distance):
                assert (
                    str(step) in f[q].keys()
                ), f"Step {step} missing from group '{q}'."
            for step in range(1, write_out_distance):
                assert (
                    str(step) not in f[q].keys()
                ), f"Unexpected step {step} in group '{q}'."
            assert "100" in f[q].keys(), f"Final step missing from group '{q}'."

        # final values must match the model outputs
        np.testing.assert_array_equal(rk_prob["rk_integration.x_final"], f["x"]["100"])
        np.testing.assert_array_equal(rk_prob["rk_integration.y_final"], f["y"]["100"])


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", (1, 10))
def test_n_d_array(
    nd_array_prob,
    integration_con,
    write_out_distance: int,
) -> None:
    """Write‑out when the stored quantity has shape > 1 (2×2 array in this case)."""
    prob, shape = nd_array_prob

    rk_prob = _make_rk_problem(
        time_stage_problem=prob,
        integration_con=integration_con,
        write_out_distance=write_out_distance,
        quantities=["time_int"],
    )
    rk_prob.setup()
    # initialise the variable that the integrator will read
    rk_prob["time_int_initial"] = np.zeros(shape)
    rk_prob.run_model()

    with h5py.File(WRITE_FILE, "r") as f:
        assert "time_int" in f, "Missing top‑level group 'time_int'."
        for step in range(0, 100, write_out_distance):
            assert str(step) in f["time_int"].keys(), f"Step {step} missing."
        for step in range(1, write_out_distance):
            assert (
                str(step) not in f["time_int"].keys()
            ), f"Unexpected step {step} written."
        assert "100" in f["time_int"].keys(), "Final step missing."

        np.testing.assert_array_equal(
            rk_prob["rk_integration.time_int_final"],
            f["time_int"]["100"],
        )


@pytest.mark.mpi
@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10])
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_parallel_write_out(
    parallel_prob,
    integration_con,
    write_out_distance: int,
    shape: Tuple[int, ...],
) -> None:
    """
    Parallel write‑out test (requires an MPI‑enabled build of h5py).

    * Each rank supplies a different initial value.
    * The final dataset is read back on each rank and compared to the local view
      of the distributed variable stored in the OpenMDAO problem.
    """

    prob, _ = parallel_prob
    integration_con = StepTerminationIntegrationControl(0.01, 100, 1.0)

    # Build the RK problem that writes to a temporary file.
    # Use the core driver when available – it creates an in‑memory file on each rank.
    driver = "core" if hasattr(h5py.File, "driver") else None

    rk_prob = _make_rk_problem(
        time_stage_problem=prob,
        integration_con=integration_con,
        write_out_distance=write_out_distance,
        quantities=["time_int"],
    )
    time_initial = om.IndepVarComp()
    time_initial.add_output("time_int_initial", shape=shape, distributed=True)
    rk_prob.model.add_subsystem("time_initial", time_initial, promotes=["*"])

    rk_prob.setup()

    # Rank‑0 gets zeros, all other ranks get ones – this matches the original test.
    init_val = np.zeros(shape) if prob.comm.rank == 0 else np.ones(shape)
    rk_prob["time_int_initial"] = init_val

    rk_prob.run_model()

    read_kwargs = {}
    if driver == "core":
        read_kwargs["driver"] = "core"
        read_kwargs["backing_store"] = False

    with h5py.File(WRITE_FILE, "r", **read_kwargs) as f:
        assert "time_int" in f.keys(), "Missing group 'time_int' in parallel file."

        for step in range(0, 100, write_out_distance):
            assert str(step) in f["time_int"].keys(), f"Step {step} missing."

        for step in range(1, write_out_distance):
            assert str(step) not in f["time_int"].keys(), f"Unexpected step {step}."

        assert "100" in f["time_int"].keys(), "Final step missing."

        local_final = rk_prob.get_val(
            name="rk_integration.time_int_final", get_remote=False
        )
        slice_obj = slice(0, 2) if prob.comm.rank == 0 else slice(2, 4)
        h5_final = f["time_int"]["100"][slice_obj, ...]
        np.testing.assert_array_equal(local_final, h5_final)


# Tests for read_hdf_file


@pytest.mark.rk
def test_read_hdf5_file_monodisciplinary(mono_h5):
    quantities, solution = mono_h5
    time_dict, error_dict, result_dict = read_hdf5_file(
        WRITE_FILE, quantities, solution
    )
    with h5py.File(WRITE_FILE, "r") as f:
        # time attributes
        for step_str, dset in f["x"].items():
            step = int(step_str)
            assert time_dict[step] == dset.attrs["time"]
            # stored results in an array
            np.testing.assert_array_equal(result_dict["x"][step], dset[...])

        for step in result_dict["x"].keys():
            computed_error = np.abs(
                solution(time_dict[step], result_dict["x"][0], time_dict[0])
                - result_dict["x"][step]
            )
            assert error_dict["x"][step] == computed_error


@pytest.mark.mpi
@pytest.mark.rk
def test_read_hdf5_file_multidisciplinary_h5(multi_h5):
    quantities, solution = multi_h5
    time_dict, error_dict, result_dict = read_hdf5_file(
        WRITE_FILE, quantities, solution
    )
    with h5py.File(WRITE_FILE, "r") as f:
        for q in quantities:
            # time attributes
            for step_str, dset in f[q].items():
                step = int(step_str)
                assert time_dict[step] == dset.attrs["time"]
                # stored results in an array
                assert result_dict[q][step] == dset[...]

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


@pytest.mark.rk
def test_read_last_local_error_exact_step(mono_h5):
    """Write an ``error_measure`` group with several steps and ask for the
    error belonging to the step that exactly matches ``time_objective/step_size``."""
    err = read_last_local_error(WRITE_FILE)
    with h5py.File(WRITE_FILE, "r") as f:
        assert err == f["error_measure"]["100"][0]
