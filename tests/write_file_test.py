"""Tests the correct writing into hdf5-files."""

import pathlib
from typing import Union

import openmdao.api as om
import h5py
import pytest
import numpy as np

from rkopenmdao.integration_control import (
    StepTerminationIntegrationControl,
)
from rkopenmdao.butcher_tableaux import (
    embedded_third_order_four_stage_esdirk,
)
from rkopenmdao.file_writer import read_hdf5_file, read_last_local_error
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from .test_components import TestComp1, Testcomp51, Testcomp52


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10, 20, 30])
@pytest.mark.parametrize("write_file", ["file.h5", "other_file.h5"])
def test_monodisciplinary(write_out_distance, write_file):
    """Tests write-out for monodisciplinary problems."""
    test_prob = om.Problem()
    integration_control = StepTerminationIntegrationControl(0.01, 100, 1.0)

    butcher_tableau = embedded_third_order_four_stage_esdirk

    test_prob.model.add_subsystem(
        "test_comp", TestComp1(integration_control=integration_control)
    )

    time_int_prob = om.Problem()
    time_int_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=test_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_out_distance=write_out_distance,
            write_file=write_file,
            time_integration_quantities=["x"],
        ),
    )

    time_int_prob.setup()
    time_int_prob.run_model()

    with h5py.File(write_file) as f:
        assert "x" in f.keys()
        for i in range(0, 100, write_out_distance):
            assert str(i) in f["x"].keys()
        for i in range(1, write_out_distance):
            assert str(i) not in f["x"].keys()
        assert str(100) in f["x"].keys()
        assert time_int_prob["rk_integration.x_final"] == f["x"][str(100)][:]


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10, 20, 30])
@pytest.mark.parametrize("write_file", ["file.h5", "other_file.h5"])
def test_time_attribute(write_out_distance, write_file):
    """Checks that the time attribute in the written out file is correct."""
    test_prob = om.Problem()
    integration_control = StepTerminationIntegrationControl(0.01, 100, 1.0)

    butcher_tableau = embedded_third_order_four_stage_esdirk

    test_prob.model.add_subsystem(
        "test_comp", TestComp1(integration_control=integration_control)
    )

    time_int_prob = om.Problem()
    time_int_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=test_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_out_distance=write_out_distance,
            write_file=write_file,
            time_integration_quantities=["x"],
        ),
    )

    time_int_prob.setup()
    time_int_prob.run_model()

    with h5py.File(write_file) as f:
        for i in range(0, 100, write_out_distance):
            assert f["x"][str(i)].attrs["time"] == pytest.approx(i * 0.01 + 1.0)
        assert f["x"][str(100)].attrs["time"] == pytest.approx(2.0)


@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10, 20, 30])
@pytest.mark.parametrize("write_file", ["file.h5", "other_file.h5"])
def test_multidisciplinary(write_out_distance, write_file):
    """Tests write-out for multidisciplinary problems."""
    test_prob = om.Problem()
    integration_control = StepTerminationIntegrationControl(0.01, 100, 1.0)

    butcher_tableau = embedded_third_order_four_stage_esdirk

    test_prob.model.add_subsystem(
        "test_comp_1",
        Testcomp51(integration_control=integration_control),
        promotes=["*"],
    )
    test_prob.model.add_subsystem(
        "test_comp_2",
        Testcomp52(integration_control=integration_control),
        promotes=["*"],
    )

    time_int_prob = om.Problem()
    time_int_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=test_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_out_distance=write_out_distance,
            write_file=write_file,
            time_integration_quantities=["x", "y"],
        ),
    )

    time_int_prob.setup()
    time_int_prob.run_model()

    with h5py.File(write_file) as f:
        assert "x" in f.keys()
        assert "y" in f.keys()
        for i in range(0, 100, write_out_distance):
            assert str(i) in f["x"].keys()
            assert str(i) in f["y"].keys()
        for i in range(1, write_out_distance):
            assert str(i) not in f["x"].keys()
            assert str(i) not in f["y"].keys()
        assert str(100) in f["x"].keys()
        assert str(100) in f["y"].keys()
        assert time_int_prob["rk_integration.x_final"] == f["x"][str(100)]
        assert time_int_prob["rk_integration.y_final"] == f["y"][str(100)]


@pytest.mark.rk
@pytest.mark.parametrize(
    "write_out_distance",
    (1, 10),
)
def test_n_d_array(write_out_distance):
    """Tests write-out when shape isn't just 1D."""
    test_prob = om.Problem()
    integration_control = StepTerminationIntegrationControl(0.01, 100, 1.0)

    butcher_tableau = embedded_third_order_four_stage_esdirk

    time_int_indep = om.IndepVarComp()
    time_int_indep.add_output(
        "time_int_test_output",
        shape=(2, 2),
        val=0.0,
        tags=["stage_output_var", "time_int"],
    )
    test_prob.model.add_subsystem("time_int_indep", time_int_indep)
    time_int_prob = om.Problem()
    time_int_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=test_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_out_distance=write_out_distance,
            write_file="file.h5",
            time_integration_quantities=["time_int"],
        ),
        promotes=["*"],
    )
    time_int_prob.setup()
    time_int_prob["time_int_initial"] = np.zeros((2, 2))
    time_int_prob.run_model()

    with h5py.File("file.h5") as f:
        assert "time_int" in f.keys()
        for i in range(0, 100, write_out_distance):
            assert str(i) in f["time_int"].keys()
        for i in range(1, write_out_distance):
            assert str(i) not in f["time_int"].keys()
        assert str(100) in f["time_int"].keys()
        assert np.array_equal(
            time_int_prob["rk_integration.time_int_final"], f["time_int"][str(100)]
        )


@pytest.mark.mpi
@pytest.mark.rk
@pytest.mark.parametrize("write_out_distance", [1, 10])
@pytest.mark.parametrize("shape", [(2,), (2, 2), (2, 2, 2)])
def test_parallel_write_out(write_out_distance, shape):
    """Tests write-out with parallel execution. (Needs h5py that has MPI support.)"""
    test_prob = om.Problem()
    integration_control = StepTerminationIntegrationControl(0.01, 100, 1.0)

    butcher_tableau = embedded_third_order_four_stage_esdirk

    time_int_indep = om.IndepVarComp()
    time_int_indep.add_output(
        "time_int_test_output",
        distributed=True,
        shape=shape,
        val=0.0,
        tags=["stage_output_var", "time_int"],
    )
    test_prob.model.add_subsystem("time_int_indep", time_int_indep)
    time_int_prob = om.Problem()
    time_int_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=test_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_out_distance=write_out_distance,
            write_file="file.h5",
            time_integration_quantities=["time_int"],
        ),
        promotes=["*"],
    )
    time_prob_indep = om.IndepVarComp()
    time_prob_indep.add_output("time_int_initial", shape=shape, distributed=True)
    time_int_prob.model.add_subsystem(
        "time_prob_indep",
        time_prob_indep,
        promotes=["*"],
    )
    time_int_prob.setup()
    time_int_prob["time_int_initial"] = (
        np.zeros(shape) if test_prob.comm.rank == 0 else np.ones(shape)
    )
    time_int_prob.run_model()

    with h5py.File("file.h5", mode="r") as f:
        assert "time_int" in f.keys()
        for i in range(0, 100, write_out_distance):
            assert str(i) in f["time_int"].keys()
        for i in range(1, write_out_distance):
            assert str(i) not in f["time_int"].keys()
        assert str(100) in f["time_int"].keys()
        assert np.array_equal(
            time_int_prob.get_val(
                name="rk_integration.time_int_final", get_remote=False
            )[:],
            f["time_int"][str(100)][
                slice(0, 2) if test_prob.comm.rank == 0 else slice(2, 4), ...
            ],
        )


def _build_test_h5(
    tmp_path: pathlib.Path,
    *,
    quantities: list[tuple[str, np.ndarray]],
    steps: list[int],
    error_vals: Union[dict[int, float], None] = None,
) -> pathlib.Path:
    """Create an HDF5 file and return its path.

    * ``quantities``  – list of ``(group_name, full_array)`` pairs.
    * ``steps``       – list of integer steps that will become dataset names.
    * ``error_vals``  – optional ``step → error`` mapping written to the
      ``error_measure`` group.
    """
    file_path = tmp_path / "test_file.h5"
    with h5py.File(file_path, mode="w") as f:
        # ---- quantities -------------------------------------------------
        for name, full_array in quantities:
            grp = f.create_group(name)
            for step in steps:
                ds = grp.create_dataset(str(step), data=full_array, dtype=np.float64)
                ds.attrs["time"] = step * 0.1  # deterministic time stamp

        # ---- optional error_measure ------------------------------------
        if error_vals is not None:
            err_grp = f.create_group("error_measure")
            for step, val in error_vals.items():
                ds = err_grp.create_dataset(str(step), shape=(1,), dtype=np.float64)
                ds.attrs["time"] = step * 0.1
                ds[0] = val
    return file_path


# Fixtures


@pytest.fixture
def single_quantity_data():
    """2x3 matrix used by the “single‑quantity” test."""
    return np.arange(6, dtype=np.float64).reshape(2, 3)


@pytest.fixture
def multi_quantity_data():
    """Two 1-D vectors (length 4) used by the “multiple-quantities” test."""
    q1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    q2 = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    return q1, q2


@pytest.mark.unit
def test_read_hdf5_file_single_quantity(
    tmp_path: pathlib.Path,
    single_quantity_data: np.ndarray,
):
    """A file with one quantity; the analytical solution returns the same data,
    therefore the computed error dictionary should contain only zeros."""
    # Build file with two steps (0 and 1)
    file_path = _build_test_h5(
        tmp_path,
        quantities=[("my_quantity", single_quantity_data)],
        steps=[0, 1],
        error_vals={0: 0.123, 1: 0.456},
    )

    # Analytical solution: ignores the time argument and returns the exact matrix
    def analytic_solution(_time: float) -> np.ndarray:
        return single_quantity_data

    time, error, result = read_hdf5_file(
        file=str(file_path),
        quantities=["my_quantity"],
        solution=analytic_solution,
    )

    # ---- expectations -------------------------------------------------
    assert time == {0: 0.0, 1: 0.1}
    np.testing.assert_array_equal(result["my_quantity"][0], single_quantity_data)
    np.testing.assert_array_equal(result["my_quantity"][1], single_quantity_data)
    np.testing.assert_array_almost_equal(error["my_quantity"][0], 0.0)
    np.testing.assert_array_almost_equal(error["my_quantity"][1], 0.0)


@pytest.mark.unit
def test_read_hdf5_file_multiple_quantities(
    tmp_path: pathlib.Path,
    multi_quantity_data: tuple[np.ndarray, np.ndarray],
):
    """Two quantities are stored; the analytical solution returns a tuple
    (one array per quantity) that is offset by a constant.  The test checks
    that the absolute error is computed correctly for each quantity."""
    q1_data, q2_data = multi_quantity_data

    file_path = _build_test_h5(
        tmp_path,
        quantities=[("q_one", q1_data), ("q_two", q2_data)],
        steps=[0, 1],
        error_vals={0: 0.9, 1: 1.1},
    )

    # Analytical solution – adds +0.5 to q_one and –0.5 to q_two
    def analytic_solution(_time: float) -> tuple[np.ndarray, np.ndarray]:
        return q1_data + 0.5, q2_data - 0.5

    time, error, result = read_hdf5_file(
        file=str(file_path),
        quantities=["q_one", "q_two"],
        solution=analytic_solution,
    )

    # ---- expectations -------------------------------------------------
    assert time == {0: 0.0, 1: 0.1}
    np.testing.assert_array_equal(result["q_one"][0], q1_data)
    np.testing.assert_array_equal(result["q_two"][0], q2_data)
    np.testing.assert_array_equal(result["q_one"][1], q1_data)
    np.testing.assert_array_equal(result["q_two"][1], q2_data)

    # Expected absolute error = 0.5 for both quantities, both steps
    for step in (0, 1):
        np.testing.assert_allclose(error["q_one"][step], 0.5)
        np.testing.assert_allclose(error["q_two"][step], 0.5)


@pytest.mark.unit
def test_read_last_local_error_exact_step(tmp_path: pathlib.Path):
    """Write an ``error_measure`` group with several steps and ask for the
    error belonging to the step that exactly matches ``time_objective/step_size``."""
    error_vals = {i: i * 0.1 for i in range(5)}  # 0.0, 0.1, …, 0.4
    file_path = _build_test_h5(
        tmp_path,
        quantities=[],  # no quantity groups needed for this test
        steps=[],
        error_vals=error_vals,
    )

    time_objective = 0.6  # 0.6 / 0.2 = 3  → step 3
    step_size = 0.2

    result = read_last_local_error(
        file_path=str(file_path),
        time_objective=time_objective,
        step_size=step_size,
    )
    assert result == pytest.approx(0.3)  # error for step 3


@pytest.mark.unit
def test_read_last_local_error_truncated_step(tmp_path: pathlib.Path):
    """When ``time_objective/step_size`` is not an integer the function truncates
    the value (via ``int(...)``).  Verify that the truncated step is used."""
    error_vals = {i: float(i) for i in range(10)}  # 0,1,2,…,9
    file_path = _build_test_h5(
        tmp_path,
        quantities=[],
        steps=[],
        error_vals=error_vals,
    )

    # 0.95 / 0.2 = 4.75 → int() == 4
    time_objective = 0.95
    step_size = 0.2

    result = read_last_local_error(
        file_path=str(file_path),
        time_objective=time_objective,
        step_size=step_size,
    )
    assert result == pytest.approx(4.0)  # step 4 stores error value 4.0


@pytest.mark.unit
def test_read_last_local_error_missing_step_raises(tmp_path: pathlib.Path):
    """If the calculated step does not exist in the HDF5 file a ``KeyError`` is
    raised (the usual h5py behavior).  The test ensures the exception is
    indeed propagated."""
    error_vals = {0: 0.0, 1: 0.1}
    file_path = _build_test_h5(
        tmp_path,
        quantities=[],
        steps=[],
        error_vals=error_vals,
    )

    # Ask for step 5 (which is not present)
    time_objective = 5 * 0.3
    step_size = 0.3

    with pytest.raises(KeyError):
        read_last_local_error(
            file_path=str(file_path),
            time_objective=time_objective,
            step_size=step_size,
        )
