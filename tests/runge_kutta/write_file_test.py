import openmdao.api as om
import h5py
import pytest

from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.butcher_tableaux import (
    third_order_four_stage_esdirk,
)
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import RungeKuttaIntegrator
from .test_components import TestComp1, TestComp5_1, TestComp5_2
from .test_postprocessing_problems import SquaringComponent


@pytest.mark.rk
@pytest.mark.parametrize(
    "write_out_distance, write_file",
    (
        [1, "file.h5"],
        [10, "file.h5"],
        [20, "file.h5"],
        [30, "file.h5"],
        [1, "other_file.h5"],
        [10, "other_file.h5"],
        [20, "other_file.h5"],
        [30, "other_file.h5"],
    ),
)
def test_monodisciplinary(write_out_distance, write_file):
    test_prob = om.Problem()
    integration_control = IntegrationControl(1.0, 100, 0.01)

    butcher_tableau = third_order_four_stage_esdirk

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
        assert time_int_prob["rk_integration.x_final"] == f["x"][str(100)]


@pytest.mark.rk
@pytest.mark.parametrize(
    "write_out_distance, write_file",
    (
        [1, "file.h5"],
        [10, "file.h5"],
        [20, "file.h5"],
        [30, "file.h5"],
        [1, "other_file.h5"],
        [10, "other_file.h5"],
        [20, "other_file.h5"],
        [30, "other_file.h5"],
    ),
)
def test_multidisciplinary(write_out_distance, write_file):
    test_prob = om.Problem()
    integration_control = IntegrationControl(1.0, 100, 0.01)

    butcher_tableau = third_order_four_stage_esdirk

    test_prob.model.add_subsystem(
        "test_comp_1",
        TestComp5_1(integration_control=integration_control),
        promotes=["*"],
    )
    test_prob.model.add_subsystem(
        "test_comp_2",
        TestComp5_2(integration_control=integration_control),
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
    "write_out_distance, write_file",
    (
        [1, "file.h5"],
        [10, "file.h5"],
        [20, "file.h5"],
        [30, "file.h5"],
        [1, "other_file.h5"],
        [10, "other_file.h5"],
        [20, "other_file.h5"],
        [30, "other_file.h5"],
    ),
)
def test_postprocessing(write_out_distance, write_file):
    test_prob = om.Problem()
    integration_control = IntegrationControl(1.0, 100, 0.01)

    butcher_tableau = third_order_four_stage_esdirk

    test_prob.model.add_subsystem(
        "test_comp", TestComp1(integration_control=integration_control)
    )

    postproc_problem = om.Problem()
    postproc_problem.model.add_subsystem(
        "squarer", SquaringComponent(quantity_list=[("x", 1)])
    )

    time_int_prob = om.Problem()
    time_int_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=test_prob,
            postprocessing_problem=postproc_problem,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_out_distance=write_out_distance,
            write_file=write_file,
            time_integration_quantities=["x"],
            postprocessing_quantities=["squared_x"],
        ),
    )

    time_int_prob.setup()
    time_int_prob.run_model()

    with h5py.File(write_file) as f:
        assert "x" in f.keys()
        assert "squared_x" in f.keys()
        for i in range(0, 100, write_out_distance):
            assert str(i) in f["x"].keys()
            assert str(i) in f["squared_x"].keys()
        for i in range(1, write_out_distance):
            assert str(i) not in f["x"].keys()
            assert str(i) not in f["squared_x"].keys()
        assert str(100) in f["x"].keys()
        assert str(100) in f["squared_x"].keys()
        assert time_int_prob["rk_integration.x_final"] == f["x"][str(100)]
        assert (
            time_int_prob["rk_integration.squared_x_final"] == f["squared_x"][str(100)]
        )
