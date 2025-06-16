import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals
import numpy as np
import pytest

from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk,
)
from rkopenmdao.checkpoint_interface.no_checkpointer import NoCheckpointer
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_estimator import (
    SimpleErrorEstimator,
    _non_mpi_partial_norm,
    _mpi_partial_norm,
)
from rkopenmdao.integration_control import (
    IntegrationControl,
    StepTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from mpi4py import MPI
import h5py

from .distributed_variables_test import (
    Test2Component1 as TestComponent1,
    Test2Component2 as TestComponent2,
    setup_parallel_two_distributed_problem_and_integration_control,
)


def setup_time_integration_problem(
    stage_prob,
    butcher_tableau,
    integration_control,
    time_integration_quantities,
    initial_values=None,
    adaptive_time_stepping=False,
    exclude: list = None,
    name: str = "None",
):
    """Sets up the time integration problem for the following test."""
    if exclude is None:
        exclude = []
    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=stage_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        time_integration_quantities=time_integration_quantities,
        adaptive_time_stepping=adaptive_time_stepping,
        error_controller=[pseudo],
        error_estimator_options={"exclude": exclude},
        write_file=f"{name}.h5",
        write_out_distance=1,
    )

    time_test_prob = om.Problem()
    time_test_prob.model.add_subsystem("rk_integrator", rk_integrator, promotes=["*"])
    time_ivc = om.IndepVarComp()
    for var in time_integration_quantities:
        time_ivc.add_output(f"{var}_initial", shape=1, distributed=True)
    time_test_prob.model.add_subsystem("time_ivc", time_ivc, promotes=["*"])
    time_test_prob.setup()
    if initial_values is not None:
        for var in time_integration_quantities:
            time_test_prob[f"{var}_initial"][:] = initial_values[var][
                time_test_prob.comm.rank
            ]

    return time_test_prob


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize("butcher_tableau", [embedded_second_order_two_stage_sdirk])
@pytest.mark.parametrize(
    "initial_values", [{"x12": [1, 1], "x43": [1, 1]}, {"x12": [0, 0], "x43": [1, 1]}]
)
def test_parallel_two_distributed_time_integration(
    num_steps, butcher_tableau, initial_values
):
    """Tests time integration with distributed variables for a problem with two
    components."""
    delta_t = 0.1
    test_prob, integration_control = (
        setup_parallel_two_distributed_problem_and_integration_control(
            delta_t, num_steps, butcher_tableau
        )
    )
    time_test_prob = setup_time_integration_problem(
        test_prob,
        butcher_tableau,
        integration_control,
        ["x12", "x43"],
        initial_values,
        adaptive_time_stepping=True,
        name="nonexcluded",
    )
    time_test_prob.run_model()
    time_test_prob2 = setup_time_integration_problem(
        test_prob,
        butcher_tableau,
        integration_control,
        ["x12", "x43"],
        initial_values,
        adaptive_time_stepping=True,
        exclude=["x12"],
        name="x12excluded",
    )
    time_test_prob2.run_model()
    time_test_prob3 = setup_time_integration_problem(
        test_prob,
        butcher_tableau,
        integration_control,
        ["x12", "x43"],
        initial_values,
        adaptive_time_stepping=True,
        exclude=["x43"],
        name="x43excluded",
    )
    time_test_prob3.run_model()

    norm_dict = {}
    with h5py.File("nonexcluded.h5", mode="r") as f:
        group = f["Norm"]
        for key in group.keys():
            norm_dict.update({int(key): group[key][0]})
        norm_dict = dict(sorted(norm_dict.items()))

    norm_dictb = {}
    with h5py.File("x12excluded.h5", mode="r") as f:
        group = f["Norm"]
        for key in group.keys():
            norm_dictb.update({int(key): group[key][0]})
        norm_dictb = dict(sorted(norm_dictb.items()))
    norm_dictc = {}
    with h5py.File("x43excluded.h5", mode="r") as f:
        group = f["Norm"]
        for key in group.keys():
            norm_dictc.update({int(key): group[key][0]})
        norm_dictc = dict(sorted(norm_dictc.items()))

    for idx, norm in enumerate(norm_dict.values()):
        if idx != 0:
            assert (norm_dictb[idx] ** 2 + norm_dictc[idx] ** 2) ** (
                1 / 2
            ) == pytest.approx(norm, rel=1e-4)
