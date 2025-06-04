import openmdao.api as om
import os
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

count=0
class TestComponent1(om.ImplicitComponent):
    """Models the first two equations from above, with the first being on rank 0 and the
    second on rank 1"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x43", shape=1, distributed=True)
        self.add_input(
            "x12_old", shape=1, distributed=True, tags=["x12", "step_input_var"]
        )
        self.add_input(
            "s12_i", shape=1, distributed=True, tags=["x12", "accumulated_stage_var"]
        )
        self.add_output(
            "k12_i", shape=1, distributed=True, tags=["x12", "stage_output_var"]
        )
        self.add_output("x12", shape=1, distributed=True)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        residuals["x12"] = (
            inputs["x12_old"]
            + delta_t * (inputs["s12_i"] + butcher_diagonal_element * outputs["k12_i"])
            - outputs["x12"]
        )
        exch_data = np.zeros(1)
        if self.comm.rank == 0:
            exch_data[0] = outputs["x12"]
            self.comm.Send(exch_data, dest=1, tag=0)
            residuals["k12_i"] = inputs["x43"] - outputs["k12_i"]
        elif self.comm.rank == 1:
            self.comm.Recv(exch_data, source=0, tag=0)
            residuals["k12_i"] = exch_data[0] - outputs["k12_i"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        if mode == "fwd":
            d_residuals["x12"] += (
                d_inputs["x12_old"]
                + delta_t
                * (d_inputs["s12_i"] + butcher_diagonal_element * d_outputs["k12_i"])
                - d_outputs["x12"]
            )
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                exch_data[0] = d_outputs["x12"]
                self.comm.Send(exch_data, dest=1, tag=0)
                d_residuals["k12_i"] += d_inputs["x43"] - d_outputs["k12_i"]

            elif self.comm.rank == 1:
                self.comm.Recv(exch_data, source=0, tag=0)
                d_residuals["k12_i"] += exch_data[0] - d_outputs["k12_i"]
        # but they seem wrong somehow?
        elif mode == "rev":
            d_inputs["x12_old"] += d_residuals["x12"]
            d_inputs["s12_i"] += delta_t * d_residuals["x12"]
            d_outputs["k12_i"] += (
                delta_t * butcher_diagonal_element * d_residuals["x12"]
            )
            d_outputs["x12"] -= d_residuals["x12"]
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Recv(exch_data, source=1, tag=1)
                d_outputs["x12"] += exch_data[0]
                d_inputs["x43"] += d_residuals["k12_i"]
                d_outputs["k12_i"] -= d_residuals["k12_i"]
            elif self.comm.rank == 1:
                exch_data[0] = d_residuals["k12_i"]
                self.comm.Send(exch_data, dest=0, tag=1)
                d_outputs["k12_i"] -= d_residuals["k12_i"]


class TestComponent2(om.ImplicitComponent):
    """Models the last two equations from above, with the first being on rank 1 and the
    second on rank 0"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x12", shape=1, distributed=True)
        self.add_input(
            "x43_old", shape=1, distributed=True, tags=["x43", "step_input_var"]
        )
        self.add_input(
            "s43_i", shape=1, distributed=True, tags=["x43", "accumulated_stage_var"]
        )
        self.add_output(
            "k43_i", shape=1, distributed=True, tags=["x43", "stage_output_var"]
        )
        self.add_output("x43", shape=1, distributed=True)

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        residuals["x43"] = (
            inputs["x43_old"]
            + delta_t * (inputs["s43_i"] + butcher_diagonal_element * outputs["k43_i"])
            - outputs["x43"]
        )
        exch_data = np.zeros(1)
        if self.comm.rank == 0:
            self.comm.Recv(exch_data, source=1, tag=3)
            residuals["k43_i"] = exch_data[0] - outputs["k43_i"]
        elif self.comm.rank == 1:
            exch_data[0] = outputs["x43"]
            self.comm.Send(exch_data, dest=0, tag=3)
            residuals["k43_i"] = inputs["x12"] - outputs["k43_i"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        if mode == "fwd":
            d_residuals["x43"] += (
                d_inputs["x43_old"]
                + delta_t
                * (d_inputs["s43_i"] + butcher_diagonal_element * d_outputs["k43_i"])
                - d_outputs["x43"]
            )
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Recv(exch_data, source=1, tag=3)
                d_residuals["k43_i"] += exch_data[0] - d_outputs["k43_i"]
            elif self.comm.rank == 1:
                exch_data[0] = d_outputs["x43"]
                self.comm.Send(exch_data, dest=0, tag=3)
                d_residuals["k43_i"] += d_inputs["x12"] - d_outputs["k43_i"]
        # but they seem wrong somehow?
        elif mode == "rev":
            d_inputs["x43_old"] += d_residuals["x43"]
            d_inputs["s43_i"] += delta_t * d_residuals["x43"]
            d_outputs["k43_i"] += (
                delta_t * butcher_diagonal_element * d_residuals["x43"]
            )
            d_outputs["x43"] -= d_residuals["x43"]
            exch_data = np.zeros(1)

            if self.comm.rank == 0:
                exch_data[0] = d_residuals["k43_i"]
                self.comm.Send(exch_data, dest=1, tag=2)

                d_outputs["k43_i"] -= d_residuals["k43_i"]
            elif self.comm.rank == 1:
                self.comm.Recv(exch_data, source=0, tag=2)
                d_outputs["x43"] += exch_data[0]
                d_inputs["x12"] += d_residuals["k43_i"]
                d_outputs["k43_i"] -= d_residuals["k43_i"]


def ode_2_analytical_solution(time, initial_values):
    """Analytical solution to the above system of ODEs modelled by the two components"""
    a = np.sum(initial_values) / 4
    b = (np.sum(initial_values[0:3:2]) - np.sum(initial_values[1:4:2])) / 4
    c = (initial_values[3] - initial_values[1]) / 2
    d = (initial_values[0] - initial_values[2]) / 2
    return np.array(
        [
            a * np.exp(time) + b * np.exp(-time) + c * np.sin(time) + d * np.cos(time),
            a * np.exp(time) - b * np.exp(-time) - c * np.cos(time) + d * np.sin(time),
            a * np.exp(time) + b * np.exp(-time) - c * np.sin(time) - d * np.cos(time),
            a * np.exp(time) - b * np.exp(-time) + c * np.cos(time) - d * np.sin(time),
        ]
    )



def setup_parallel_two_distributed_problem_and_integration_control(
    delta_t, num_steps, butcher_tableau, setup_mode="auto"
):
    """Sets up the stage problem for the two component test cases."""
    integration_control = StepTerminationIntegrationControl(delta_t, num_steps, 0.0)
    integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[
        -1, -1
    ]
    test_comp1 = TestComponent1(integration_control=integration_control)
    test_comp2 = TestComponent2(integration_control=integration_control)
    test_prob = om.Problem()
    test_prob.model.add_subsystem("test_comp1", test_comp1, promotes=["*"])
    test_prob.model.add_subsystem("test_comp2", test_comp2, promotes=["*"])

    ivc = om.IndepVarComp()
    ivc.add_output("x12_old", shape=1, distributed=True)
    ivc.add_output("s12_i", shape=1, distributed=True)
    ivc.add_output("x43_old", shape=1, distributed=True)
    ivc.add_output("s43_i", shape=1, distributed=True)

    test_prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    test_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=False, iprint=-1
    )
    test_prob.model.linear_solver = om.PETScKrylov(atol=1e-12, rtol=1e-12, iprint=-1)
    test_prob.setup(mode=setup_mode)
    return test_prob, integration_control

def setup_time_integration_problem(
    stage_prob,
    butcher_tableau,
    integration_control,
    time_integration_quantities,
    initial_values=None,
    #setup_mode="auto",
    adaptive_time_stepping=False,
    exclude:list=[],
    name:str="None"
):
    """Sets up the time integration problem for the following test."""
    print(time_integration_quantities)
    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=stage_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        time_integration_quantities=time_integration_quantities,
        adaptive_time_stepping=adaptive_time_stepping,
        error_controller=[pseudo],
        error_estimator_options={'exclude': exclude},
        write_file=f"{name}.h5",
        write_out_distance=1,
    )

    time_test_prob = om.Problem()
    time_test_prob.model.add_subsystem("rk_integrator", rk_integrator, promotes=["*"])
    time_ivc = om.IndepVarComp()
    for var in time_integration_quantities:
        time_ivc.add_output(f"{var}_initial", shape=1, distributed=True)
    time_test_prob.model.add_subsystem("time_ivc", time_ivc, promotes=["*"])
    # these solvers are only necessary so that check_partials actually produces results
    #time_test_prob.model.nonlinear_solver = om.NonlinearBlockJac(iprint=-1)
    #time_test_prob.model.linear_solver = om.LinearBlockJac(iprint=-1)
    time_test_prob.setup()
    if initial_values is not None:
        for var in time_integration_quantities:
            time_test_prob[f"{var}_initial"][:] = initial_values[var][
                time_test_prob.comm.rank
            ]

    return time_test_prob



@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1,10])
@pytest.mark.parametrize(
    "butcher_tableau", [embedded_second_order_two_stage_sdirk]
)
@pytest.mark.parametrize(
    "initial_values", [{"x12": [1, 1], "x43": [1, 1]},{"x12": [0, 0], "x43": [1,1]}]
)
def test_parallel_two_distributed_time_integration(
    num_steps, butcher_tableau, initial_values
):
    """Tests time integration with distributed variables for a problem with two
    components."""
    global count
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
            name='nonexcluded'
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
        name='x12excluded'
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
        name='x43excluded'
    )
    time_test_prob3.run_model()
    
    count+=1

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

    for idx,norm in enumerate(norm_dict.values()):
        if idx != 0:
            assert(np.abs(norm - (norm_dictb[idx]**2 + norm_dictc[idx]**2)**(1/2))<1e-7)
   