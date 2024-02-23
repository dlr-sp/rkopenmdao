import itertools

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals
import numpy as np
import pytest
from mpi4py import MPI

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    third_order_four_stage_esdirk,
    implicit_euler,
    explicit_euler,
)


class Test1Component1(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x_old", shape=1, distributed=True, tags=["x", "step_input_var"])
        self.add_input(
            "s_i", shape=1, distributed=True, tags=["x", "accumulated_stage_var"]
        )
        self.add_output(
            "k_i", shape=1, distributed=True, tags=["x", "stage_output_var"]
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        send_data = np.zeros(1)
        send_data += inputs["x_old"] + delta_t * (
            inputs["s_i"] + butcher_diagonal_element * outputs["k_i"]
        )
        recv_data = np.zeros(1)

        if self.comm.rank == 0:
            self.comm.Send(send_data, dest=1, tag=0)
            self.comm.Recv(recv_data, source=1, tag=1)
            residuals["k_i"] = outputs["k_i"] - recv_data
        elif self.comm.rank == 1:
            self.comm.Recv(recv_data, source=0, tag=0)
            self.comm.Send(send_data, dest=0, tag=1)
            residuals["k_i"] = outputs["k_i"] + recv_data

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        if mode == "fwd":
            send_data = d_inputs["x_old"] + delta_t * (
                d_inputs["s_i"] + butcher_diagonal_element * d_outputs["k_i"]
            )
            recv_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Send(send_data, dest=1, tag=0)
                self.comm.Recv(recv_data, source=1, tag=1)
                d_residuals["k_i"] += d_outputs["k_i"] - recv_data
            elif self.comm.rank == 1:
                self.comm.Recv(recv_data, source=0, tag=0)
                self.comm.Send(send_data, dest=0, tag=1)
                d_residuals["k_i"] += d_outputs["k_i"] + recv_data
        elif mode == "rev":
            send_data = d_residuals["k_i"]
            recv_data = np.zeros(1)
            # TODO: Are the += and -= correct? According to check_totals they are, but they seem wrong somehow?
            if self.comm.rank == 0:
                self.comm.Send(send_data, dest=1, tag=0)
                self.comm.Recv(recv_data, source=1, tag=1)
                d_inputs["x_old"] += recv_data
                d_inputs["s_i"] += delta_t * recv_data
                d_outputs["k_i"] += delta_t * butcher_diagonal_element * recv_data
                d_outputs["k_i"] += d_residuals["k_i"]
            elif self.comm.rank == 1:
                self.comm.Recv(recv_data, source=0, tag=0)
                self.comm.Send(send_data, dest=0, tag=1)
                d_inputs["x_old"] -= recv_data
                d_inputs["s_i"] -= delta_t * recv_data
                d_outputs["k_i"] -= delta_t * butcher_diagonal_element * recv_data
                d_outputs["k_i"] += d_residuals["k_i"]


class Test2Component1(om.ImplicitComponent):
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
        # TODO: Are the Sends and Recvs at the right place? According to check_totals they are, but they seem wrong somehow?
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


class Test2Component2(om.ImplicitComponent):
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
        # TODO: Are the Sends and Recvs at the right place? According to check_totals they are, but they seem wrong somehow?
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


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [explicit_euler, implicit_euler, third_order_four_stage_esdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
def test_parallel_single_distributed_partial(
    num_steps, butcher_tableau, test_direction
):
    integration_control = IntegrationControl(0.0, num_steps, 0.1)
    integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[
        -1, -1
    ]
    test_comp = Test1Component1(integration_control=integration_control)

    test_prob = om.Problem()
    test_prob.model.add_subsystem("test_comp", test_comp, promotes=["*"])

    ivc = om.IndepVarComp()
    ivc.add_output("x_old", shape=1, distributed=True)
    ivc.add_output("s_i", shape=1, distributed=True)

    test_prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    test_prob.model.nonlinear_solver = om.NewtonSolver(
        iprint=-1, solve_subsystems=False
    )
    test_prob.model.linear_solver = om.PETScKrylov(iprint=-1, atol=1e-12, rtol=1e-12)
    test_prob.setup(mode=test_direction)

    test_prob.run_model()

    if test_prob.comm.rank > 0:
        data = test_prob.check_totals(
            of=["k_i"],
            wrt=["x_old", "s_i"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
            out_stream=None,
        )
    else:
        data = test_prob.check_totals(
            of=["k_i"], wrt=["x_old", "s_i"], abs_err_tol=1e-4, rel_err_tol=1e-4
        )
    assert_check_totals(data, atol=1e-4, rtol=1e-4)

    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=test_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        time_integration_quantities=["x"],
    )

    time_test_prob = om.Problem()
    time_test_prob.model.add_subsystem("rk_integrator", rk_integrator, promotes=["*"])
    time_ivc = om.IndepVarComp()
    time_ivc.add_output("x_initial", shape=1, distributed=True)
    time_test_prob.model.add_subsystem("time_ivc", time_ivc, promotes=["*"])
    # these solvers are only necessary so that check_partials actually produces results
    time_test_prob.model.nonlinear_solver = om.NonlinearBlockJac(iprint=-1)
    time_test_prob.model.linear_solver = om.LinearBlockJac(iprint=-1)
    time_test_prob.setup(mode=test_direction)
    time_test_prob.run_model()

    if test_prob.comm.rank > 0:
        data = time_test_prob.check_totals(
            of=["x_final"],
            wrt=["x_initial"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
            out_stream=None,
        )
    else:
        data = time_test_prob.check_totals(
            of=["x_final"], wrt=["x_initial"], abs_err_tol=1e-4, rel_err_tol=1e-4
        )
    assert_check_totals(data, atol=1e-4, rtol=1e-4)


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [explicit_euler, implicit_euler, third_order_four_stage_esdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
def test_parallel_two_distributed_partial(num_steps, butcher_tableau, test_direction):
    integration_control = IntegrationControl(0.0, num_steps, 0.1)
    integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[
        -1, -1
    ]
    test_comp1 = Test2Component1(integration_control=integration_control)
    test_comp2 = Test2Component2(integration_control=integration_control)
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
    test_prob.setup(mode=test_direction)

    test_prob.run_model()

    if test_prob.comm.rank > 0:
        data = test_prob.check_totals(
            of=["k12_i", "x12", "k43_i", "x43"],
            wrt=["x12_old", "s12_i", "x43_old", "s43_i"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
            out_stream=None,
        )
    else:
        data = test_prob.check_totals(
            of=["k12_i", "x12", "k43_i", "x43"],
            wrt=["x12_old", "s12_i", "x43_old", "s43_i"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
        )
    assert_check_totals(data, atol=1e-4, rtol=1e-4)

    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=test_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        time_integration_quantities=["x12", "x43"],
    )

    time_test_prob = om.Problem()
    time_test_prob.model.add_subsystem("rk_integrator", rk_integrator, promotes=["*"])
    time_ivc = om.IndepVarComp()
    time_ivc.add_output("x12_initial", shape=1, distributed=True)
    time_ivc.add_output("x43_initial", shape=1, distributed=True)
    time_test_prob.model.add_subsystem("time_ivc", time_ivc, promotes=["*"])
    # these solvers are only necessary so that check_partials actually produces results
    time_test_prob.model.nonlinear_solver = om.NonlinearBlockJac(iprint=-1)
    time_test_prob.model.linear_solver = om.LinearBlockJac(iprint=-1)
    time_test_prob.setup(mode=test_direction)
    time_test_prob.run_model()
    if test_prob.comm.rank > 0:
        data = time_test_prob.check_totals(
            of=["x12_final", "x43_final"],
            wrt=["x12_initial", "x43_initial"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
            out_stream=None,
        )
    else:
        data = time_test_prob.check_totals(
            of=["x12_final", "x43_final"],
            wrt=["x12_initial", "x43_initial"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
        )
    assert_check_totals(data, atol=1e-4, rtol=1e-4)
