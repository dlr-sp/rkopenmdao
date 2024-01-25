import itertools

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.general_utils import ContainsAll
import numpy as np
import pytest

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    third_order_four_stage_esdirk,
    implicit_euler,
    explicit_euler,
    runge_kutta_four,
)
from rkopenmdao.utils.parallel_check_partials import parallel_check_partials


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
            if self.comm.rank == 0:
                self.comm.Send(send_data, dest=1, tag=0)
                self.comm.Recv(recv_data, source=1, tag=1)
                d_inputs["x_old"] -= recv_data
                d_inputs["s_i"] -= delta_t * recv_data
                d_outputs["k_i"] -= delta_t * butcher_diagonal_element * recv_data
                d_outputs["k_i"] += d_residuals["k_i"]
            elif self.comm.rank == 1:
                self.comm.Recv(recv_data, source=0, tag=0)
                self.comm.Send(send_data, dest=0, tag=1)
                d_inputs["x_old"] += recv_data
                d_inputs["s_i"] += delta_t * recv_data
                d_outputs["k_i"] += delta_t * butcher_diagonal_element * recv_data
                d_outputs["k_i"] += d_residuals["k_i"]

    def solve_linear(self, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        if mode == "fwd":
            send_data = np.zeros(1)
            send_data[0] = d_residuals["k_i"]
            recv_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Send(send_data, dest=1, tag=0)
                self.comm.Recv(recv_data, source=1, tag=1)
                d_outputs["k_i"] = (
                    d_residuals["k_i"]
                    + delta_t * butcher_diagonal_element * recv_data[0]
                ) / (1 + (delta_t * butcher_diagonal_element) ** 2)
            else:
                self.comm.Recv(recv_data, source=0, tag=0)
                self.comm.Send(send_data, dest=0, tag=1)
                d_outputs["k_i"] = (
                    d_residuals["k_i"]
                    - delta_t * butcher_diagonal_element * recv_data[0]
                ) / (1 + (delta_t * butcher_diagonal_element) ** 2)
        elif mode == "rev":
            send_data = np.zeros(1)
            send_data[0] = d_outputs["k_i"]
            recv_data = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Send(send_data, dest=1, tag=0)
                self.comm.Recv(recv_data, source=1, tag=1)
                d_residuals["k_i"] = (
                    d_outputs["k_i"] - delta_t * butcher_diagonal_element * recv_data[0]
                ) / (1 + (delta_t * butcher_diagonal_element) ** 2)
            else:
                self.comm.Recv(recv_data, source=0, tag=0)
                self.comm.Send(send_data, dest=0, tag=1)
                d_residuals["k_i"] = (
                    d_outputs["k_i"] + delta_t * butcher_diagonal_element * recv_data[0]
                ) / (1 + (delta_t * butcher_diagonal_element) ** 2)


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
        elif mode == "rev":
            d_inputs["x12_old"] += d_residuals["x12"]
            d_inputs["s12_i"] += delta_t * d_residuals["x12"]
            d_outputs["k12_i"] += (
                delta_t * butcher_diagonal_element * d_residuals["x12"]
            )
            d_outputs["x12"] -= d_residuals["x12"]
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                exch_data[0] = d_residuals["k12_i"]
                self.comm.Send(exch_data, dest=1, tag=1)
                d_inputs["x43"] += d_residuals["k12_i"]
                d_outputs["k12_i"] -= d_residuals["k12_i"]
            elif self.comm.rank == 1:
                self.comm.Recv(exch_data, source=0, tag=1)
                d_outputs["x12"] += exch_data[0]
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

        elif mode == "rev":
            d_inputs["x43_old"] += d_residuals["x43"]
            d_inputs["s43_i"] += delta_t * d_residuals["x43"]
            d_outputs["k43_i"] += (
                delta_t * butcher_diagonal_element * d_residuals["x43"]
            )
            d_outputs["x43"] -= d_residuals["x43"]
            exch_data = np.zeros(1)

            if self.comm.rank == 0:
                self.comm.Recv(exch_data, source=1, tag=2)
                d_outputs["x43"] += exch_data[0]
                d_outputs["k43_i"] -= d_residuals["k43_i"]
            elif self.comm.rank == 1:
                exch_data[0] = d_residuals["k43_i"]
                self.comm.Send(exch_data, dest=0, tag=2)
                d_inputs["x12"] += d_residuals["k43_i"]
                d_outputs["k43_i"] -= d_residuals["k43_i"]


@pytest.mark.mpi
@pytest.mark.parametrize(
    """num_steps, delta_t, butcher_tableau""",
    (
        itertools.product(
            [1, 10, 20],
            [0.1, 0.01, 0.001],
            [explicit_euler, implicit_euler, third_order_four_stage_esdirk],
        )
    ),
)
def test_parallel_single_distributed_partial(num_steps, delta_t, butcher_tableau):
    integration_control = IntegrationControl(0.0, num_steps, delta_t)
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
    test_prob.model.linear_solver = om.PETScKrylov(iprint=-1)

    test_prob.setup()

    test_prob.run_model()
    fwd_mat = np.zeros(6)
    rev_mat = np.zeros(6)
    inputs = test_comp._vector_class("nonlinear", "input", test_prob.model)
    outputs = test_comp._vector_class("nonlinear", "output", test_prob.model)
    d_inputs = test_comp._vector_class("linear", "input", test_prob.model)
    d_outputs = test_comp._vector_class("linear", "output", test_prob.model)
    d_residuals = test_comp._vector_class("linear", "residual", test_prob.model)
    for i in range(2):
        for j, tup in enumerate(
            [
                (1.0 * ((i + test_prob.comm.rank) % 2 == 0), 0.0, 0.0),
                (0.0, 1.0 * ((i + test_prob.comm.rank) % 2 == 0), 0.0),
                (0.0, 0.0, 1.0 * ((i + test_prob.comm.rank) % 2 == 0)),
            ]
        ):
            d_inputs["x_old"] = tup[0]
            d_inputs["s_i"] = tup[1]
            d_outputs["k_i"] = tup[2]
            d_residuals["k_i"] = 0.0
            test_comp.apply_linear(
                inputs, outputs, d_inputs, d_outputs, d_residuals, mode="fwd"
            )
            fwd_mat[i + 2 * j] = d_residuals["k_i"]

        d_inputs["x_old"] = 0.0
        d_inputs["s_i"] = 0.0
        d_outputs["k_i"] = 0.0
        d_residuals["k_i"] = 1.0 * ((i + test_prob.comm.rank) % 2 == 0)
        test_comp.apply_linear(
            inputs, outputs, d_inputs, d_outputs, d_residuals, mode="rev"
        )
        rev_mat[i] = d_inputs["x_old"]
        rev_mat[i + 2] = d_inputs["s_i"]
        rev_mat[i + 4] = d_outputs["k_i"]
    assert np.allclose(fwd_mat, rev_mat)

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
    time_test_prob.setup()
    time_test_prob.run_model()

    fwd_mat = np.zeros(2)
    rev_mat = np.zeros(2)
    for i in range(2):
        time_test_prob.model._dinputs.set_val(0.0)
        time_test_prob.model._doutputs.set_val(0.0)
        time_test_prob.model._dresiduals.set_val(0.0)
        src_name = time_test_prob.model.get_source("x_initial")
        if time_test_prob.comm.rank == 0:
            time_test_prob.model._doutputs[src_name] = 1 - i
        else:
            time_test_prob.model._doutputs[src_name] = i
        time_test_prob.model.run_apply_linear(mode="fwd")
        fwd_mat[i] = time_test_prob.model._dresiduals["x_final"]
        time_test_prob.model._dinputs.set_val(0.0)
        time_test_prob.model._doutputs.set_val(0.0)
        time_test_prob.model._dresiduals.set_val(0.0)
        if time_test_prob.comm.rank == 0:
            time_test_prob.model._dresiduals["x_final"] = 1.0 - i
        else:
            time_test_prob.model._dresiduals["x_final"] = i
        time_test_prob.model.run_apply_linear(mode="rev")
        rev_mat[i] = time_test_prob.model._doutputs[src_name]
    print(time_test_prob.comm.rank, fwd_mat)
    print(time_test_prob.comm.rank, rev_mat)
    assert np.allclose(fwd_mat, rev_mat)


@pytest.mark.mpi
@pytest.mark.parametrize(
    """num_steps, delta_t, butcher_tableau""",
    (
        itertools.product(
            [1],  # , 10, 20],
            [0.1],  # 0.01, 0.001],
            [runge_kutta_four],  # , implicit_euler, third_order_four_stage_esdirk],
        )
    ),
)
def test_parallel_two_distributed_partial(num_steps, delta_t, butcher_tableau):
    integration_control = IntegrationControl(0.0, num_steps, delta_t)
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
    test_prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    test_prob.model.linear_solver = om.PETScKrylov()
    test_prob.setup()

    test_prob.run_model()

    src_names_fwd = [
        test_prob.model.get_source("x12_old"),
        test_prob.model.get_source("s12_i"),
        "k12_i",
        "x12",
        test_prob.model.get_source("x43_old"),
        test_prob.model.get_source("s43_i"),
        "k43_i",
        "x43",
    ]
    src_names_rev = ["k12_i", "x12", "k43_i", "x43"]

    fwd_mat = np.zeros((4, 16))
    rev_mat = np.zeros((4, 16))
    for i in range(2):
        for j in range(8):
            test_prob.model._dinputs.set_val(0.0)
            test_prob.model._doutputs.set_val(0.0)
            test_prob.model._dresiduals.set_val(0.0)
            if test_prob.comm.rank == 0:
                test_prob.model._doutputs[src_names_fwd[j]] = 1 - i
            else:
                test_prob.model._doutputs[src_names_fwd[j]] = i
            test_prob.model.run_apply_linear(mode="fwd")
            for k in range(4):
                fwd_mat[k, 2 * j + i] = test_prob.model._dresiduals[src_names_rev[k]]
        for k in range(4):
            test_prob.model._dinputs.set_val(0.0)
            test_prob.model._doutputs.set_val(0.0)
            test_prob.model._dresiduals.set_val(0.0)
            if test_prob.comm.rank == 0:
                test_prob.model._dresiduals[src_names_rev[k]] = 1 - i
            else:
                test_prob.model._dresiduals[src_names_rev[k]] = i
            test_prob.model.run_apply_linear(mode="rev")
            for j in range(8):
                rev_mat[k, 2 * j + i] = test_prob.model._doutputs[src_names_fwd[j]]
    print(test_prob.comm.rank, fwd_mat)
    print(test_prob.comm.rank, rev_mat)
    assert np.allclose(fwd_mat, rev_mat)

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
    time_test_prob.setup()
    time_test_prob.run_model()

    print(parallel_check_partials(time_test_prob, range(2, 4), range(0, 2)))

    # fwd_mat = np.zeros((2, 4))
    # rev_mat = np.zeros((2, 4))
    # solution = np.zeros((4, 4))
    #
    # src_names_fwd = [
    #     time_test_prob.model.get_source("x12_initial"),
    #     time_test_prob.model.get_source("x43_initial"),
    # ]
    # src_names_rev = ["x12_final", "x43_final"]
    # for i in range(2):
    #     for j in range(2):
    #         time_test_prob.model._dinputs.set_val(0.0)
    #         time_test_prob.model._doutputs.set_val(0.0)
    #         time_test_prob.model._dresiduals.set_val(0.0)
    #         if time_test_prob.comm.rank == 0:
    #             time_test_prob.model._doutputs[src_names_fwd[j]] = 1 - i
    #         else:
    #             time_test_prob.model._doutputs[src_names_fwd[j]] = i
    #         time_test_prob.model.run_apply_linear(mode="fwd")
    #         for k in range(2):
    #             fwd_mat[k, 2 * j + i] = time_test_prob.model._dresiduals[
    #                 src_names_rev[k]
    #             ]
    #     for k in range(2):
    #         time_test_prob.model._dinputs.set_val(0.0)
    #         time_test_prob.model._doutputs.set_val(0.0)
    #         time_test_prob.model._dresiduals.set_val(0.0)
    #         if time_test_prob.comm.rank == 0:
    #             time_test_prob.model._dresiduals[src_names_rev[k]] = 1 - i
    #         else:
    #             time_test_prob.model._dresiduals[src_names_rev[k]] = i
    #         time_test_prob.model.run_apply_linear(mode="rev")
    #         for j in range(2):
    #             rev_mat[k, 2 * j + i] = time_test_prob.model._doutputs[src_names_fwd[j]]
    # print(time_test_prob.comm.rank, fwd_mat)
    # print(time_test_prob.comm.rank, rev_mat)
    # assert np.allclose(fwd_mat, rev_mat)
