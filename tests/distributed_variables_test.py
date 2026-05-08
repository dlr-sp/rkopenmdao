"""Test to make sure that rkopenmdao works with problems containing distributed
variables."""

import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_totals
import numpy as np
import pytest

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk,
)
from rkopenmdao.checkpoint_interface.checkpointed_time_integration import (
    CheckpointInterface,
)
from rkopenmdao.checkpoint_interface.no_checkpoint_time_integration import (
    NoCheckpointer,
)
from rkopenmdao.checkpoint_interface.all_checkpoint_time_integration import (
    AllCheckpointer,
)
from rkopenmdao.checkpoint_interface.pyrevolve_time_integration import (
    PyrevolveTimeIntegration,
)
from rkopenmdao.components import ImplicitUnsteadyComponent
from rkopenmdao.error_controllers import integral
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps


class Test1Component1(ImplicitUnsteadyComponent):
    """Models x' = y, y' = -x and distributed the equations over two cores."""

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
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
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
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
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


def ode_1_analytical_solution(time, initial_values):
    """Analytical solution for the ODE of the above component."""
    return np.array(
        [
            initial_values[0] * np.cos(time) + initial_values[1] * np.sin(time),
            -initial_values[0] * np.sin(time) + initial_values[1] * np.cos(time),
        ]
    )


# The following two components model the system
#   x_1' = x_4
#   x_2' = x_1
#   x_3' = x_2
#   x_4' = x_3
class Test2Component1(ImplicitUnsteadyComponent):
    """Models the first two equations from above, with the first being on rank 0 and the
    second on rank 1"""

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
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        residuals["x12"] = (
            inputs["x12_old"]
            + delta_t * (inputs["s12_i"] + butcher_diagonal_element * outputs["k12_i"])
            - outputs["x12"]
        )
        exch_data = np.zeros(1)
        if self.comm.rank == 0:
            exch_data[0] = outputs["x12"][0]
            self.comm.Send(exch_data, dest=1, tag=0)
            residuals["k12_i"] = inputs["x43"] - outputs["k12_i"]
        elif self.comm.rank == 1:
            self.comm.Recv(exch_data, source=0, tag=0)
            residuals["k12_i"] = exch_data[0] - outputs["k12_i"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        if mode == "fwd":
            d_residuals["x12"] += (
                d_inputs["x12_old"]
                + delta_t
                * (d_inputs["s12_i"] + butcher_diagonal_element * d_outputs["k12_i"])
                - d_outputs["x12"]
            )
            exch_data = np.zeros(1)
            if self.comm.rank == 0:
                exch_data[0] = d_outputs["x12"][0]
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
                exch_data[0] = d_residuals["k12_i"][0]
                self.comm.Send(exch_data, dest=0, tag=1)
                d_outputs["k12_i"] -= d_residuals["k12_i"]


class Test2Component2(ImplicitUnsteadyComponent):
    """Models the last two equations from above, with the first being on rank 1 and the
    second on rank 0"""

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
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
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
            exch_data[0] = outputs["x43"][0]
            self.comm.Send(exch_data, dest=0, tag=3)
            residuals["k43_i"] = inputs["x12"] - outputs["k43_i"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
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
                exch_data[0] = d_outputs["x43"][0]
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
                exch_data[0] = d_residuals["k43_i"][0]
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


def setup_time_integration_problem(
    stage_prob: om.Problem,
    butcher_tableau: ButcherTableau,
    integration_config: IntegrationConfig,
    time_integration_quantities: list[str],
    initial_values: dict | None = None,
    setup_mode: str = "auto",
    checkpointing_implementation: type[CheckpointInterface] = NoCheckpointer,
):
    """Sets up the time integration problem for the following test."""
    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=stage_prob,
        butcher_tableau=butcher_tableau,
        integration_config=integration_config,
        time_integration_quantities=time_integration_quantities,
        checkpointing_type=checkpointing_implementation,
        error_controller=[integral],
    )

    time_test_prob = om.Problem()
    time_test_prob.model.add_subsystem("rk_integrator", rk_integrator, promotes=["*"])
    time_ivc = om.IndepVarComp()
    for var in time_integration_quantities:
        time_ivc.add_output(f"{var}_initial", shape=1, distributed=True)
    time_test_prob.model.add_subsystem("time_ivc", time_ivc, promotes=["*"])
    # these solvers are only necessary so that check_partials actually produces results
    time_test_prob.model.nonlinear_solver = om.NonlinearBlockJac(iprint=-1)
    time_test_prob.model.linear_solver = om.LinearBlockJac(iprint=-1)
    time_test_prob.setup(setup_mode)
    time_test_prob["time_initial"][0] = 0.0
    if initial_values is not None:
        for var in time_integration_quantities:
            time_test_prob[f"{var}_initial"][:] = initial_values[var][
                time_test_prob.comm.rank
            ]

    return time_test_prob


def setup_integration_config(delta_t: float, num_steps: int):
    """Sets up the configuration for the majority of time integrations."""
    return IntegrationConfig(False, PredefinedNumberOfSteps(num_steps), delta_t)


def setup_parallel_single_distributed_problem(
    setup_mode="auto",
) -> om.Problem:
    """Sets up the stage problem for the single component test cases."""
    test_comp = Test1Component1()

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
    test_prob.setup(setup_mode)
    return test_prob


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_two_stage_sdirk]
)
@pytest.mark.parametrize("initial_values", [{"x": [1, 1]}, {"x": [0, 0]}])
def test_parallel_single_distributed_time_integration(
    num_steps, butcher_tableau, initial_values
):
    """Tests time integration with distributed variables for a single component."""
    delta_t = 0.0001
    integration_config = setup_integration_config(delta_t, num_steps)
    test_prob = setup_parallel_single_distributed_problem()

    test_prob.run_model()
    time_test_prob = setup_time_integration_problem(
        test_prob, butcher_tableau, integration_config, ["x"], initial_values
    )

    time_test_prob.run_model()

    analytical_solution = ode_1_analytical_solution(
        num_steps * delta_t, initial_values["x"]
    )

    assert time_test_prob.get_val("x_final", get_remote=False) == pytest.approx(
        analytical_solution[time_test_prob.comm.rank]
    )


@pytest.mark.mpi
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_two_stage_sdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
@pytest.mark.parametrize(
    "num_steps, checkpointing_implementation",
    [(1, AllCheckpointer), (10, AllCheckpointer), (10, PyrevolveTimeIntegration)],
)
def test_parallel_single_distributed_totals(
    num_steps, butcher_tableau, test_direction, checkpointing_implementation
):
    """Tests totals of time integration with distributed variables for a single
    component."""
    delta_t = 0.1
    integration_config = setup_integration_config(delta_t, num_steps)
    test_prob = setup_parallel_single_distributed_problem(test_direction)

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

    time_test_prob = setup_time_integration_problem(
        test_prob,
        butcher_tableau,
        integration_config,
        ["x"],
        setup_mode=test_direction,
        checkpointing_implementation=checkpointing_implementation,
    )
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


def setup_parallel_two_distributed_problem(
    setup_mode: str = "auto",
) -> om.Problem:
    """Sets up the stage problem for the two component test cases."""
    test_comp1 = Test2Component1()
    test_comp2 = Test2Component2()
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
    return test_prob


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_two_stage_sdirk]
)
@pytest.mark.parametrize(
    "initial_values", [{"x12": [1, 1], "x43": [1, 1]}, {"x12": [0, 0], "x43": [0, 0]}]
)
def test_parallel_two_distributed_time_integration(
    num_steps, butcher_tableau, initial_values
):
    """Tests time integration with distributed variables for a problem with two
    components."""
    delta_t = 0.0001
    integration_config = setup_integration_config(delta_t, num_steps)
    test_prob = setup_parallel_two_distributed_problem()

    time_test_prob = setup_time_integration_problem(
        test_prob,
        butcher_tableau,
        integration_config,
        ["x12", "x43"],
        initial_values,
    )
    time_test_prob.run_model()

    analytical_solution = ode_2_analytical_solution(
        num_steps * delta_t, initial_values["x12"] + initial_values["x43"]
    )

    numerical_solution = np.array(
        [
            time_test_prob.get_val("x12_final", get_remote=False),
            time_test_prob.get_val("x43_final", get_remote=False),
        ]
    ).flatten()

    assert np.allclose(
        numerical_solution,
        analytical_solution[[time_test_prob.comm.rank, 3 - time_test_prob.comm.rank]],
    )


@pytest.mark.mpi
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_two_stage_sdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
@pytest.mark.parametrize(
    "num_steps, checkpointing_implementation",
    [(1, AllCheckpointer), (10, AllCheckpointer), (10, PyrevolveTimeIntegration)],
)
def test_parallel_two_distributed_totals(
    num_steps, butcher_tableau, test_direction, checkpointing_implementation
):
    """Tests totals of time integration with distributed variables for a problem with
    two components."""
    delta_t = 0.1
    integration_config = setup_integration_config(delta_t, num_steps)
    test_prob = setup_parallel_two_distributed_problem(test_direction)

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
    # assert_check_totals(data, atol=1e-4, rtol=1e-4)

    time_test_prob = setup_time_integration_problem(
        test_prob,
        butcher_tableau,
        integration_config,
        ["x12", "x43"],
        setup_mode=test_direction,
        checkpointing_implementation=checkpointing_implementation,
    )
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


@pytest.mark.mpi
@pytest.mark.parametrize("steps", [1, 10])
def test_distributed_adaptive_step(steps):
    """Test for Adaptive timestepping"""
    # Distributed:
    integration_config = IntegrationConfig(True, PredefinedNumberOfSteps(steps), 1.0)
    test_prob = setup_parallel_single_distributed_problem()
    test_prob.run_model()
    time_test_prob = setup_time_integration_problem(
        test_prob,
        embedded_second_order_two_stage_sdirk,
        integration_config,
        ["x"],
    )
    time_test_prob.run_model()
