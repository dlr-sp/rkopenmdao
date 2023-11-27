from mpi4py import MPI

import openmdao.api as om
import numpy as np
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    third_order_four_stage_esdirk,
    implicit_euler,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator


class ComponentPart1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x_old", shape=1, tags=["x", "step_input_var"])
        self.add_input("sx_i", shape=1, tags=["x", "accumulated_stage_var"])
        self.add_input("y_stage", shape=1)
        self.add_output("kx_i", shape=1, tags=["x", "stage_output_var"])
        self.add_output("x_stage", shape=1)

    def compute(self, inputs, outputs):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        outputs["x_stage"] = (
            inputs["x_old"]
            + delta_t * inputs["sx_i"]
            - delta_t * butcher_diagonal_element * inputs["y_stage"]
        )
        outputs["kx_i"] = -inputs["y_stage"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t

        if mode == "fwd":
            d_outputs["x_stage"] += (
                d_inputs["x_old"]
                + delta_t * d_inputs["sx_i"]
                - delta_t * butcher_diagonal_element * d_inputs["y_stage"]
            )
            d_outputs["kx_i"] -= d_inputs["y_stage"]
        elif mode == "rev":
            d_inputs["x_old"] += d_outputs["x_stage"]
            d_inputs["sx_i"] += delta_t * d_outputs["x_stage"]
            d_inputs["y_stage"] -= (
                delta_t * butcher_diagonal_element * d_outputs["x_stage"]
                + d_outputs["kx_i"]
            )


class ComponentPart2(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y_old", shape=1, tags=["y", "step_input_var"])
        self.add_input("sy_i", shape=1, tags=["y", "accumulated_stage_var"])
        self.add_input("x_stage", shape=1)
        self.add_output("ky_i", shape=1, tags=["y", "stage_output_var"])
        self.add_output("y_stage", shape=1)

    def compute(self, inputs, outputs):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t
        outputs["y_stage"] = (
            inputs["y_old"]
            + delta_t * inputs["sy_i"]
            + delta_t * butcher_diagonal_element * inputs["x_stage"]
        )
        outputs["ky_i"] = inputs["x_stage"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        delta_t = self.options["integration_control"].delta_t

        if mode == "fwd":
            d_outputs["y_stage"] += (
                d_inputs["y_old"]
                + delta_t * d_inputs["sy_i"]
                + delta_t * butcher_diagonal_element * d_inputs["x_stage"]
            )
            d_outputs["ky_i"] += d_inputs["x_stage"]
        elif mode == "rev":
            d_inputs["y_old"] += d_outputs["y_stage"]
            d_inputs["sy_i"] += delta_t * d_outputs["y_stage"]
            d_inputs["x_stage"] += (
                delta_t * butcher_diagonal_element * d_outputs["y_stage"]
                + d_outputs["ky_i"]
            )


if __name__ == "__main__":
    butcher_tableau = implicit_euler
    integration_control = IntegrationControl(0.0, 3, 0.1)
    prob = om.Problem()

    par_group = om.ParallelGroup()
    first_group = om.Group()
    first_group.add_subsystem(
        "first", ComponentPart1(integration_control=integration_control), promotes=["*"]
    )
    first_indep = om.IndepVarComp()
    first_indep.add_output("x_old")
    first_indep.add_output("sx_i")
    first_group.add_subsystem("first_indep", first_indep, promotes=["*"])
    par_group.add_subsystem("first_group", first_group, promotes=["*"])

    second_group = om.Group()
    second_group.add_subsystem(
        "second",
        ComponentPart2(integration_control=integration_control),
        promotes=["*"],
    )
    second_indep = om.IndepVarComp()
    second_indep.add_output("y_old")
    second_indep.add_output("sy_i")
    second_group.add_subsystem("second_indep", second_indep, promotes=["*"])
    par_group.add_subsystem("second_group", second_group, promotes=["*"])

    prob.model.add_subsystem("parallel_group", par_group, promotes=["*"])
    # indep = om.IndepVarComp()
    # if prob.comm.rank == 0:
    #     indep.add_output("x_old", val=1.0, shape=1, distributed=True)
    # elif prob.comm.rank == 1:
    #     indep.add_output("x_old", val=0.0, shape=1, distributed=True)
    # indep.add_output("s_i", val=0.0, shape=1, distributed=True)

    # prob.model.add_subsystem("indep", indep, promotes=["*"])
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.linear_solver = om.PETScKrylov()
    prob.setup()
    outer_prob = om.Problem()
    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=prob,
        integration_control=integration_control,
        butcher_tableau=butcher_tableau,
        time_integration_quantities=["x", "y"]
        if prob.comm.size == 1
        else ["x"]
        if prob.comm.rank == 0
        else ["y"],
    )

    outer_prob.model.add_subsystem(
        "rk_integrator",
        rk_integrator,
        promotes=["*"],
    )

    # outer_indep = om.IndepVarComp()
    # if outer_prob.comm.rank == 0:
    #     outer_indep.add_output("x_initial", val=1.0, shape=1, distributed=True)
    # elif outer_prob.comm.rank == 1:
    #     outer_indep.add_output("x_initial", val=0.0, shape=1, distributed=True)
    # outer_prob.model.add_subsystem("outer_indep", outer_indep, promotes=["*"])
    #
    outer_prob.setup()

    outer_prob.run_model()

    # outer_prob.check_partials()

    # outer_prob.check_totals(
    #     ["x_final", "y_final"]
    #     if prob.comm.size == 1
    #     else ["x_final"]
    #     if prob.comm.rank == 0
    #     else ["y_final"],
    #     ["x_initial", "y_initial"]
    #     if prob.comm.size == 1
    #     else ["x_initial"]
    #     if prob.comm.rank == 0
    #     else ["y_initial"],
    # )

    d_inputs = om.DefaultVector("linear", "input", rk_integrator)

    for i in range(2):
        if outer_prob.comm.rank == 0:
            d_inputs["x_initial"] = i
        elif outer_prob.comm.rank == 1:
            d_inputs["y_initial"] = 1 - i

        d_outputs = om.DefaultVector("linear", "output", rk_integrator)
        rk_integrator.compute_jacvec_product(
            outer_prob.model._inputs, d_inputs, d_outputs, "fwd"
        )
        outer_prob.comm.Barrier()
        if outer_prob.comm.rank == 0:
            print("fwd", outer_prob.comm.rank, d_outputs["x_final"])
        elif outer_prob.comm.rank == 1:
            print("fwd", outer_prob.comm.rank, d_outputs["y_final"])

        if outer_prob.comm.rank == 0:
            d_outputs["x_final"] = i
            d_inputs["x_initial"] = 0.0
        elif outer_prob.comm.rank == 1:
            d_outputs["y_final"] = 1 - i

        outer_prob.comm.Barrier()
        print("reverse test")
        rk_integrator.compute_jacvec_product(
            outer_prob.model._inputs, d_inputs, d_outputs, "rev"
        )
        outer_prob.comm.Barrier()
        if outer_prob.comm.rank == 0:
            print("rev", outer_prob.comm.rank, d_inputs["x_initial"])
        elif outer_prob.comm.rank == 1:
            print("rev", outer_prob.comm.rank, d_inputs["y_initial"])