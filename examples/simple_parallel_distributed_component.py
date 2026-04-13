"""Example to show how the RungeKuttaIntegrator works with MPI using distributed
components in the time_stage_problem"""

import openmdao.api as om
import numpy as np

from rkopenmdao.butcher_tableaux import embedded_third_order_four_stage_esdirk
from rkopenmdao.components import ExplicitUnsteadyComponent
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps


# pylint: disable=arguments-differ
class SimpleDistributedComponent(ExplicitUnsteadyComponent):
    """Component for the ODE system x' = -y, y' = x. The equations are distributed over
    two ranks."""

    def setup(self):
        self.add_input("x_old", shape=1, distributed=True, tags=["x", "step_input_var"])
        self.add_input(
            "s_i", shape=1, distributed=True, tags=["x", "accumulated_stage_var"]
        )
        self.add_output(
            "k_i", shape=1, distributed=True, tags=["x", "stage_output_var"]
        )

    def compute(self, inputs, outputs):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        own_influence = inputs["x_old"] + delta_t * inputs["s_i"]
        other_influence = np.zeros(1)
        if self.comm.rank == 0:
            self.comm.Send(own_influence, dest=1, tag=self.comm.rank)
            self.comm.Recv(other_influence, source=1, tag=self.comm.rank)
            outputs["k_i"] = (
                -delta_t * butcher_diagonal_element * own_influence - other_influence
            ) / ((delta_t * butcher_diagonal_element) ** 2 + 1)

        elif self.comm.rank == 1:
            self.comm.Recv(other_influence, source=0, tag=1 - self.comm.rank)
            self.comm.Send(own_influence, dest=0, tag=1 - self.comm.rank)
            outputs["k_i"] = (
                -delta_t * butcher_diagonal_element * own_influence + other_influence
            ) / ((delta_t * butcher_diagonal_element) ** 2 + 1)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        if mode == "fwd":
            own_influence = d_inputs["x_old"] + delta_t * d_inputs["s_i"]
            other_influence = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Send(own_influence, dest=1, tag=self.comm.rank)
                self.comm.Recv(other_influence, source=1, tag=self.comm.rank)
                d_outputs["k_i"] += (
                    -delta_t * butcher_diagonal_element * own_influence
                    - other_influence
                ) / ((delta_t * butcher_diagonal_element) ** 2 + 1)
            elif self.comm.rank == 1:
                self.comm.Recv(other_influence, source=0, tag=1 - self.comm.rank)
                self.comm.Send(own_influence, dest=0, tag=1 - self.comm.rank)
                d_outputs["k_i"] += (
                    -delta_t * butcher_diagonal_element * own_influence
                    + other_influence
                ) / ((delta_t * butcher_diagonal_element) ** 2 + 1)
        elif mode == "rev":
            other_influence = np.zeros(1)
            if self.comm.rank == 0:
                self.comm.Send(d_outputs["k_i"], dest=1, tag=self.comm.rank)
                self.comm.Recv(other_influence, source=1, tag=self.comm.rank)

                d_inputs["x_old"] += (
                    -delta_t * butcher_diagonal_element * d_outputs["k_i"]
                    + other_influence
                ) / ((delta_t * butcher_diagonal_element) ** 2 + 1)
                d_inputs["s_i"] += (
                    delta_t
                    * (
                        -delta_t * butcher_diagonal_element * d_outputs["k_i"]
                        + other_influence
                    )
                    / ((delta_t * butcher_diagonal_element) ** 2 + 1)
                )

            elif self.comm.rank == 1:
                self.comm.Recv(other_influence, source=0, tag=1 - self.comm.rank)
                self.comm.Send(d_outputs["k_i"], dest=0, tag=1 - self.comm.rank)

                d_inputs["x_old"] += (
                    -delta_t * butcher_diagonal_element * d_outputs["k_i"]
                    - other_influence
                ) / ((delta_t * butcher_diagonal_element) ** 2 + 1)
                d_inputs["s_i"] += (
                    delta_t
                    * (
                        -delta_t * butcher_diagonal_element * d_outputs["k_i"]
                        - other_influence
                    )
                    / ((delta_t * butcher_diagonal_element) ** 2 + 1)
                )


if __name__ == "__main__":
    butcher_tableau = embedded_third_order_four_stage_esdirk
    integration_config = IntegrationConfig(False, PredefinedNumberOfSteps(3), 0.1)
    prob = om.Problem()

    prob.model.add_subsystem(
        "parallel_comp",
        SimpleDistributedComponent(),
        promotes=["*"],
    )
    indep = om.IndepVarComp()
    if prob.comm.rank == 0:
        indep.add_output("x_old", val=1.0, shape=1, distributed=True)
    elif prob.comm.rank == 1:
        indep.add_output("x_old", val=0.0, shape=1, distributed=True)
    indep.add_output("s_i", val=0.0, shape=1, distributed=True)

    prob.model.add_subsystem("indep", indep, promotes=["*"])
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=-1)
    prob.model.linear_solver = om.PETScKrylov(iprint=-1)
    prob.setup()

    outer_prob = om.Problem()
    rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=prob,
        integration_config=integration_config,
        butcher_tableau=butcher_tableau,
        time_integration_quantities=["x"],
    )

    outer_prob.model.add_subsystem(
        "rk_integrator",
        rk_integrator,
        promotes=["*"],
    )

    outer_indep = om.IndepVarComp()
    if outer_prob.comm.rank == 0:
        outer_indep.add_output("x_initial", val=1.0, shape=1, distributed=True)
    elif outer_prob.comm.rank == 1:
        outer_indep.add_output("x_initial", val=0.0, shape=1, distributed=True)
    outer_prob.model.add_subsystem("outer_indep", outer_indep, promotes=["*"])

    outer_prob.setup()

    outer_prob.run_model()
