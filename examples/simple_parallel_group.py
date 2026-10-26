"""Example to show how the OpenMDAOTimeStepping works with MPI using parallel groups
in the time_stage_problem."""

import openmdao.api as om

from rkopenmdao.butcher_tableaux import (
    embedded_third_order_four_stage_esdirk,
)
from rkopenmdao.checkpointed_time_integration.no_checkpoint_time_integration import (
    NoCheckpointTimeIntegration,
)
from rkopenmdao.components import ExplicitUnsteadyComponent
from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.openmdao_time_stepping import OpenMDAOTimeStepping
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
)


# pylint: disable=arguments-differ
class ComponentPart1(ExplicitUnsteadyComponent):
    """This component models x' = -y, part 2 models y' = x"""

    def setup(self):
        self.add_input("x_old", shape=1, tags=["x", "step_input_var"])
        self.add_input("sx_i", shape=1, tags=["x", "accumulated_stage_var"])
        self.add_input("y_stage", shape=1)
        self.add_output("kx_i", shape=1, tags=["x", "stage_output_var"])
        self.add_output("x_stage", shape=1)

    def compute(self, inputs, outputs):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        outputs["x_stage"] = (
            inputs["x_old"]
            + delta_t * inputs["sx_i"]
            - delta_t * butcher_diagonal_element * inputs["y_stage"]
        )
        outputs["kx_i"] = -inputs["y_stage"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size

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


class ComponentPart2(ExplicitUnsteadyComponent):
    """This component models y' = x, part 1 models x' = -y"""

    def setup(self):
        self.add_input("y_old", shape=1, tags=["y", "step_input_var"])
        self.add_input("sy_i", shape=1, tags=["y", "accumulated_stage_var"])
        self.add_input("x_stage", shape=1)
        self.add_output("ky_i", shape=1, tags=["y", "stage_output_var"])
        self.add_output("y_stage", shape=1)

    def compute(self, inputs, outputs):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size
        outputs["y_stage"] = (
            inputs["y_old"]
            + delta_t * inputs["sy_i"]
            + delta_t * butcher_diagonal_element * inputs["x_stage"]
        )
        outputs["ky_i"] = inputs["x_stage"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        butcher_diagonal_element = self.om_data_exchange.stage_factor
        delta_t = self.om_data_exchange.step_size

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
    butcher_tableau = embedded_third_order_four_stage_esdirk
    integration_config = IntegrationConfig(False, PredefinedNumberOfSteps(3), 0.1)
    prob = om.Problem()

    par_group = om.ParallelGroup()
    first_group = om.Group()
    first_group.add_subsystem("first", ComponentPart1(), promotes=["*"])
    first_indep = om.IndepVarComp()
    first_indep.add_output("x_old")
    first_indep.add_output("sx_i")
    first_group.add_subsystem("first_indep", first_indep, promotes=["*"])
    par_group.add_subsystem("first_group", first_group, promotes=["*"])

    second_group = om.Group()
    second_group.add_subsystem(
        "second",
        ComponentPart2(),
        promotes=["*"],
    )
    second_indep = om.IndepVarComp()
    second_indep.add_output("y_old")
    second_indep.add_output("sy_i")
    second_group.add_subsystem("second_indep", second_indep, promotes=["*"])
    par_group.add_subsystem("second_group", second_group, promotes=["*"])

    prob.model.add_subsystem("parallel_group", par_group, promotes=["*"])

    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.linear_solver = om.PETScKrylov()
    prob.setup()
    prob.final_setup()
    outer_prob = om.Problem()
    rk_indep = om.IndepVarComp()
    rk_indep.add_output("x_initial", shape_by_conn=True, distributed=True)
    rk_indep.add_output("y_initial", shape_by_conn=True, distributed=True)
    time_integration = NoCheckpointTimeIntegration(
        ode=OpenMDAOODE(prob, ["x", "y"]),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(
            butcher_tableau
        ),
        time_integration_config=integration_config,
    )
    outer_prob.model.add_subsystem("rk_indep", rk_indep, promotes=["*"])
    outer_prob.model.add_subsystem(
        "time_integration",
        OpenMDAOTimeStepping(time_integrator=time_integration),
        promotes=["*"],
    )

    outer_prob.setup()

    outer_prob.run_model()
