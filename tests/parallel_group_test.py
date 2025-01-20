"""Test to make sure that rkopenmdao works with problems containing parallel groups."""

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals, assert_check_partials
import pytest
import numpy as np

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.integration_control import IntegrationControl, TerminationCriterion
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_three_stage_esdirk,
)
from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.checkpoint_interface.no_checkpointer import NoCheckpointer
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer

# pylint: disable = arguments-differ, too-many-branches


# The following components model the ODE system
#   d' = d
#   c' = c - d
#   b' = b + d
#   a' = a + b + c


class FirstComponent(om.ExplicitComponent):
    """Models the first equation of the above system."""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("d_old", shape=1, tags=["d", "step_input_var"])
        self.add_input(
            "d_accumulated_stages", shape=1, tags=["d", "accumulated_stage_var"]
        )
        self.add_output("d_update", shape=1, tags=["d", "stage_output_var"])
        self.add_output("d_state", shape=1)

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element == 0.0:
            factor = 1.0
        else:
            factor = 1 / (1 - delta_t * butcher_diagonal_element)
        old_influence = inputs["d_old"] + delta_t * inputs["d_accumulated_stages"]
        outputs["d_update"] = factor * old_influence
        outputs["d_state"] = factor * old_influence

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element == 0.0:
            factor = 1.0
        else:
            factor = 1 / (1 - delta_t * butcher_diagonal_element)
        if mode == "fwd":
            if "d_update" in d_outputs:
                if "d_old" in d_inputs:
                    d_outputs["d_update"] += factor * d_inputs["d_old"]
                if "d_accumulated_stages" in d_inputs:
                    d_outputs["d_update"] += (
                        delta_t * factor * d_inputs["d_accumulated_stages"]
                    )
            if "d_state" in d_outputs:
                if "d_old" in d_inputs:
                    d_outputs["d_state"] += factor * d_inputs["d_old"]
                if "d_accumulated_stages" in d_inputs:
                    d_outputs["d_state"] += (
                        delta_t * factor * d_inputs["d_accumulated_stages"]
                    )
        elif mode == "rev":
            if "d_update" in d_outputs:
                if "d_old" in d_inputs:
                    d_inputs["d_old"] += factor * d_outputs["d_update"]
                if "d_accumulated_stages" in d_inputs:
                    d_inputs["d_accumulated_stages"] += (
                        delta_t * factor * d_outputs["d_update"]
                    )
            if "d_state" in d_outputs:
                if "d_old" in d_inputs:
                    d_inputs["d_old"] += factor * d_outputs["d_state"]
                if "d_accumulated_stages" in d_inputs:
                    d_inputs["d_accumulated_stages"] += (
                        delta_t * factor * d_outputs["d_state"]
                    )


class SecondComponent1(om.ExplicitComponent):
    """Models the second equation of the above system."""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("c_old", shape=1, tags=["c", "step_input_var"])
        self.add_input(
            "c_accumulated_stages", shape=1, tags=["c", "accumulated_stage_var"]
        )
        self.add_input("d", shape=1)
        self.add_output("c_update", shape=1, tags=["c", "stage_output_var"])
        self.add_output("c_state", shape=1)

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        old_influence = inputs["c_old"] + delta_t * inputs["c_accumulated_stages"]
        if butcher_diagonal_element != 0.0:
            outputs["c_update"] = (old_influence - inputs["d"]) / (
                1 - butcher_diagonal_element * delta_t
            )
            outputs["c_state"] = (
                old_influence - delta_t * butcher_diagonal_element * inputs["d"]
            ) / (1 - butcher_diagonal_element * delta_t)
        else:
            outputs["c_update"] = old_influence - inputs["d"]
            outputs["c_state"] = old_influence

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element != 0.0:
            factor = 1.0 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        if mode == "fwd":
            if "c_old" in d_inputs:
                if "c_state" in d_outputs:
                    d_outputs["c_state"] += factor * d_inputs["c_old"]
                if "c_update" in d_outputs:
                    d_outputs["c_update"] += factor * d_inputs["c_old"]
            if "c_accumulated_stages" in d_inputs:
                if "c_state" in d_outputs:
                    d_outputs["c_state"] += (
                        factor * delta_t * d_inputs["c_accumulated_stages"]
                    )
                if "c_update" in d_outputs:
                    d_outputs["c_update"] += (
                        factor * delta_t * d_inputs["c_accumulated_stages"]
                    )
            if "d" in d_inputs:
                if "c_update" in d_outputs:
                    d_outputs["c_update"] -= factor * d_inputs["d"]
                if "c_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_outputs["c_state"] -= (
                            factor * delta_t * butcher_diagonal_element * d_inputs["d"]
                        )
        if mode == "rev":
            if "c_old" in d_inputs:
                if "c_state" in d_outputs:
                    d_inputs["c_old"] += factor * d_outputs["c_state"]
                if "c_update" in d_outputs:
                    d_inputs["c_old"] += factor * d_outputs["c_update"]
            if "c_accumulated_stages" in d_inputs:
                if "c_state" in d_outputs:
                    d_inputs["c_accumulated_stages"] += (
                        factor * delta_t * d_outputs["c_state"]
                    )
                if "c_update" in d_outputs:
                    d_inputs["c_accumulated_stages"] += (
                        factor * delta_t * d_outputs["c_update"]
                    )
            if "d" in d_inputs:
                if "c_update" in d_outputs:
                    d_inputs["d"] -= factor * d_outputs["c_update"]
                if "c_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_inputs["d"] -= (
                            factor
                            * delta_t
                            * butcher_diagonal_element
                            * d_outputs["c_state"]
                        )


class SecondComponent2(om.ExplicitComponent):
    """Models the third equation of the above system."""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("b_old", shape=1, tags=["b", "step_input_var"])
        self.add_input(
            "b_accumulated_stages", shape=1, tags=["b", "accumulated_stage_var"]
        )
        self.add_input("d", shape=1)
        self.add_output("b_update", shape=1, tags=["b", "stage_output_var"])
        self.add_output("b_state", shape=1)

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        old_influence = inputs["b_old"] + delta_t * inputs["b_accumulated_stages"]
        if butcher_diagonal_element != 0.0:
            outputs["b_update"] = (old_influence + inputs["d"]) / (
                1 - butcher_diagonal_element * delta_t
            )
            outputs["b_state"] = (
                old_influence + delta_t * butcher_diagonal_element * inputs["d"]
            ) / (1 - butcher_diagonal_element * delta_t)
        else:
            outputs["b_update"] = old_influence + inputs["d"]
            outputs["b_state"] = old_influence

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element != 0.0:
            factor = 1.0 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        if mode == "fwd":
            if "b_old" in d_inputs:
                if "b_state" in d_outputs:
                    d_outputs["b_state"] += factor * d_inputs["b_old"]
                if "b_update" in d_outputs:
                    d_outputs["b_update"] += factor * d_inputs["b_old"]
            if "b_accumulated_stages" in d_inputs:
                if "b_state" in d_outputs:
                    d_outputs["b_state"] += (
                        factor * delta_t * d_inputs["b_accumulated_stages"]
                    )
                if "b_update" in d_outputs:
                    d_outputs["b_update"] += (
                        factor * delta_t * d_inputs["b_accumulated_stages"]
                    )
            if "d" in d_inputs:
                if "b_update" in d_outputs:
                    d_outputs["b_update"] += factor * d_inputs["d"]
                if "b_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_outputs["b_state"] += (
                            factor * delta_t * butcher_diagonal_element * d_inputs["d"]
                        )
        if mode == "rev":
            if "b_old" in d_inputs:
                if "b_state" in d_outputs:
                    d_inputs["b_old"] += factor * d_outputs["b_state"]
                if "b_update" in d_outputs:
                    d_inputs["b_old"] += factor * d_outputs["b_update"]
            if "b_accumulated_stages" in d_inputs:
                if "b_state" in d_outputs:
                    d_inputs["b_accumulated_stages"] += (
                        factor * delta_t * d_outputs["b_state"]
                    )
                if "b_update" in d_outputs:
                    d_inputs["b_accumulated_stages"] += (
                        factor * delta_t * d_outputs["b_update"]
                    )
            if "d" in d_inputs:
                if "b_update" in d_outputs:
                    d_inputs["d"] += factor * d_outputs["b_update"]
                if "b_state" in d_outputs:
                    if butcher_diagonal_element != 0.0:
                        d_inputs["d"] += (
                            factor
                            * delta_t
                            * butcher_diagonal_element
                            * d_outputs["b_state"]
                        )


class ThirdComponent(om.ExplicitComponent):
    """Models the fourth equation of the above system."""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("a_old", shape=1, tags=["a", "step_input_var"])
        self.add_input(
            "a_accumulated_stages", shape=1, tags=["a", "accumulated_stage_var"]
        )
        self.add_input("b", shape=1)
        self.add_input("c", shape=1)
        self.add_output("a_update", shape=1, tags=["a", "stage_output_var"])
        self.add_output("a_state", shape=1)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element != 0.0:
            factor = 1 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        old_influcence = inputs["a_old"] + delta_t * inputs["a_accumulated_stages"]
        outputs["a_update"] = factor * (old_influcence + inputs["b"] + inputs["c"])
        outputs["a_state"] = factor * (
            old_influcence
            + delta_t * butcher_diagonal_element * (inputs["b"] + inputs["c"])
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element != 0.0:
            factor = 1 / (1 - butcher_diagonal_element * delta_t)
        else:
            factor = 1.0
        if mode == "fwd":
            if "a_old" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += factor * d_inputs["a_old"]
                if "a_state" in d_outputs:
                    d_outputs["a_state"] += factor * d_inputs["a_old"]
            if "a_accumulated_stages" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += (
                        factor * delta_t * d_inputs["a_accumulated_stages"]
                    )
                if "a_state" in d_outputs:
                    d_outputs["a_state"] += (
                        factor * delta_t * d_inputs["a_accumulated_stages"]
                    )
            if "b" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += factor * d_inputs["b"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_outputs["a_state"] += (
                        factor * butcher_diagonal_element * delta_t * d_inputs["b"]
                    )
            if "c" in d_inputs:
                if "a_update" in d_outputs:
                    d_outputs["a_update"] += factor * d_inputs["c"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_outputs["a_state"] += (
                        factor * butcher_diagonal_element * delta_t * d_inputs["c"]
                    )
        if mode == "rev":
            if "a_old" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["a_old"] += factor * d_outputs["a_update"]
                if "a_state" in d_outputs:
                    d_inputs["a_old"] += factor * d_outputs["a_state"]
            if "a_accumulated_stages" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["a_accumulated_stages"] += (
                        factor * delta_t * d_outputs["a_update"]
                    )
                if "a_state" in d_outputs:
                    d_inputs["a_accumulated_stages"] += (
                        factor * delta_t * d_outputs["a_state"]
                    )
            if "b" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["b"] += factor * d_outputs["a_update"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_inputs["b"] += (
                        factor
                        * butcher_diagonal_element
                        * delta_t
                        * d_outputs["a_state"]
                    )
            if "c" in d_inputs:
                if "a_update" in d_outputs:
                    d_inputs["c"] += factor * d_outputs["a_update"]
                if "a_state" in d_outputs and butcher_diagonal_element != 0.0:
                    d_inputs["c"] += (
                        factor
                        * butcher_diagonal_element
                        * delta_t
                        * d_outputs["a_state"]
                    )


def ode_system_solution(time: float, initial_values: np.ndarray):
    """
    Analytical solution to the above ODE system. Expects in order d to a,
    and returns in the same order
    """
    return np.array(
        [
            initial_values[0] * np.exp(time),
            (initial_values[1] - time * initial_values[0]) * np.exp(time),
            (initial_values[2] + time * initial_values[0]) * np.exp(time),
            (initial_values[3] + time * (initial_values[1] + initial_values[2]))
            * np.exp(time),
        ]
    )


class CosPostprocComp(om.ExplicitComponent):
    """To be used in postprocessing. Applies cosine on input quantity."""

    def initialize(self):
        self.options.declare("quantity_name", types=str)

    def setup(self):
        self.add_input(
            self.options["quantity_name"],
            tags=["postproc_input_var", self.options["quantity_name"]],
        )
        self.add_output("cos_" + self.options["quantity_name"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["cos_" + self.options["quantity_name"]] = np.cos(
            inputs[self.options["quantity_name"]]
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        if mode == "fwd":
            d_outputs["cos_" + self.options["quantity_name"]] -= (
                np.sin(inputs[self.options["quantity_name"]])
                * d_inputs[self.options["quantity_name"]]
            )

        elif mode == "rev":
            d_inputs[self.options["quantity_name"]] -= (
                np.sin(inputs[self.options["quantity_name"]])
                * d_outputs["cos_" + self.options["quantity_name"]]
            )


class PostprocSumComp(om.ExplicitComponent):
    """To be used in postprocessing. Sums it's to inputs up."""

    def setup(self):
        self.add_input("cos_b")
        self.add_input("cos_c")
        self.add_output("sum", tags=["postproc_output_var", "sum"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["sum"] = inputs["cos_b"] + inputs["cos_c"]

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        if mode == "fwd":
            if "cos_b" in d_inputs:
                d_outputs["sum"] += d_inputs["cos_b"]
            if "cos_c" in d_inputs:
                d_outputs["sum"] += d_inputs["cos_c"]
        elif mode == "rev":
            if "cos_b" in d_inputs:
                d_inputs["cos_b"] += d_outputs["sum"]
            if "cos_c" in d_inputs:
                d_inputs["cos_c"] += d_outputs["sum"]


def setup_stage_problem_and_integration_control(num_steps, delta_t, butcher_tableau):
    """Setup for the stage problem in the following test cases."""
    prob = om.Problem()
    termination_criterion = TerminationCriterion("num_steps", num_steps)
    integration_control = IntegrationControl(0.0, termination_criterion, delta_t)
    integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[
        -1, -1
    ]
    prob.model.add_subsystem(
        "First", FirstComponent(integration_control=integration_control)
    )
    par_group = om.ParallelGroup()
    group_1 = om.Group()
    ivc21 = om.IndepVarComp()
    ivc21.add_output("c_old")
    ivc21.add_output("c_accumulated_stages")
    group_1.add_subsystem("ivc21", ivc21, promotes=["*"])
    group_1.add_subsystem(
        "comp_21",
        SecondComponent1(integration_control=integration_control),
        promotes=["*"],
    )
    par_group.add_subsystem("Second_1", group_1)

    group_2 = om.Group()
    ivc22 = om.IndepVarComp()
    ivc22.add_output("b_old")
    ivc22.add_output("b_accumulated_stages")
    group_1.add_subsystem("ivc22", ivc22, promotes=["*"])
    group_2.add_subsystem(
        "comp_22",
        SecondComponent2(integration_control=integration_control),
        promotes=["*"],
    )
    par_group.add_subsystem("Second_2", group_2)
    prob.model.add_subsystem("Second", par_group)
    prob.model.add_subsystem(
        "Third", ThirdComponent(integration_control=integration_control)
    )
    prob.model.connect("First.d_state", "Second.Second_1.d")
    prob.model.connect("First.d_state", "Second.Second_2.d")
    prob.model.connect("Second.Second_1.c_state", "Third.c")
    prob.model.connect("Second.Second_2.b_state", "Third.b")

    prob.setup()

    return prob, integration_control


def setup_postprocessing_problem():
    """Setup for the postprocessing problems in the following test cases that use it."""
    postproc_prob = om.Problem()

    c_group = om.Group()
    c_group.add_subsystem(
        "ivc_c",
        om.IndepVarComp("c"),
        promotes=["*"],
    )
    c_group.add_subsystem("cos_c", CosPostprocComp(quantity_name="c"), promotes=["*"])
    b_group = om.Group()
    b_group.add_subsystem(
        "ivc_b",
        om.IndepVarComp("b"),
        promotes=["*"],
    )
    b_group.add_subsystem("cos_b", CosPostprocComp(quantity_name="b"), promotes=["*"])
    par_group = om.ParallelGroup()
    par_group.add_subsystem("c_group", c_group, promotes=["*"])
    par_group.add_subsystem("b_group", b_group, promotes=["*"])
    postproc_prob.model.add_subsystem("par", par_group, promotes=["*"])

    postproc_prob.model.add_subsystem("sum", PostprocSumComp(), promotes=["*"])

    return postproc_prob


def setup_time_integration_problem(
    stage_prob,
    integration_control,
    butcher_tableau,
    test_direction="auto",
    postproc_problem=None,
    postprocessing_quantities=None,
    checkpointing_implementation=NoCheckpointer,
):
    """Setup for the time integrator in the following test cases."""
    if postprocessing_quantities is None:
        postprocessing_quantities = []
    time_integration_prob = om.Problem()
    time_integration_indep = om.IndepVarComp()
    time_integration_indep.add_output("b_initial", shape_by_conn=True, distributed=True)
    time_integration_indep.add_output("c_initial", shape_by_conn=True, distributed=True)
    time_integration_prob.model.add_subsystem(
        "indep", time_integration_indep, promotes=["*"]
    )
    time_integration_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=stage_prob,
            postprocessing_problem=postproc_problem,
            time_integration_quantities=["a", "b", "c", "d"],
            postprocessing_quantities=postprocessing_quantities,
            integration_control=integration_control,
            butcher_tableau=butcher_tableau,
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )
    time_integration_prob.model.nonlinear_solver = om.NonlinearBlockJac()
    time_integration_prob.model.linear_solver = om.LinearBlockGS()
    time_integration_prob.setup(mode=test_direction)
    return time_integration_prob


def set_time_integration_initial_values(
    time_integration_prob: om.Problem, initial_values: list
) -> None:
    """Convenience function for setting the inputs if the time integration."""
    time_integration_prob["d_initial"] = np.array([initial_values[0]])
    if time_integration_prob.comm.rank == 0:
        time_integration_prob["b_initial"] = np.zeros(0)
        time_integration_prob["c_initial"] = np.array([initial_values[1]])
    else:
        time_integration_prob["b_initial"] = np.array([initial_values[2]])
        time_integration_prob["c_initial"] = np.zeros(0)
    time_integration_prob["a_initial"] = np.array([initial_values[3]])


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_three_stage_esdirk]
)
@pytest.mark.parametrize("initial_values", [[1, 1, 1, 1], [0, 0, 0, 0]])
def test_parallel_group_time_integration(
    num_steps: int, butcher_tableau: ButcherTableau, initial_values: list
):
    """Tests the time integration for a problem with parallel groups."""
    delta_t = 0.0001
    prob, integration_control = setup_stage_problem_and_integration_control(
        num_steps, delta_t, butcher_tableau
    )
    prob.run_model()

    time_integration_prob = setup_time_integration_problem(
        prob,
        integration_control,
        butcher_tableau,
    )
    set_time_integration_initial_values(time_integration_prob, initial_values)
    time_integration_prob.run_model()

    result_array = np.array(
        [
            time_integration_prob["d_final"][:],
            time_integration_prob.get_val(
                "c_final" if time_integration_prob.comm.rank == 0 else "b_final",
                get_remote=False,
            )[:],
            time_integration_prob["a_final"][:],
        ]
    ).flatten()
    analytical_result = ode_system_solution(
        num_steps * delta_t, np.array(initial_values)
    )[[0, 1 if time_integration_prob.comm.rank == 0 else 2, 3]]
    assert np.allclose(result_array, analytical_result)


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_three_stage_esdirk]
)
@pytest.mark.parametrize("initial_values", [[1, 1, 1, 1], [0, 0, 0, 0]])
def test_parallel_group_time_integration_with_postprocessing(
    num_steps: int, butcher_tableau: ButcherTableau, initial_values: list
):
    """Tests the postprocessing of the time integration for a problem with parallel
    groups."""
    delta_t = 0.0001
    prob, integration_control = setup_stage_problem_and_integration_control(
        num_steps, delta_t, butcher_tableau
    )
    prob.run_model()

    postproc_prob = setup_postprocessing_problem()

    time_integration_prob = setup_time_integration_problem(
        prob,
        integration_control,
        butcher_tableau,
        postproc_problem=postproc_prob,
        postprocessing_quantities=["sum"],
    )
    set_time_integration_initial_values(time_integration_prob, initial_values)
    time_integration_prob.run_model()

    analytical_result = ode_system_solution(
        num_steps * delta_t, np.array(initial_values)
    )
    postprocessed_analytical_result = np.cos(analytical_result[1]) + np.cos(
        analytical_result[2]
    )
    assert time_integration_prob["sum_final"] == pytest.approx(
        postprocessed_analytical_result
    )


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_three_stage_esdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_parallel_group_time_integration_totals(
    num_steps: int,
    butcher_tableau: ButcherTableau,
    test_direction: str,
    checkpointing_implementation,
):
    """Tests the totals of the time integration for a problem with parallel groups."""
    delta_t = 0.1
    prob, integration_control = setup_stage_problem_and_integration_control(
        num_steps, delta_t, butcher_tableau
    )
    prob.run_model()
    data = prob.check_partials()
    assert_check_partials(data)

    time_integration_prob = setup_time_integration_problem(
        prob,
        integration_control,
        butcher_tableau,
        test_direction,
        checkpointing_implementation=checkpointing_implementation,
    )
    time_integration_prob.run_model()

    # we omit b and c here, since they can't be tested one by one (at least not with
    # functionality already provided by OpenMDAO, and testing them together gives
    # (expectedly) wrong results
    if prob.comm.rank == 0:
        data = time_integration_prob.check_totals(
            ["a_final", "d_final"],
            [
                "a_initial",
                "d_initial",
            ],
        )
    else:
        data = time_integration_prob.check_totals(
            ["a_final", "d_final"],
            [
                "a_initial",
                "d_initial",
            ],
            out_stream=None,
        )
    assert_check_totals(data)


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, embedded_second_order_three_stage_esdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_parallel_group_time_integration_totals_with_postprocessing(
    num_steps: int,
    butcher_tableau: ButcherTableau,
    test_direction: str,
    checkpointing_implementation,
):
    """Tests the totals of the postprocessing after the time integration for a problem
    with parallel groups."""
    delta_t = 0.1
    prob, integration_control = setup_stage_problem_and_integration_control(
        num_steps, delta_t, butcher_tableau
    )
    prob.run_model()
    data = prob.check_partials()
    assert_check_partials(data)

    postproc_prob = setup_postprocessing_problem()

    time_integration_prob = setup_time_integration_problem(
        prob,
        integration_control,
        butcher_tableau,
        test_direction,
        postproc_prob,
        ["sum"],
        checkpointing_implementation,
    )
    time_integration_prob.run_model()

    # we omit b and c, since they can't be tested one by one (at least not with
    # functionality already provided by OpenMDAO, and testing them together gives
    # (expectedly) wrong results
    if prob.comm.rank == 0:
        data = time_integration_prob.check_totals(
            ["sum_final"],
            [
                "a_initial",
                "d_initial",
            ],
        )
    else:
        data = time_integration_prob.check_totals(
            ["sum_final"],
            [
                "a_initial",
                "d_initial",
            ],
            out_stream=None,
        )
    assert_check_totals(data, rtol=1e-4, atol=1e-4)
