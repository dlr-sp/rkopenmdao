import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals, assert_check_partials
import pytest
import numpy as np

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    third_order_four_stage_esdirk,
    implicit_euler,
    explicit_euler,
)
from rkopenmdao.butcher_tableau import ButcherTableau


class FirstComponent(om.ExplicitComponent):
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


@pytest.mark.mpi
@pytest.mark.parametrize("num_steps", [1, 10])
@pytest.mark.parametrize(
    "butcher_tableau", [explicit_euler, implicit_euler, third_order_four_stage_esdirk]
)
@pytest.mark.parametrize("test_direction", ["fwd", "rev"])
def test_parallel_group_time_integration(
    num_steps: int, butcher_tableau: ButcherTableau, test_direction: str
):
    prob = om.Problem()
    integration_control = IntegrationControl(0.0, num_steps, 0.1)
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
    prob.run_model()
    data = prob.check_partials()
    assert_check_partials(data)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=prob,
            time_integration_quantities=["a", "b" if prob.comm.rank == 1 else "c", "d"],
            integration_control=integration_control,
            butcher_tableau=butcher_tableau,
        ),
        promotes=["*"],
    )
    time_integration_prob.model.nonlinear_solver = om.NonlinearBlockJac()
    time_integration_prob.model.linear_solver = om.LinearBlockGS()
    time_integration_prob.setup(mode=test_direction)
    time_integration_prob.run_model()

    # we test b and c later, since they can't be tested one by one (at least not with functionality already provided by
    # OpenMDAO, and testing them together gives (expected) wrong results
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

    # TODO: maybe move this to a different test

    # time_integration_prob.model._doutputs.set_val(0.0)
    # time_integration_prob.model._dresiduals.set_val(0.0)
    # if time_integration_prob.comm.rank == 0:
    #     time_integration_prob.model._doutputs[
    #         time_integration_prob.model.get_source("c_initial")
    #     ] = 1.0
    # time_integration_prob.model.run_apply_linear(mode="fwd")
    # db_fin = np.zeros(1)
    # fwd_res = 0
    # if time_integration_prob.comm.rank == 0:
    #     time_integration_prob.comm.Recv(db_fin, source=1, tag=0)
    #     fwd_res = (
    #         time_integration_prob.model._dresiduals["a_final"]
    #         + db_fin
    #         + time_integration_prob.model._dresiduals["c_final"]
    #         + time_integration_prob.model._dresiduals["d_final"]
    #     )
    # else:
    #     db_fin = time_integration_prob.model._dresiduals["b_final"]
    #     time_integration_prob.comm.Send(db_fin, dest=0, tag=0)
    #
    # time_integration_prob.model._doutputs.set_val(0.0)
    # time_integration_prob.model._dresiduals.set_val(0.0)
    #
    # if time_integration_prob.comm.rank == 0:
    #     time_integration_prob.model._dresiduals["a_final"] = 1.0
    #     time_integration_prob.model._dresiduals["c_final"] = 1.0
    #     time_integration_prob.model._dresiduals["d_final"] = 1.0
    # elif time_integration_prob.comm.rank == 1:
    #     time_integration_prob.model._dresiduals["b_final"] = 1.0
    #
    # time_integration_prob.model.run_apply_linear(mode="rev")
    # if time_integration_prob.comm.rank == 0:
    #     rev_res = time_integration_prob.model._doutputs[
    #         time_integration_prob.model.get_source("c_initial")
    #     ]
    #     assert fwd_res == pytest.approx(rev_res)
