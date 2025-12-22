"""Components and analytical solutions for some ODEs used for convergence tests."""

import numpy as np
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl
from .decorators import indexed_static

# pylint: disable=arguments-differ, unused-argument, too-many-branches


class KapsComponent1(om.ImplicitComponent):
    """A component for Kaps problem (see Kennedy, Christopher A. and Mark H. Carpenter.
    “Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A
    Review.” (2016). section 10"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("epsilon", types=float)

    def setup(self):
        self.add_input("y_1_old", val=1.0, tags=["y_1", "step_input_var"])
        self.add_input(
            "y_1_accumulated_stages", val=0.0, tags=["y_1", "accumulated_stage_var"]
        )
        self.add_input("y_2", val=1.0)
        self.add_output("y_1", val=1.0)
        self.add_output("y_1_stage", val=1.0, tags=["y_1", "stage_output_var"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element

        epsilon = self.options["epsilon"]

        residuals["y_1"] = (
            inputs["y_1_old"]
            + delta_t * inputs["y_1_accumulated_stages"]
            + delta_t * butcher_diagonal_element * outputs["y_1_stage"]
            - outputs["y_1"]
        )

        residuals["y_1_stage"] = (
            -(1 + 2 * epsilon) * outputs["y_1"]
            + inputs["y_2"] ** 2
            - epsilon * outputs["y_1_stage"]
        )

    def solve_nonlinear(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element

        epsilon = self.options["epsilon"]

        outputs["y_1_stage"] = (
            -(1 + 2 * epsilon)
            * (inputs["y_1_old"] + delta_t * inputs["y_1_accumulated_stages"])
            + inputs["y_2"] ** 2
        ) / (epsilon + butcher_diagonal_element * delta_t * (1 + 2 * epsilon))

        outputs["y_1"] = inputs["y_1_old"] + delta_t * (
            inputs["y_1_accumulated_stages"]
            + butcher_diagonal_element * outputs["y_1_stage"]
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element

        epsilon = self.options["epsilon"]
        if mode == "fwd":
            if "y_1" in d_residuals:
                if "y_1_old" in d_inputs:
                    d_residuals["y_1"] += d_inputs["y_1_old"]
                if "y_1_accumulated_stages" in d_inputs:
                    d_residuals["y_1"] += delta_t * d_inputs["y_1_accumulated_stages"]
                if "y_1_stage" in d_outputs:
                    d_residuals["y_1"] += (
                        delta_t * butcher_diagonal_element * d_outputs["y_1_stage"]
                    )
                if "y_1" in d_outputs:
                    d_residuals["y_1"] -= d_outputs["y_1"]

            if "y_1_stage" in d_residuals:
                if "y_2" in d_inputs:
                    d_residuals["y_1_stage"] += 2 * inputs["y_2"] * d_inputs["y_2"]
                if "y_1_stage" in d_outputs:
                    d_residuals["y_1_stage"] -= epsilon * d_outputs["y_1_stage"]
                if "y_1" in d_outputs:
                    d_residuals["y_1_stage"] -= (1 + 2 * epsilon) * d_outputs["y_1"]
        elif mode == "rev":
            if "y_1" in d_residuals:
                if "y_1_old" in d_inputs:
                    d_inputs["y_1_old"] += d_residuals["y_1"]
                if "y_1_accumulated_stages" in d_inputs:
                    d_inputs["y_1_accumulated_stages"] += delta_t * d_residuals["y_1"]
                if "y_1_stage" in d_outputs:
                    d_outputs["y_1_stage"] += (
                        delta_t * butcher_diagonal_element * d_residuals["y_1"]
                    )
                if "y_1" in d_outputs:
                    d_outputs["y_1"] -= d_residuals["y_1"]

            if "y_1_stage" in d_residuals:
                if "y_2" in d_inputs:
                    d_inputs["y_2"] += 2 * inputs["y_2"] * d_residuals["y_1_stage"]
                if "y_1_stage" in d_outputs:
                    d_outputs["y_1_stage"] -= epsilon * d_residuals["y_1_stage"]
                if "y_1" in d_outputs:
                    d_outputs["y_1"] -= (1 + 2 * epsilon) * d_residuals["y_1_stage"]

    def solve_linear(self, d_outputs, d_residuals, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element

        epsilon = self.options["epsilon"]

        factor = 1 / (epsilon + (1 + 2 * epsilon) * delta_t * butcher_diagonal_element)
        if mode == "fwd":
            d_outputs["y_1_stage"] = factor * (
                -d_residuals["y_1_stage"] + (1 + 2 * epsilon) * d_residuals["y_1"]
            )
            d_outputs["y_1"] = factor * (
                -delta_t * butcher_diagonal_element * d_residuals["y_1_stage"]
                - epsilon * d_residuals["y_1"]
            )
        elif mode == "rev":
            d_residuals["y_1_stage"] = factor * (
                -d_outputs["y_1_stage"]
                - delta_t * butcher_diagonal_element * d_outputs["y_1"]
            )

            d_residuals["y_1"] = factor * (
                (1 + 2 * epsilon) * d_outputs["y_1_stage"] - epsilon * d_outputs["y_1"]
            )


class KapsComponent2(om.ImplicitComponent):
    """A component for Kaps problem (see Kennedy, Christopher A. and Mark H. Carpenter.
    “Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A
    Review.” (2016). section 10"""

    y_2: float

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y_2_old", val=1.0, tags=["y_2", "step_input_var"])
        self.add_input(
            "y_2_accumulated_stages", val=0.0, tags=["y_2", "accumulated_stage_var"]
        )
        self.add_input("y_1", val=1.0)
        self.add_output("y_2", val=1.0)
        self.add_output("y_2_stage", val=1.0, tags=["y_2", "stage_output_var"])

    def apply_nonlinear(self, inputs, outputs, residuals):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element

        residuals["y_2"] = (
            inputs["y_2_old"]
            + delta_t * inputs["y_2_accumulated_stages"]
            + delta_t * butcher_diagonal_element * outputs["y_2_stage"]
            - outputs["y_2"]
        )

        residuals["y_2_stage"] = (
            inputs["y_1"] - outputs["y_2"] - outputs["y_2"] ** 2 - outputs["y_2_stage"]
        )

    def solve_nonlinear(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if butcher_diagonal_element != 0.0:
            outputs["y_2_stage"] = -(
                delta_t * inputs["y_2_accumulated_stages"]
                + inputs["y_2_old"]
                + (butcher_diagonal_element * delta_t + 1)
                / (2 * butcher_diagonal_element * delta_t)
                - np.sqrt(
                    4 * butcher_diagonal_element**2 * delta_t**2 * inputs["y_1"]
                    + butcher_diagonal_element**2 * delta_t**2
                    + 4
                    * butcher_diagonal_element
                    * delta_t**2
                    * inputs["y_2_accumulated_stages"]
                    + 4 * butcher_diagonal_element * delta_t * inputs["y_2_old"]
                    + 2 * butcher_diagonal_element * delta_t
                    + 1
                )
                / (2 * butcher_diagonal_element * delta_t)
            ) / (butcher_diagonal_element * delta_t)

        else:
            outputs["y_2_stage"] = (
                inputs["y_1"]
                - (inputs["y_2_old"] + delta_t * inputs["y_2_accumulated_stages"])
                - (inputs["y_2_old"] + delta_t * inputs["y_2_accumulated_stages"]) ** 2
            )

        outputs["y_2"] = inputs["y_2_old"] + delta_t * (
            inputs["y_2_accumulated_stages"]
            + butcher_diagonal_element * outputs["y_2_stage"]
        )

    def linearize(self, inputs, outputs, partials):
        self.y_2 = outputs["y_2"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            if "y_2" in d_residuals:
                if "y_2_old" in d_inputs:
                    d_residuals["y_2"] += d_inputs["y_2_old"]
                if "y_2_accumulated_stages" in d_inputs:
                    d_residuals["y_2"] += delta_t * d_inputs["y_2_accumulated_stages"]
                if "y_2_stage" in d_outputs:
                    d_residuals["y_2"] += (
                        delta_t * butcher_diagonal_element * d_outputs["y_2_stage"]
                    )
                if "y_2" in d_outputs:
                    d_residuals["y_2"] -= d_outputs["y_2"]
            if "y_2_stage" in d_residuals:
                if "y_1" in d_inputs:
                    d_residuals["y_2_stage"] += d_inputs["y_1"]
                if "y_2" in d_outputs:
                    d_residuals["y_2_stage"] -= (1 + 2 * outputs["y_2"]) * d_outputs[
                        "y_2"
                    ]
                if "y_2_stage" in d_outputs:
                    d_residuals["y_2_stage"] -= d_outputs["y_2_stage"]
        elif mode == "rev":
            if "y_2" in d_residuals:
                if "y_2_old" in d_inputs:
                    d_inputs["y_2_old"] += d_residuals["y_2"]
                if "y_2_accumulated_stages" in d_inputs:
                    d_inputs["y_2_accumulated_stages"] += delta_t * d_residuals["y_2"]
                if "y_2_stage" in d_outputs:
                    d_outputs["y_2_stage"] += (
                        delta_t * butcher_diagonal_element * d_residuals["y_2"]
                    )
                if "y_2" in d_outputs:
                    d_outputs["y_2"] -= d_residuals["y_2"]
            if "y_2_stage" in d_residuals:
                if "y_1" in d_inputs:
                    d_inputs["y_1"] += d_residuals["y_2_stage"]
                if "y_2" in d_outputs:
                    d_outputs["y_2"] -= (1 + 2 * outputs["y_2"]) * d_residuals[
                        "y_2_stage"
                    ]
                if "y_2_stage" in d_outputs:
                    d_outputs["y_2_stage"] -= d_residuals["y_2_stage"]

    def solve_linear(self, d_outputs, d_residuals, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element

        if mode == "fwd":
            d_outputs["y_2_stage"] = (
                -d_residuals["y_2_stage"]
                + 2 * d_residuals["y_2"] * self.y_2
                + d_residuals["y_2"]
            ) / (
                2 * butcher_diagonal_element * delta_t * self.y_2
                + butcher_diagonal_element * delta_t
                + 1
            )
            d_outputs["y_2"] = (
                -d_residuals["y_2_stage"] * butcher_diagonal_element * delta_t
                - d_residuals["y_2"]
            ) / (
                2 * butcher_diagonal_element * delta_t * self.y_2
                + butcher_diagonal_element * delta_t
                + 1
            )

        elif mode == "rev":
            d_residuals["y_2_stage"] = (
                -butcher_diagonal_element * delta_t * d_outputs["y_2"]
                - d_outputs["y_2_stage"]
            ) / (
                2
                * butcher_diagonal_element
                * delta_t
                * self.y_2
                * butcher_diagonal_element
                * delta_t
                + 1
            )
            d_residuals["y_2"] = (
                2 * d_outputs["y_2_stage"] * self.y_2
                + d_outputs["y_2_stage"]
                - d_outputs["y_2"]
            ) / (
                2
                * butcher_diagonal_element
                * delta_t
                * self.y_2
                * butcher_diagonal_element
                * delta_t
                + 1
            )


class KapsGroup(om.Group):
    """Creates group out of the 2 Kaps components"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("epsilon", types=float)

    def setup(self):
        self.add_subsystem(
            "Kaps1",
            KapsComponent1(
                integration_control=self.options["integration_control"],
                epsilon=self.options["epsilon"],
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            "Kaps2",
            KapsComponent2(integration_control=self.options["integration_control"]),
            promotes=["*"],
        )
        self.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True,
            err_on_non_converge=True,
            iprint=-1,
            maxiter=11,
            max_sub_solves=4,
            atol=1e-7,
            rtol=1e-9,
        )

        self.linear_solver = om.PETScKrylov(
            iprint=-1, restart=10, atol=1e-15, rtol=1e-15
        )
        self.linear_solver.precon = om.LinearBlockJac(iprint=-1, maxiter=1)

    @indexed_static
    def get_initial_values():
        return np.array([1.0, 1.0])


# y' = -y
class SimpleLinearODE(om.ExplicitComponent):
    """Component modelling the ODE y' = -y , y(0) = 1 (by default)"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y_old", val=1.0, tags=["y", "step_input_var"])
        self.add_input(
            "y_accumulated_stages", val=0.0, tags=["y", "accumulated_stage_var"]
        )
        self.add_output("y_stage", val=1.0, tags=["y", "stage_output_var"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["y_stage"] = -(
            inputs["y_old"] + delta_t * inputs["y_accumulated_stages"]
        ) / (1 + delta_t * butcher_diagonal_element)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        factor = (1 + delta_t * butcher_diagonal_element) ** -1
        if mode == "fwd":
            d_outputs["y_stage"] -= factor * d_inputs["y_old"]
            d_outputs["y_stage"] -= delta_t * factor * d_inputs["y_accumulated_stages"]
        elif mode == "rev":
            d_inputs["y_old"] -= factor * d_outputs["y_stage"]
            d_inputs["y_accumulated_stages"] -= delta_t * factor * d_outputs["y_stage"]


def kaps_solution(time):
    "Analytical solution to Kaps problem to compare with the components above."
    return np.array([np.exp(-2 * time), np.exp(-time)])


def simple_linear_solution(time):
    """Analytical solution to y' = -y, y(0) = 1"""
    return np.array([np.exp(-time)])
