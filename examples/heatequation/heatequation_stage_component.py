# pylint: disable=missing-module-docstring
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl

from .heatequation import HeatEquation


class HeatEquationStageComponent(om.ImplicitComponent):
    """Implements the instationary residual of the RK-scheme for a heat equation."""

    def initialize(self):
        self.options.declare("heat_equation", types=HeatEquation)
        self.options.declare("shared_boundary", types=list, default=[])  # : List[str]

        self.options.declare("integration_control", types=IntegrationControl)

        self.options.declare("domain_num", types=int, default=0)

        self.options.declare("enable_scaling", types=bool, default=False)

    def setup(self):
        domain_num = self.options["domain_num"]
        self.add_input(
            "heat",
            val=self.options["heat_equation"].initial_vector,
            shape=self.options["heat_equation"].initial_vector.shape,
            tags=["step_input_var", f"heat_{domain_num}"],
        )

        for segment in self.options["shared_boundary"]:
            self.add_input(
                f"boundary_segment_{segment}",
                shape=(
                    self.options["heat_equation"].domain.n_y
                    if segment in ["left", "right"]
                    else self.options["heat_equation"].domain.n_x
                ),
            )
        self.add_input(
            "accumulated_stages",
            shape=self.options["heat_equation"].initial_vector.shape,
            tags=["accumulated_stage_var", f"heat_{domain_num}"],
        )
        self.add_output(
            "result_stage_slope",
            shape=self.options["heat_equation"].initial_vector.shape,
            tags=["stage_output_var", f"heat_{domain_num}"],
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals
    ):  # pylint: disable = arguments-differ
        heat_equation: HeatEquation = self.options["heat_equation"]
        upper = None
        lower = None
        left = None
        right = None
        for segment in self.options["shared_boundary"]:
            if segment == "upper":
                upper = inputs[f"boundary_segment_{segment}"]

            if segment == "lower":
                lower = inputs[f"boundary_segment_{segment}"]

            if segment == "left":
                left = inputs[f"boundary_segment_{segment}"]

            if segment == "right":
                right = inputs[f"boundary_segment_{segment}"]

        heat_equation.inhomogeneity_vector.update_boundary_inhomogeneity(
            upper=upper, lower=lower, left=left, right=right
        )
        residuals["result_stage_slope"] = (
            heat_equation.heat_equation_time_stage_residual(
                self.options["integration_control"].stage_time,
                self.options["integration_control"].delta_t,
                self.options["integration_control"].butcher_diagonal_element,
                inputs["heat"],
                inputs["accumulated_stages"],
                outputs["result_stage_slope"],
            )
        )

    def solve_nonlinear(self, inputs, outputs):
        heat_equation: HeatEquation = self.options["heat_equation"]
        upper = None
        lower = None
        left = None
        right = None
        for segment in self.options["shared_boundary"]:
            if segment == "upper":
                upper = inputs[f"boundary_segment_{segment}"]

            if segment == "lower":
                lower = inputs[f"boundary_segment_{segment}"]

            if segment == "left":
                left = inputs[f"boundary_segment_{segment}"]

            if segment == "right":
                right = inputs[f"boundary_segment_{segment}"]

        heat_equation.inhomogeneity_vector.update_boundary_inhomogeneity(
            upper=upper, lower=lower, left=left, right=right
        )
        outputs["result_stage_slope"] = (
            heat_equation.solve_heat_equation_time_stage_residual(
                self.options["integration_control"].stage_time,
                self.options["integration_control"].delta_t,
                self.options["integration_control"].butcher_diagonal_element,
                inputs["heat"],
                inputs["accumulated_stages"],
                outputs["result_stage_slope"],
            )
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # pylint: disable=too-many-branches
        # splitting this into function for fwd and reverse wouldn't help too much with
        # readability.
        heat_equation: HeatEquation = self.options["heat_equation"]
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            if "result_stage_slope" in d_residuals:
                if "heat" in d_inputs:
                    d_residuals[
                        "result_stage_slope"
                    ] += heat_equation.d_stage_d_old_value().matvec(d_inputs["heat"])
                if "accumulated_stages" in d_inputs:
                    d_residuals[
                        "result_stage_slope"
                    ] += heat_equation.d_stage_d_accumulated_stages(delta_t).matvec(
                        d_inputs["accumulated_stages"]
                    )

                if "result_stage_slope" in d_outputs:
                    d_residuals["result_stage_slope"] += heat_equation.d_stage_d_stage(
                        delta_t,
                        butcher_diagonal_element,
                    ).matvec(d_outputs["result_stage_slope"])

                for segment in self.options["shared_boundary"]:
                    if f"boundary_segment_{segment}" in d_inputs:
                        delta = (
                            heat_equation.domain.delta_x
                            if segment in ("left", "right")
                            else heat_equation.domain.delta_y
                        )
                        d_residuals[
                            "result_stage_slope"
                        ] += heat_equation.d_stage_d_boundary_segment(
                            delta, segment
                        ).matvec(
                            d_inputs[f"boundary_segment_{segment}"]
                        )

        elif mode == "rev":
            if "result_stage_slope" in d_residuals:
                if "heat" in d_inputs:
                    d_inputs["heat"] += heat_equation.d_stage_d_old_value().rmatvec(
                        d_residuals["result_stage_slope"]
                    )
                if "accumulated_stages" in d_inputs:
                    d_inputs[
                        "accumulated_stages"
                    ] += heat_equation.d_stage_d_accumulated_stages(delta_t).rmatvec(
                        d_residuals["result_stage_slope"]
                    )
                if "result_stage_slope" in d_outputs:
                    d_outputs["result_stage_slope"] += heat_equation.d_stage_d_stage(
                        delta_t,
                        butcher_diagonal_element,
                    ).rmatvec(d_residuals["result_stage_slope"])

                for segment in self.options["shared_boundary"]:
                    if f"boundary_segment_{segment}" in d_inputs:
                        delta = (
                            heat_equation.domain.delta_x
                            if segment in ("left", "right")
                            else heat_equation.domain.delta_y
                        )
                        d_inputs[
                            f"boundary_segment_{segment}"
                        ] += heat_equation.d_stage_d_boundary_segment(
                            delta, segment
                        ).rmatvec(
                            d_residuals["result_stage_slope"]
                        )

    def solve_linear(self, d_outputs, d_residuals, mode):
        heat_equation: HeatEquation = self.options["heat_equation"]
        if mode == "fwd":
            d_outputs["result_stage_slope"] = heat_equation.solve_d_stage_d_stage(
                self.options["integration_control"].delta_t,
                self.options["integration_control"].butcher_diagonal_element,
                mode,
                d_residuals["result_stage_slope"],
                d_outputs["result_stage_slope"],
            )
        elif mode == "rev":
            d_residuals["result_stage_slope"] = self.options[
                "heat_equation"
            ].solve_d_stage_d_stage(
                self.options["integration_control"].delta_t,
                self.options["integration_control"].butcher_diagonal_element,
                mode,
                d_outputs["result_stage_slope"],
                d_residuals["result_stage_slope"],
            )
