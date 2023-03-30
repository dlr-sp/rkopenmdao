# from typing import List
import openmdao.api as om

from .heatequation import HeatEquation


class HeatEquationStageComponent(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("heat_equation", types=HeatEquation)
        self.options.declare("shared_boundary", types=list, default=[])  # : List[str]

        self.options.declare("delta_t", default=0.01)
        self.options.declare("time", default=0.0)
        self.options.declare("butcher_diagonal_element", default=0.0)
        self.options.declare("step", default=0)
        self.options.declare("stage", default=0)

        self.options.declare("domain_num", types=int)

    def setup(self):
        domain_num = self.options["domain_num"]
        self.add_input(
            "heat",
            shape=self.options["heat_equation"].initial_vector.shape,
            tags=["step_input_var", f"heat_{domain_num}"],
        )
        for segment in self.options["shared_boundary"]:
            self.add_input(
                f"boundary_segment_{segment}",
                shape=self.options["heat_equation"].domain.n_y
                if segment in ["left", "right"]
                else self.options["heat_equation"].domain.n_x,
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
        self.add_output(
            "result_stage_heat",
            shape=self.options["heat_equation"].initial_vector.shape,
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

        heat_equation.inhomogenity_vector.update_boundary_inhomogenity(
            upper=upper, lower=lower, left=left, right=right
        )
        residuals[
            "result_stage_slope"
        ] = heat_equation.heat_equation_time_stage_residual(
            self.options["time"],
            self.options["delta_t"],
            self.options["butcher_diagonal_element"],
            inputs["heat"],
            inputs["accumulated_stages"],
            outputs["result_stage_slope"],
        )
        residuals["result_stage_heat"] = (
            outputs["result_stage_heat"]
            - inputs["heat"]
            - self.options["delta_t"]
            * (
                inputs["accumulated_stages"]
                + self.options["butcher_diagonal_element"]
                * outputs["result_stage_slope"]
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

        heat_equation.inhomogenity_vector.update_boundary_inhomogenity(
            upper=upper, lower=lower, left=left, right=right
        )
        outputs[
            "result_stage_slope"
        ] = heat_equation.solve_heat_equation_time_stage_residual(
            self.options["time"],
            self.options["delta_t"],
            self.options["butcher_diagonal_element"],
            inputs["heat"],
            inputs["accumulated_stages"],
        )
        outputs["result_stage_heat"] = inputs["heat"] + self.options["delta_t"] * (
            inputs["accumulated_stages"]
            + self.options["butcher_diagonal_element"] * outputs["result_stage_slope"]
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        heat_equation: HeatEquation = self.options["heat_equation"]
        if mode == "fwd":
            d_residuals[
                "result_stage_slope"
            ] += heat_equation.d_stage_d_old_value().matvec(d_inputs["heat"])
            d_residuals["result_stage_slope"] += heat_equation.d_stage_d_stage(
                self.options["delta_t"], self.options["butcher_diagonal_element"]
            ).matvec(d_outputs["result_stage_slope"])

            d_residuals[
                "result_stage_slope"
            ] += heat_equation.d_stage_d_accumulated_stages(
                self.options["delta_t"]
            ).matvec(
                d_inputs["accumulated_stages"]
            )
            for segment in self.options["shared_boundary"]:
                delta = (
                    heat_equation.domain.delta_x
                    if segment in ("left", "right")
                    else heat_equation.domain.delta_y
                )
                d_residuals[
                    "result_stage_slope"
                ] += heat_equation.d_stage_d_boundary_segment(delta, segment).matvec(
                    d_inputs[f"boundary_segment_{segment}"]
                )
            d_residuals["result_stage_heat"] += d_outputs["result_stage_heat"]
            d_residuals["result_stage_heat"] -= d_inputs["heat"]
            d_residuals["result_stage_heat"] -= (
                self.options["delta_t"] * d_inputs["accumulated_stages"]
            )
            d_residuals["result_stage_heat"] -= (
                self.options["delta_t"]
                * self.options["butcher_diagonal_element"]
                * d_outputs["result_stage_slope"]
            )

        elif mode == "rev":
            d_inputs["heat"] += heat_equation.d_stage_d_old_value().rmatvec(
                d_residuals["result_stage_slope"]
            )
            d_outputs["result_stage_slope"] += heat_equation.d_stage_d_stage(
                self.options["delta_t"], self.options["butcher_diagonal_element"]
            ).rmatvec(d_residuals["result_stage_slope"])

            d_inputs[
                "accumulated_stages"
            ] += heat_equation.d_stage_d_accumulated_stages(
                self.options["delta_t"]
            ).rmatvec(
                d_residuals["result_stage_slope"]
            )

            for segment in self.options["shared_boundary"]:
                delta = (
                    heat_equation.domain.delta_x
                    if segment in ("left", "right")
                    else heat_equation.domain.delta_y
                )
                d_inputs[
                    f"boundary_segment_{segment}"
                ] += heat_equation.d_stage_d_boundary_segment(delta, segment).rmatvec(
                    d_residuals["result_stage_slope"]
                )

            d_outputs["result_stage_heat"] += d_residuals["result_stage_heat"]
            d_inputs["heat"] -= d_residuals["result_stage_heat"]
            d_inputs["accumulated_stages"] -= (
                self.options["delta_t"] * d_residuals["result_stage_heat"]
            )
            d_outputs["result_stage_slope"] -= (
                self.options["delta_t"]
                * self.options["butcher_diagonal_element"]
                * d_residuals["result_stage_heat"]
            )

    def solve_linear(self, d_outputs, d_residuals, mode):
        heat_equation: HeatEquation = self.options["heat_equation"]
        if mode == "fwd":
            d_outputs["result_stage_slope"] = heat_equation.solve_d_stage_d_stage(
                self.options["delta_t"],
                self.options["butcher_diagonal_element"],
                mode,
                d_residuals["result_stage_slope"],
            )
            d_outputs["result_stage_heat"] = (
                d_residuals["result_stage_heat"]
                + self.options["delta_t"]
                * self.options["butcher_diagonal_element"]
                * d_outputs["result_stage_slope"]
            )

        elif mode == "rev":
            d_residuals["result_stage_heat"] = d_outputs["result_stage_heat"]
            d_residuals["result_stage_slope"] = self.options[
                "heat_equation"
            ].solve_d_stage_d_stage(
                self.options["delta_t"],
                self.options["butcher_diagonal_element"],
                mode,
                d_outputs["result_stage_slope"]
                + self.options["delta_t"]
                * self.options["butcher_diagonal_element"]
                * d_residuals["result_stage_heat"],
            )
