import openmdao.api as om
import numpy as np

from runge_kutta_openmdao.heatequation.domain import Domain
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class BoundaryComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("domain", types=Domain)
        self.options.declare("segment", values=["left", "right", "upper", "lower"])
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        domain: Domain = self.options["domain"]
        segment: str = self.options["segment"]
        self.add_input("heat_field", shape=domain.n_x * domain.n_y)
        self.add_output(
            "boundary_value", shape=domain.n_y if segment in ["left", "right"] else domain.n_x
        )

    def compute(self, inputs, outputs):
        domain: Domain = self.options["domain"]
        segment: str = self.options["segment"]
        outputs["boundary_value"] = (-1.0 if segment in ["left", "lower"] else 1.0) * (
            np.sin(self.options["integration_control"].stage_time)
            - np.sin(
                np.pi
                * (
                    np.linspace(domain.y_range[0], domain.y_range[1], domain.n_y)
                    if segment in ["left", "right"]
                    else np.linspace(domain.x_range[0], domain.x_range[1], domain.n_x)
                )
            )
            * (
                np.trapz(
                    np.trapz(
                        inputs["heat_field"].reshape((domain.n_y, domain.n_x)),
                        dx=domain.delta_x,
                        axis=1,
                    ),
                    dx=domain.delta_y,
                    axis=0,
                )
            )
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        domain: Domain = self.options["domain"]
        segment: str = self.options["segment"]
        if mode == "fwd":
            d_outputs["boundary_value"] -= (-1.0 if segment in ["left", "lower"] else 1.0) * (
                np.sin(
                    np.pi
                    * (
                        np.linspace(domain.y_range[0], domain.y_range[1], domain.n_y)
                        if segment in ["left", "right"]
                        else np.linspace(domain.x_range[0], domain.x_range[1], domain.n_x)
                    )
                )
                * (
                    np.trapz(
                        np.trapz(
                            d_inputs["heat_field"].reshape((domain.n_y, domain.n_x)),
                            dx=domain.delta_x,
                            axis=1,
                        ),
                        dx=domain.delta_y,
                        axis=0,
                    )
                )
            )
        elif mode == "rev":
            accumulated_influence = (-1.0 if segment in ["left", "lower"] else 1.0) * np.dot(
                d_outputs["boundary_value"],
                np.sin(
                    np.pi
                    * (
                        np.linspace(domain.y_range[0], domain.y_range[1], domain.n_y)
                        if segment in ["left", "right"]
                        else np.linspace(domain.x_range[0], domain.x_range[1], domain.n_x)
                    )
                ),
            )
            d_inputs["heat_field"] -= domain.delta_x * domain.delta_y * accumulated_influence
            d_inputs["heat_field"][0 : domain.n_x] /= 2
            d_inputs["heat_field"][(domain.n_y - 1) * domain.n_x : domain.n_y * domain.n_x] /= 2
            d_inputs["heat_field"][0 : domain.n_y * domain.n_x : domain.n_x] /= 2
            d_inputs["heat_field"][domain.n_x - 1 : domain.n_y * domain.n_x : domain.n_x] /= 2


if __name__ == "__main__":
    integration_control = IntegrationControl(0.0, 10, 0.1, 2, 2.0, 3.0, 1, 0.2, 1.0)
    prob = om.Problem()
    domain = Domain([0, 1], [0, 1], 11, 11)
    prob.model.add_subsystem(
        "boundary_comp",
        BoundaryComp(domain=domain, segment="lower", integration_control=integration_control),
    )

    prob.setup()
    prob.run_model()

    prob.check_partials()
