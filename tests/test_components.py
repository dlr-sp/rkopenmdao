"""Components for various ODEs with which the RungeKuttaIntegrator gets tested in test_component_test.py"""
import openmdao.api as om
import numpy as np

from rkopenmdao.integration_control import IntegrationControl

# pylint: disable=arguments-differ


class TestComp1(om.ExplicitComponent):
    """
    Models the stage of the ODE x' = x. The following formula for the stage
    results:
    k_i = (x_n + dt * s_i)/(1 - dt * a_ii)
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = (dx_n + dt * ds_i)/(1 - dt * a_ii)
    (rev) dx_n = dk_i / (1 - dt * a_ii)
    (rev) ds_i = dt * dk_i / (1 - dt * a_ii)

    This is the simplest linear ODE (apart from x' = const), so this is the least that has to work.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        outputs["x_stage"] = (inputs["x"] + delta_t * inputs["acc_stages"]) / (
            1 - delta_t * self.options["integration_control"].butcher_diagonal_element
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        divisor = (
            1 - delta_t * self.options["integration_control"].butcher_diagonal_element
        )
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["x"] / divisor
            d_outputs["x_stage"] += delta_t * d_inputs["acc_stages"] / divisor
        elif mode == "rev":
            d_inputs["x"] += d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += delta_t * d_outputs["x_stage"] / divisor


def Test1Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return initial_value * np.exp(time - initial_time)


class TestComp2(om.ExplicitComponent):
    """
    Models the stage of the ODE x' = t. The following formula for the stage
    results:
    k_i = t_n + dt * c_i = t_n^i
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = 0
    (rev) dx_n = 0
    (rev) ds_i = 0

    This ODE could be solved by simple (numerically) integrating, since there is no actual
    dependence between the searched for function and its derivative. However, even such a
    simple case needs to work. This is also the simplest non-autonomous test case.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        stages_time = self.options["integration_control"].stage_time
        outputs["x_stage"] = stages_time

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        pass


def Test2Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return initial_value + 0.5 * (time**2 - initial_time**2)


class TestComp3(om.ExplicitComponent):
    """
    Models the stage of the ODE x' = t*x. The following formula for the stage
    results:
    k_i = t_n^i * (x_n + dt * s_i) / (1 - dt * a_ii * t_n^i)
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = t_n^i * (dx_n + dt * ds_i) / (1 - dt * a_ii * t_n^i)
    (rev) dx_n = t_n^i * dk_i / (1 - dt * a_ii * t_n^i)
    (rev) ds_i = t_n^i * dt * dk_i / (1 - dt * a_ii * t_n^i)

    This is the simplest non-autonomous case that actually needs an ODE-solver.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        stage_time = self.options["integration_control"].stage_time
        outputs["x_stage"] = (
            stage_time
            * (inputs["x"] + delta_t * inputs["acc_stages"])
            / (
                1
                - delta_t
                * self.options["integration_control"].butcher_diagonal_element
                * stage_time
            )
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        stage_time = self.options["integration_control"].stage_time

        divisor = (
            1
            - delta_t
            * self.options["integration_control"].butcher_diagonal_element
            * stage_time
        )
        if mode == "fwd":
            d_outputs["x_stage"] += stage_time * d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                delta_t * stage_time * d_inputs["acc_stages"] / divisor
            )
        elif mode == "rev":
            d_inputs["x"] += stage_time * d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                delta_t * stage_time * d_outputs["x_stage"] / divisor
            )


def Test3Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return initial_value * np.exp(0.5 * (time**2 - initial_time**2))


class TestComp4(om.ExplicitComponent):
    """
    Models the stage of the ODE system  x' = y, y'=x. The following formula for the stage
    results:
    k_i^1 = (x_n^2 + dt * s_i^2 + dt * a_ii * (x_n^1 + dt * s_i^1)) / (1-dt**2 * a_ii**2)
    k_i^2 = (x_n^1 + dt * s_i^1 + dt * a_ii * (x_n^2 + dt * s_i^2)) / (1-dt**2 * a_ii**2)
    The formulas for the fwd/rev derivatives are:
    (fwd)
    dk_i^1 = (dx_n^2 + dt * ds_i^2 + dt * a_ii * (dx_n^1 + dt * ds_i^1)) / (1-dt**2 * a_ii**2)
    dk_i^2 = (dx_n^1 + dt * ds_i^1 + dt * a_ii * (dx_n^2 + dt * ds_i^2)) / (1-dt**2 * a_ii**2)
    (rev)
    dx_n^1 = ((dt * a_ii) * dk_i^1 + dk_i^2)/(1-dt**2 * a_ii**2)
    ds_i^1 = dt * ((dt * a_ii) * dk_i^1 + dk_i^2)/(1-dt**2 * a_ii**2)
    dx_n^2 = ((dt * a_ii) * dk_i^2 + dk_i^1)/(1-dt**2 * a_ii**2)
    ds_i^2 = dt * ((dt * a_ii) * dk_i^2 + dk_i^1)/(1-dt**2 * a_ii**2)

    This is one of the simplest multi-dimensional ODEs. This is necessary to make sure that
    the RK-Integrator can work with such multidimensional problems.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=2, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=2, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=2, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        factor = delta_t * self.options["integration_control"].butcher_diagonal_element
        outputs["x_stage"][0] = (
            factor * inputs["x"][0]
            + inputs["x"][1]
            + delta_t * (factor * inputs["acc_stages"][0] + inputs["acc_stages"][1])
        ) / (1 - factor**2)
        outputs["x_stage"][1] = (
            inputs["x"][0]
            + factor * inputs["x"][1]
            + delta_t * (inputs["acc_stages"][0] + factor * inputs["acc_stages"][1])
        ) / (1 - factor**2)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        factor = delta_t * self.options["integration_control"].butcher_diagonal_element

        if mode == "fwd":
            d_outputs["x_stage"][0] += (
                factor * d_inputs["x"][0] + d_inputs["x"][1]
            ) / (1 - factor**2)
            d_outputs["x_stage"][1] += (
                d_inputs["x"][0] + factor * d_inputs["x"][1]
            ) / (1 - factor**2)
            d_outputs["x_stage"][0] += (
                delta_t
                * (factor * d_inputs["acc_stages"][0] + d_inputs["acc_stages"][1])
                / (1 - factor**2)
            )
            d_outputs["x_stage"][1] += (
                delta_t
                * (d_inputs["acc_stages"][0] + factor * d_inputs["acc_stages"][1])
                / (1 - factor**2)
            )
        elif mode == "rev":
            d_inputs["x"][0] += (
                factor * d_outputs["x_stage"][0] + d_outputs["x_stage"][1]
            ) / (1 - factor**2)
            d_inputs["x"][1] += (
                d_outputs["x_stage"][0] + factor * d_outputs["x_stage"][1]
            ) / (1 - factor**2)
            d_inputs["acc_stages"][0] += (
                delta_t
                * (factor * d_outputs["x_stage"][0] + d_outputs["x_stage"][1])
                / (1 - factor**2)
            )
            d_inputs["acc_stages"][1] += (
                delta_t
                * (d_outputs["x_stage"][0] + factor * d_outputs["x_stage"][1])
                / (1 - factor**2)
            )


def Test4Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return np.array(
        [
            [
                initial_value[0, 0] * np.cosh(time - initial_time)
                + initial_value[0, 1] * np.sinh(time - initial_time),
                initial_value[0, 0] * np.sinh(time - initial_time)
                + initial_value[0, 1] * np.cosh(time - initial_time),
            ]
        ]
    )


# The next two components also model the ODE system  x' = y, y'=x. However, this time
# the formulas are implemented in 2 components. This can then be used to test that both
# ways (one or two components) work the same.


class TestComp5_1(om.ExplicitComponent):
    """
    Models the stage of the ODE x' = y. The following formula for the stage
    results:
    k_i^1 = x_n^2 + dt * s_i^2 + dt * a_ii * k_i^2
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i^1 = dx_n^2 + dt * ds_i^2 + dt * a_ii * (dk_i^2)
    (rev) dx_n^2 = dk_i^1
    (rev) ds_i^2 = dt * dk_i^1
    (rev) dk_i^2 = dt * a_ii * dk_i^1
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y", shape=1, tags=["step_input_var", "y"])
        self.add_input("acc_stages_y", shape=1, tags=["accumulated_stage_var", "y"])
        self.add_input("y_stage", shape=1)
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["x_stage"] = (
            inputs["y"]
            + delta_t * inputs["acc_stages_y"]
            + delta_t * butcher_diagonal_element * inputs["y_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["y"]
            d_outputs["x_stage"] += delta_t * d_inputs["acc_stages_y"]
            d_outputs["x_stage"] += (
                delta_t * butcher_diagonal_element * d_inputs["y_stage"]
            )
        elif mode == "rev":
            d_inputs["y"] += d_outputs["x_stage"]
            d_inputs["acc_stages_y"] += delta_t * d_outputs["x_stage"]
            d_inputs["y_stage"] += (
                delta_t * butcher_diagonal_element * d_outputs["x_stage"]
            )


class TestComp5_2(om.ExplicitComponent):
    """
    Models the stage of the ODE y' = x. The following formula for the stage
    results:
    k_i^2 = x_n^1 + dt * s_i^1 + dt * a_ii * k_i^1
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i^2 = dx_n^1 + dt * ds_i^1 + dt * a_ii * (dk_i^1)
    (rev) dx_n^1 = dk_i^2
    (rev) ds_i^1 = dt * dk_i^2
    (rev) dk_i^1 = dt * a_ii * dk_i^2
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages_x", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_input("x_stage", shape=1)
        self.add_output("y_stage", shape=1, tags=["stage_output_var", "y"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["y_stage"] = (
            inputs["x"]
            + delta_t * inputs["acc_stages_x"]
            + delta_t * butcher_diagonal_element * inputs["x_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["y_stage"] += d_inputs["x"]
            d_outputs["y_stage"] += delta_t * d_inputs["acc_stages_x"]
            d_outputs["y_stage"] += (
                delta_t * butcher_diagonal_element * d_inputs["x_stage"]
            )
        elif mode == "rev":
            d_inputs["x"] += d_outputs["y_stage"]
            d_inputs["acc_stages_x"] += delta_t * d_outputs["y_stage"]
            d_inputs["x_stage"] += (
                delta_t * butcher_diagonal_element * d_outputs["y_stage"]
            )


def Test5Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above components."""
    return np.array(
        [
            initial_value[0]
            + np.cosh(time - initial_time)
            + initial_value[1]
            + np.sinh(time - initial_time),
            initial_value[0]
            + np.sinh(time - initial_time)
            + initial_value[1]
            + np.cosh(time - initial_time),
        ]
    )


class TestComp6(om.ExplicitComponent):
    """
    Models the stage of the ODE x' = x**0.5. The following formula for the stage
    results:
    k_i = 0.5 * dt * a_ii + (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**0.5
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = 0.5 * (dx_n + dt * ds_i) * (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**-0.5
    (rev) dx_n = 0.5 * dk_i * (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**-0.5
    (rev) ds_i = 0.5 * dt * dk_i * (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**-0.5

    The test cases up until now were all linear. But we also want to use nonlinear ODEs.
    This is a simple test case for that.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["x_stage"] = 0.5 * delta_t * butcher_diagonal_element + np.sqrt(
            0.25 * delta_t**2 * butcher_diagonal_element**2
            + inputs["x"]
            + delta_t * inputs["acc_stages"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        divisor = 2 * np.sqrt(
            0.25 * delta_t**2 * butcher_diagonal_element**2
            + inputs["x"]
            + delta_t * inputs["acc_stages"]
        )
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["x"] / divisor
            d_outputs["x_stage"] += delta_t * d_inputs["acc_stages"] / divisor

        elif mode == "rev":
            d_inputs["x"] += d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += delta_t * d_outputs["x_stage"] / divisor


class TestComp6a(om.ImplicitComponent):
    """Same as test 6, but as an implicit component with linearize"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        residuals["x_stage"] = (
            outputs["x_stage"]
            - (
                inputs["x"]
                + delta_t
                * (inputs["acc_stages"] + butcher_diagonal_element * outputs["x_stage"])
            )
            ** 0.5
        )

    def linearize(
        self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None
    ):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        self.dr_dx_stage = (
            1
            - 0.5
            * delta_t
            * butcher_diagonal_element
            / (
                inputs["x"]
                + delta_t
                * (inputs["acc_stages"] + butcher_diagonal_element * outputs["x_stage"])
            )
            ** 0.5
        )
        self.dr_dx = (
            -0.5
            / (
                inputs["x"]
                + delta_t
                * (inputs["acc_stages"] + butcher_diagonal_element * outputs["x_stage"])
            )
            ** 0.5
        )

        self.dr_dacc_stages = (
            -0.5
            * delta_t
            / (
                inputs["x"]
                + delta_t
                * (inputs["acc_stages"] + butcher_diagonal_element * outputs["x_stage"])
            )
            ** 0.5
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_residuals["x_stage"] += self.dr_dx_stage * d_outputs["x_stage"]
            d_residuals["x_stage"] += self.dr_dx * d_inputs["x"]
            d_residuals["x_stage"] += self.dr_dacc_stages * d_inputs["acc_stages"]
        elif mode == "rev":
            d_outputs["x_stage"] += self.dr_dx_stage * d_residuals["x_stage"]
            d_inputs["x"] += self.dr_dx * d_residuals["x_stage"]
            d_inputs["acc_stages"] += self.dr_dacc_stages * d_residuals["x_stage"]


def Test6Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return (0.5 * (time - initial_time) + np.sqrt(initial_value)) ** 2


class TestComp7(om.ExplicitComponent):
    """
    Models the stage of the ODE x' = (t*x)**0.5. The following formula for the stage
    results:
    k_i = 0.5 * dt * a_ii * t_n^i + (0.25 * dt**2 * a_ii**2 * t_n^i**2 +t_n_i * (x_n + dt * s_i))**0.5
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = 0.5 * t_n^i * (dx_n + dt * ds_i) * (0.25 * dt**2 * a_ii**2 * t_n^i**2 +t_n_i * (x_n + dt * s_i))**-0.5
    (rev) dx_n = 0.5 * t_n^i * dk_i * (0.25 * dt**2 * a_ii**2 * t_n^i**2 +t_n_i * (x_n + dt * s_i))**-0.5
    (rev) ds_i = 0.5 * t_n^i * dt * dk_i * (0.25 * dt**2 * a_ii**2 * t_n^i**2 +t_n_i * (x_n + dt * s_i))**-0.5

    This is a non-autonomous version of the last nonlinear ODE. with that we also have a non-autonomous nonlinear
    testcase.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        stage_time = self.options["integration_control"].stage_time
        outputs[
            "x_stage"
        ] = 0.5 * delta_t * butcher_diagonal_element * stage_time + np.sqrt(
            0.25 * delta_t**2 * butcher_diagonal_element**2 * stage_time**2
            + stage_time * (inputs["x"] + delta_t * inputs["acc_stages"])
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        stage_time = self.options["integration_control"].stage_time
        divisor = 2 * np.sqrt(
            0.25 * delta_t**2 * butcher_diagonal_element**2 * stage_time**2
            + stage_time * (inputs["x"] + delta_t * inputs["acc_stages"])
        )
        if mode == "fwd":
            d_outputs["x_stage"] += stage_time * d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                stage_time * delta_t * d_inputs["acc_stages"] / divisor
            )

        elif mode == "rev":
            d_inputs["x"] += stage_time * d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                stage_time * delta_t * d_outputs["x_stage"] / divisor
            )


def Test7Solution(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return (
        np.sqrt(initial_value) + (np.sqrt(time**3) - np.sqrt(initial_time**3)) / 3
    ) ** 2
