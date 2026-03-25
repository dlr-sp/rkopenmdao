"""Components for various ODEs with which the RungeKuttaIntegrator gets tested in
test_component_test.py"""

import numpy as np

from rkopenmdao.components import ExplicitUnsteadyComponent, ImplicitUnsteadyComponent

# pylint: disable=arguments-differ


class TestComp1(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE x' = a * x. The following formula for the stage
    results:
    k_i = b * (x_n + dt * s_i)/(1 - dt * a_ii)
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = (dx_n + dt * ds_i)/(1 - dt * a_ii)
    (rev) dx_n = dk_i / (1 - dt * a_ii)
    (rev) ds_i = dt * dk_i / (1 - dt * a_ii)

    This is the simplest linear ODE (apart from x' = const), so this is the least that
    has to work.
    """

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_input("b", val=1, shape=1, tags=["time_independent_input_var", "b"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            inputs["b"]
            * (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
            / (1 - self.om_data_exchange.step_size * self.om_data_exchange.stage_factor)
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        divisor = (
            1 - self.om_data_exchange.step_size * self.om_data_exchange.stage_factor
        )
        if mode == "fwd":
            d_outputs["x_stage"] += inputs["b"] * d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                inputs["b"]
                * self.om_data_exchange.step_size
                * d_inputs["acc_stages"]
                / divisor
            )
            d_outputs["x_stage"] += (
                (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
                / divisor
                * d_inputs["b"]
            )
        elif mode == "rev":
            d_inputs["x"] += inputs["b"] * d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                inputs["b"]
                * self.om_data_exchange.step_size
                * d_outputs["x_stage"]
                / divisor
            )
            d_inputs["b"] += (
                (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
                / divisor
                * d_outputs["x_stage"]
            )


def solution_test1(time, initial_value, initial_time, param=1.0):
    """Analytical solution to the ODE of the above component."""
    return initial_value * np.exp(param * (time - initial_time))


class TestComp2(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE x' = t. The following formula for the stage
    results:
    k_i = t_n + dt * c_i = t_n^i
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = 0
    (rev) dx_n = 0
    (rev) ds_i = 0

    This ODE could be solved by simple (numerically) integrating, since there is no
    actual dependence between the searched for function and its derivative. However,
    even such a simple case needs to work. This is also the simplest non-autonomous
    test case.
    """

    def setup(self):
        self.add_input("time", shape=1, tags=["time_variable"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = inputs["time"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["time"]
        if mode == "rev":
            d_inputs["time"] += d_outputs["x_stage"]


def solution_test2(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return initial_value + 0.5 * (time**2 - initial_time**2)


class TestComp3(ExplicitUnsteadyComponent):
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

    def setup(self):
        self.add_input("time", shape=1, tags=["time_variable"])
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            inputs["time"]
            * (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
            / (
                1
                - self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * inputs["time"]
            )
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        divisor = (
            1
            - self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            * inputs["time"]
        )
        if mode == "fwd":
            d_outputs["x_stage"] += inputs["time"] * d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                self.om_data_exchange.step_size
                * inputs["time"]
                * d_inputs["acc_stages"]
                / divisor
            )
            d_outputs["x_stage"] += (
                (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
                * d_inputs["time"]
                / divisor**2
            )
        elif mode == "rev":
            d_inputs["x"] += inputs["time"] * d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                self.om_data_exchange.step_size
                * inputs["time"]
                * d_outputs["x_stage"]
                / divisor
            )
            d_inputs["time"] += (
                (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
                * d_outputs["x_stage"]
                / divisor**2
            )


def solution_test3(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return initial_value * np.exp(0.5 * (time**2 - initial_time**2))


class TestComp4(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE system  x' = y, y'=x. The following formula for the
    stage
    results:
    k_i^1 = (x_n^2 + dt * s_i^2 + dt * a_ii * (x_n^1 + dt * s_i^1))
            / (1-dt**2 * a_ii**2)
    k_i^2 = (x_n^1 + dt * s_i^1 + dt * a_ii * (x_n^2 + dt * s_i^2))
            / (1-dt**2 * a_ii**2)
    The formulas for the fwd/rev derivatives are:
    (fwd)
    dk_i^1 = (dx_n^2 + dt * ds_i^2 + dt * a_ii * (dx_n^1 + dt * ds_i^1))
             / (1-dt**2 * a_ii**2)
    dk_i^2 = (dx_n^1 + dt * ds_i^1 + dt * a_ii * (dx_n^2 + dt * ds_i^2))
             /(1-dt**2 * a_ii**2)
    (rev)
    dx_n^1 = ((dt * a_ii) * dk_i^1 + dk_i^2)/(1-dt**2 * a_ii**2)
    ds_i^1 = dt * ((dt * a_ii) * dk_i^1 + dk_i^2)/(1-dt**2 * a_ii**2)
    dx_n^2 = ((dt * a_ii) * dk_i^2 + dk_i^1)/(1-dt**2 * a_ii**2)
    ds_i^2 = dt * ((dt * a_ii) * dk_i^2 + dk_i^1)/(1-dt**2 * a_ii**2)

    This is one of the simplest multi-dimensional ODEs. This is necessary to make sure
    that the RK-Integrator can work with such multidimensional problems.
    """

    def setup(self):
        self.add_input("x", shape=2, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=2, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=2, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        factor = self.om_data_exchange.step_size * self.om_data_exchange.stage_factor
        outputs["x_stage"][0] = (
            factor * inputs["x"][0]
            + inputs["x"][1]
            + self.om_data_exchange.step_size
            * (factor * inputs["acc_stages"][0] + inputs["acc_stages"][1])
        ) / (1 - factor**2)
        outputs["x_stage"][1] = (
            inputs["x"][0]
            + factor * inputs["x"][1]
            + self.om_data_exchange.step_size
            * (inputs["acc_stages"][0] + factor * inputs["acc_stages"][1])
        ) / (1 - factor**2)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        factor = self.om_data_exchange.step_size * self.om_data_exchange.stage_factor

        if mode == "fwd":
            d_outputs["x_stage"][0] += (
                factor * d_inputs["x"][0] + d_inputs["x"][1]
            ) / (1 - factor**2)
            d_outputs["x_stage"][1] += (
                d_inputs["x"][0] + factor * d_inputs["x"][1]
            ) / (1 - factor**2)
            d_outputs["x_stage"][0] += (
                self.om_data_exchange.step_size
                * (factor * d_inputs["acc_stages"][0] + d_inputs["acc_stages"][1])
                / (1 - factor**2)
            )
            d_outputs["x_stage"][1] += (
                self.om_data_exchange.step_size
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
                self.om_data_exchange.step_size
                * (factor * d_outputs["x_stage"][0] + d_outputs["x_stage"][1])
                / (1 - factor**2)
            )
            d_inputs["acc_stages"][1] += (
                self.om_data_exchange.step_size
                * (d_outputs["x_stage"][0] + factor * d_outputs["x_stage"][1])
                / (1 - factor**2)
            )


def solution_test4(time, initial_value, initial_time):
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


class Testcomp51(ExplicitUnsteadyComponent):
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

    def setup(self):
        self.add_input("y", shape=1, tags=["step_input_var", "y"])
        self.add_input("acc_stages_y", shape=1, tags=["accumulated_stage_var", "y"])
        self.add_input("y_stage", shape=1)
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            inputs["y"]
            + self.om_data_exchange.step_size * inputs["acc_stages_y"]
            + self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            * inputs["y_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["y"]
            d_outputs["x_stage"] += (
                self.om_data_exchange.step_size * d_inputs["acc_stages_y"]
            )
            d_outputs["x_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_inputs["y_stage"]
            )
        elif mode == "rev":
            d_inputs["y"] += d_outputs["x_stage"]
            d_inputs["acc_stages_y"] += (
                self.om_data_exchange.step_size * d_outputs["x_stage"]
            )
            d_inputs["y_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_outputs["x_stage"]
            )


class Testcomp52(ExplicitUnsteadyComponent):
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

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages_x", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_input("x_stage", shape=1)
        self.add_output("y_stage", shape=1, tags=["stage_output_var", "y"])

    def compute(self, inputs, outputs):
        outputs["y_stage"] = (
            inputs["x"]
            + self.om_data_exchange.step_size * inputs["acc_stages_x"]
            + self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            * inputs["x_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["y_stage"] += d_inputs["x"]
            d_outputs["y_stage"] += (
                self.om_data_exchange.step_size * d_inputs["acc_stages_x"]
            )
            d_outputs["y_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_inputs["x_stage"]
            )
        elif mode == "rev":
            d_inputs["x"] += d_outputs["y_stage"]
            d_inputs["acc_stages_x"] += (
                self.om_data_exchange.step_size * d_outputs["y_stage"]
            )
            d_inputs["x_stage"] += (
                self.om_data_exchange.step_size
                * self.om_data_exchange.stage_factor
                * d_outputs["y_stage"]
            )


def solution_test5(time, initial_value, initial_time):
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


class TestComp6(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE x' = x**0.5. The following formula for the stage
    results:
    k_i = 0.5 * dt * a_ii + (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**0.5
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = 0.5 * (dx_n + dt * ds_i)
                 * (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**-0.5
    (rev) dx_n = 0.5 * dk_i * (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**-0.5
    (rev) ds_i = 0.5 * dt * dk_i * (0.25 * dt**2 * a_ii**2 + x_n + dt * s_i)**-0.5

    The test cases up until now were all linear. But we also want to use nonlinear ODEs.
    This is a simple test case for that.
    """

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            0.5 * self.om_data_exchange.step_size * self.om_data_exchange.stage_factor
            + np.sqrt(
                0.25
                * self.om_data_exchange.step_size**2
                * self.om_data_exchange.stage_factor**2
                + inputs["x"]
                + self.om_data_exchange.step_size * inputs["acc_stages"]
            )
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        divisor = 2 * np.sqrt(
            0.25
            * self.om_data_exchange.step_size**2
            * self.om_data_exchange.stage_factor**2
            + inputs["x"]
            + self.om_data_exchange.step_size * inputs["acc_stages"]
        )
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                self.om_data_exchange.step_size * d_inputs["acc_stages"] / divisor
            )

        elif mode == "rev":
            d_inputs["x"] += d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                self.om_data_exchange.step_size * d_outputs["x_stage"] / divisor
            )


class TestComp6a(ImplicitUnsteadyComponent):
    """Same as test 6, but as an implicit component with linearize"""

    dr_dx_stage: float
    dr_dx: float
    dr_dacc_stages: float

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        residuals["x_stage"] = (
            outputs["x_stage"]
            - (
                inputs["x"]
                + self.om_data_exchange.step_size
                * (
                    inputs["acc_stages"]
                    + self.om_data_exchange.stage_factor * outputs["x_stage"]
                )
            )
            ** 0.5
        )

    def linearize(
        self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None
    ):
        self.dr_dx_stage = (
            1
            - 0.5
            * self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            / (
                inputs["x"]
                + self.om_data_exchange.step_size
                * (
                    inputs["acc_stages"]
                    + self.om_data_exchange.stage_factor * outputs["x_stage"]
                )
            )
            ** 0.5
        )
        self.dr_dx = (
            -0.5
            / (
                inputs["x"]
                + self.om_data_exchange.step_size
                * (
                    inputs["acc_stages"]
                    + self.om_data_exchange.stage_factor * outputs["x_stage"]
                )
            )
            ** 0.5
        )

        self.dr_dacc_stages = (
            -0.5
            * self.om_data_exchange.step_size
            / (
                inputs["x"]
                + self.om_data_exchange.step_size
                * (
                    inputs["acc_stages"]
                    + self.om_data_exchange.stage_factor * outputs["x_stage"]
                )
            )
            ** 0.5
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode == "fwd":
            d_residuals["x_stage"] += self.dr_dx_stage * d_outputs["x_stage"]
            d_residuals["x_stage"] += self.dr_dx * d_inputs["x"]
            d_residuals["x_stage"] += self.dr_dacc_stages * d_inputs["acc_stages"]
        elif mode == "rev":
            d_outputs["x_stage"] += self.dr_dx_stage * d_residuals["x_stage"]
            d_inputs["x"] += self.dr_dx * d_residuals["x_stage"]
            d_inputs["acc_stages"] += self.dr_dacc_stages * d_residuals["x_stage"]


def solution_test6(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return (0.5 * (time - initial_time) + np.sqrt(initial_value)) ** 2


class TestComp7(ExplicitUnsteadyComponent):
    """
    Models the stage of the ODE x' = (t*x)**0.5. The following formula for the stage
    results:
    k_i = 0.5 * dt * a_ii * t_n^i + (0.25 * dt**2 * a_ii**2 * t_n^i**2 +t_n_i
         * (x_n + dt * s_i))**0.5
    The formulas for the fwd/rev derivatives are:
    (fwd) dk_i = 0.5 * t_n^i * (dx_n + dt * ds_i) * (0.25 * dt**2 * a_ii**2 * t_n^i**2
                 + t_n_i * (x_n + dt * s_i))**-0.5
    (rev) dx_n = 0.5 * t_n^i * dk_i * (0.25 * dt**2 * a_ii**2 * t_n^i**2
                 + t_n_i * (x_n + dt * s_i))**-0.5
    (rev) ds_i = 0.5 * t_n^i * dt * dk_i * (0.25 * dt**2 * a_ii**2 * t_n^i**2
                 + t_n_i * (x_n + dt * s_i))**-0.5

    This is a non-autonomous version of the last nonlinear ODE. with that we also have a
    non-autonomous nonlinear testcase.
    """

    def setup(self):
        self.add_input("time", shape=1, tags=["time_variable"])
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        outputs["x_stage"] = (
            0.5
            * self.om_data_exchange.step_size
            * self.om_data_exchange.stage_factor
            * inputs["time"]
        ) + np.sqrt(
            0.25
            * self.om_data_exchange.step_size**2
            * self.om_data_exchange.stage_factor**2
            * inputs["time"] ** 2
            + inputs["time"]
            * (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        divisor = 2 * np.sqrt(
            0.25
            * self.om_data_exchange.step_size**2
            * self.om_data_exchange.stage_factor**2
            * inputs["time"] ** 2
            + inputs["time"]
            * (inputs["x"] + self.om_data_exchange.step_size * inputs["acc_stages"])
        )
        if mode == "fwd":
            d_outputs["x_stage"] += inputs["time"] * d_inputs["x"] / divisor
            d_outputs["x_stage"] += (
                inputs["time"]
                * self.om_data_exchange.step_size
                * d_inputs["acc_stages"]
                / divisor
            )
            d_outputs["x_stage"] += (
                (
                    0.5
                    * self.om_data_exchange.step_size
                    * self.om_data_exchange.stage_factor
                )
                + (
                    0.5
                    * self.om_data_exchange.step_size**2
                    * self.om_data_exchange.stage_factor**2
                    * inputs["time"]
                    + inputs["x"]
                    + self.om_data_exchange.step_size * inputs["acc_stages"]
                )
                / divisor
            ) * d_inputs["time"]

        elif mode == "rev":
            d_inputs["x"] += inputs["time"] * d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += (
                inputs["time"]
                * self.om_data_exchange.step_size
                * d_outputs["x_stage"]
                / divisor
            )
            d_inputs["time"] += (
                (
                    0.5
                    * self.om_data_exchange.step_size
                    * self.om_data_exchange.stage_factor
                )
                + (
                    0.5
                    * self.om_data_exchange.step_size**2
                    * self.om_data_exchange.stage_factor**2
                    * inputs["time"]
                    + inputs["x"]
                    + self.om_data_exchange.step_size * inputs["acc_stages"]
                )
                / divisor
            ) * d_outputs["x_stage"]


def solution_test7(time, initial_value, initial_time):
    """Analytical solution to the ODE of the above component."""
    return (
        np.sqrt(initial_value) + (np.sqrt(time**3) - np.sqrt(initial_time**3)) / 3
    ) ** 2
