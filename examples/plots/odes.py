"""
Usable odes for plotting.
"""

import numpy as np
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl


class ODE(om.ExplicitComponent):
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
        outputs["x_stage"] = (
            0.5 * delta_t * butcher_diagonal_element * stage_time
            + np.sqrt(
                0.25 * delta_t**2 * butcher_diagonal_element**2 * stage_time**2
                + stage_time * (inputs["x"] + delta_t * inputs["acc_stages"])
            )
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

    @staticmethod
    def solution(time, initial_value, initial_time):
        """Analytical solution to the ODE of the above component."""
        return (
            np.sqrt(initial_value) + (np.sqrt(time**3) - np.sqrt(initial_time**3)) / 3
        ) ** 2


class ODE_CFD(om.ExplicitComponent):
    """
    Using ODE from Springer https://doi.org/10.1007/978-3-030-39647-3_36:
    1) x' = lambda * (x - Phi(t)) + dPhi(t)/dt
    2) Phi(t) = sin(t)
    3) x(0) = 1
    Analytical Solution x = sin(t) + e^(lambda*t)
    for lambda = -1.0e+4, x(0) = 1
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("lambda", default=-1e1, types=float)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    @staticmethod
    def phi(time):
        """
        Calculates Phi(t) = sin(t)
        """
        return np.sin(time)

    @staticmethod
    def dphi(time):
        """
        Calculates derivative of Phi'(t)= cos(t)
        """
        return np.cos(time)

    def compute(self, inputs, outputs):
        _delta_t = self.options["integration_control"].delta_t
        stage_time = self.options["integration_control"].stage_time
        lambd = self.options["lambda"]
        outputs["x_stage"] = (
            lambd
            * (inputs["x"] + _delta_t * inputs["acc_stages"] - self.phi(stage_time))
            + self.dphi(stage_time)
        ) / (
            1
            - lambd
            * _delta_t
            * self.options["integration_control"].butcher_diagonal_element
        )

    @staticmethod
    def solution(time, coefficient, initial_value=1.0):
        """Analytical solution of the ODE"""
        return np.sin(time) + initial_value * np.exp(coefficient * time)
