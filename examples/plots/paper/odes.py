"""
Usable odes for plotting.
"""

import numpy as np
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl
from openmdao.utils.array_utils import get_evenly_distributed_size


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
    for lambda = -1.0e+1, x(0) = 1
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


class ODE_CFD_REAL(om.ExplicitComponent):
    """
    Using ODE from Springer https://doi.org/10.1007/978-3-030-39647-3_36:
    1) x' = lambda * (x - Phi(t)) + dPhi(t)/dt
    2) Phi(t) = sin(pi/4+t)
    3) x(0) = sin(pi/4)
    Analytical Solution x = (cos(t) + sin(t)) * 2^-(1/2)
    for lambda = -1.0e+2
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("lambda", default=-1e2, types=float)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    @staticmethod
    def phi(time):
        """
        Calculates Phi(t) = sin(t)
        """
        return np.sin(time + np.pi / 4)

    @staticmethod
    def dphi(time):
        """
        Calculates derivative of Phi'(t)= cos(t)
        """
        return np.cos(time + np.pi / 4)

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
        return (np.sin(time) + np.cos(time)) * 2 ** (-1 / 2)


class ParallelDummy(om.ExplicitComponent):
    """Component that emulates a piece of parallelized code. Given a sequential runtime
    r  (in seconds) and a core count p, runs as long as a perfectly parallelized program
    would, i.e r/p seconds. Expects to be run on p processes. Also accepts a problem
    size s. Depending on the scalinf type (weak/strong), s is divided or multiplied by
    p."""

    par_time: float
    par_size: int
    solution_vector: np.ndarray

    def initialize(self):
        self.options.declare(
            "runtime", types=float, desc="Sequential runtime in seconds"
        )
        self.options.declare("core_count", types=int, desc="Number of cores")
        self.options.declare("size", types=int, desc="Base size of the problem")
        self.options.declare(
            "scaling_type",
            values=["weak", "strong"],
            desc="What kind of scaling is to be investigated.",
        )

    def setup(self):
        if self.options["scaling_type"] == "weak":
            self.par_time = self.options["runtime"]
            self.par_size = self.options["size"]
        else:
            self.par_time = self.options["runtime"] / self.options["core_count"]
            self.par_size = get_evenly_distributed_size(self.comm, self.options["size"])
        self.solution_vector = np.ones(self.par_size)

        self.add_input(
            "x_old",
            tags=["step_input_var", "x"],
            distributed=True,
            val=np.zeros(self.par_size),
        )
        self.add_input(
            "x_acc_stages",
            tags=["accumulated_stage_var", "x"],
            distributed=True,
            val=np.zeros(self.par_size),
        )
        self.add_output(
            "x_stage",
            tags=["stage_output_var", "x"],
            distributed=True,
            val=np.zeros(self.par_size),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # time.sleep(self.par_time)
        outputs["x_stage"] = self.solution_vector.copy()

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        # time.sleep(self.par_time)
        """NO"""


# Van der POL


class VanDerPol1(om.ImplicitComponent):
    """
    Using ODE from Springer https://doi.org/10.1007/978-3-030-39647-3_36:
    1) x' = y
    2) Phi(t) = sin(t)
    3) x(0) = 1
    Analytical Solution x = sin(t) + e^(lambda*t)
    for lambda = -1.0e+1, x(0) = 1
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("lambda", default=-1e1, types=float)

    def setup(self):
        self.add_input("x_old", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])
