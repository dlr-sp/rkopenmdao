import numpy as np
import openmdao.api as om

from rkopenmdao.build.lib.rkopenmdao.odes.prothero_robinson_ode import ProtheroRobinson
from ..integration_control import IntegrationControl


class ProtheroRobinson(om.ExplicitComponent):
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
        self.options.declare("lambda_", default=-1e2, types=float)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    @staticmethod
    def phi(time: float):
        """
        Calculate Phi(t) = sin(t)
        """
        return np.sin(time + np.pi / 4)

    @staticmethod
    def d_phi(time: float):
        """
        Calculate derivative of Phi'(t)= cos(t)
        """
        return np.cos(time + np.pi / 4)

    def compute(self, inputs: dict, outputs: dict):
        _delta_t = self.options["integration_control"].delta_t
        stage_time = self.options["integration_control"].stage_time
        lambda_ = self.options["lambda_"]
        outputs["x_stage"] = (
            lambda_
            * (inputs["x"] + _delta_t * inputs["acc_stages"] - self.phi(stage_time))
            + self.d_phi(stage_time)
        ) / (
            1
            - lambda_
            * _delta_t
            * self.options["integration_control"].butcher_diagonal_element
        )

    @staticmethod
    def get_initial_values():
        """Initial values for the ODE"""
        return np.array([np.sin(np.pi / 4)])

    @staticmethod
    def solution(
        time: float,
        initial_values=ProtheroRobinson.get_initial_values(),
        initial_time=0.0,
    ):
        """Analytical solution of the ODE"""
        return (np.sin(time) + np.cos(time)) * 2 ** (-1 / 2)
