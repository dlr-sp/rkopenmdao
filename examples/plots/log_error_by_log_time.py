"""
Generates a logarithmic graph of error over time for Runge-Kutta methods of different
orders.
"""

import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import openmdao.api as om

from rkopenmdao.integration_control import (
    IntegrationControl,
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    embedded_third_order_four_stage_esdirk as four_stage_dirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
    fifth_order_six_stage_esdirk as six_stage_esdirk,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class ODE(om.ExplicitComponent):
    """
    Using ODE from Springer https://doi.org/10.1007/978-3-030-39647-3_36:
    1) u' = lambda * (u - Phi(t)) + dPhi(t)/dt
    2) Phi(t) = sin(t + pi/4)
    3) u(0) = sin(pi/4)
    Analytical Solution u = sin(t + pi/4) + e^(lambda*t)
    True Solution: , lambda = -10^4, u(0) = sin(pi/4)
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("u", shape=1, tags=["step_input_var", "u"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "u"])
        self.add_input(
            "lambda", val=-1e2, shape=1, tags=["time_independent_input_var", "lambda"]
        )
        self.add_output("u_stage", shape=1, tags=["stage_output_var", "u"])

    @staticmethod
    def phi(time):
        """
        Calculates Phi(t) = sin(t + pi/4)
        """
        return np.sin(np.pi / 4 + time)

    @staticmethod
    def dphi(time):
        """
        Calculates derivative of Phi'(t)= cos(t+pi/4)
        """
        return np.cos(np.pi / 4 + time)

    def compute(self, inputs, outputs):
        _delta_t = self.options["integration_control"].delta_t
        stage_time = self.options["integration_control"].stage_time

        outputs["u_stage"] = (
            inputs["lambda"]
            * (inputs["u"] + _delta_t * inputs["acc_stages"] - self.phi(stage_time))
            + self.dphi(stage_time)
        ) / (
            1
            - inputs["lambda"]
            * _delta_t
            * self.options["integration_control"].butcher_diagonal_element
        )

    @staticmethod
    def solution(time, coefficient):
        """Analytical solution of the ODE"""
        return np.sin(time + np.pi / 4) + np.exp(coefficient * time)


def component_integration(
    component_class, solution, delta_t, butcher_tableau, quantities
):
    """
    Integrates the component with the Runge-kutta Integrator for a given step size
    """
    initial_values = np.array([np.sin(np.pi / 4)])
    integration_control = TimeTerminationIntegrationControl(delta_t, 10.0, 0.0)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", component_class(integration_control=integration_control)
    )
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]

    runge_kutta_prob.run_model()

    result = np.zeros_like(initial_values)
    for i, quantity in enumerate(quantities):
        result[i] = runge_kutta_prob[quantity + "_final"]

    # relatively coarse, but this isn't supposed to test the accuracy,
    # it's just to make sure the solution is in the right region
    return np.abs([solution(10.0, -1e2)] - result)


if __name__ == "__main__":
    delta_t = [1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.5e-1, 1e-1]
    delta_t = np.array(delta_t)
    error_data = {}
    butcher_tableaux = [
        implicit_euler,
        two_stage_dirk,
        four_stage_dirk,
        five_stage_esdirk,
        six_stage_esdirk,
    ]
    for scheme in butcher_tableaux:
        error_data[f"{scheme.name}"] = []
        for step_size in delta_t:
            error_data[f"{scheme.name}"].append(
                component_integration(ODE, ODE.solution, step_size, scheme, ["u"])
            )

    if rank == 0:

        fig = plt.figure()
        # x axis
        plt.xlabel("Step size t [s] (log scale)")
        plt.xscale("log")
        # y axis
        plt.ylabel("Global Error E [-] (log scale)")
        plt.yscale("log")
        plt.grid(True)
        for scheme in butcher_tableaux:
            p = scheme.p
            plt.plot(
                delta_t, error_data[f"{scheme.name}"], lw=2, label=f"{scheme.name}"
            )
            plt.plot(
                delta_t,
                (error_data[f"{scheme.name}"][-1] / delta_t[-1] ** p) * (delta_t) ** p,
                "k--",
                lw=1,
            )

        plt.xlim(delta_t[0], delta_t[-1])
        plt.legend()
        plt.show()
        fig.savefig("error_time_plot.pdf")
