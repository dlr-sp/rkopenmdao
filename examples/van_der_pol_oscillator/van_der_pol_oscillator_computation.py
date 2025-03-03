r"""
Optimization example using RKOpenMDAO on the Van Der Pol oscillator:
https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
The form used is the system of first order ODEs
y_1' = y_2,
y_2' = ε * (1 - y_1**2)*y_2 - y_1.

The functional to be minimized is
J = ∫_0^t_end (y_1(s)**2 + y_2(s)**2 - 1) * tanh(y_1(s)**2 + y_2(s)**2 - 1) ds.
This uses
a) the equation for the circle in its squared form: 1 = y_1(s)**2 + y_2(s)**2,
b) a continuous approximation for the absolute value: |x| ≈ x * tanh(x).

Combining this means that the functional measures the absolute distance between the
trajectory of the ODE system and a unit circle. The optimum would be achieved with
ε ≡ 0, assuming the initial values for the ODE system lie on the unit circle.

This script offers three arguments:
--y1_optimization: If this option is passed, the initial value of y_1 is part of the
    values varied to achieve the optimum of J.
--y2_optimization: If this option is passed, the initial value of y_2 is part of the
    values varied to achieve the optimum of J.
--epsilon_parameters: The number of different epsilons used in the time integrations for
    the optimization. The simulation time will be split into as many equally sized
    intervals, such that there is an interval for each epsilon. Default is 1.
--delta_t: The step size to be used for the time integrations. Default is 0.1,
--epsilon_input_file: Optional file for the initial values of epsilon at the start of
    the optimization. Must be formatted such that it can be read via np.loadtxt into an
    array of size --num_parameters. If not provided, an array filled with 0.3 will be
    used as default.
"""

import argparse

import numpy as np
import openmdao.api as om

from rkopenmdao.butcher_tableaux import third_order_four_stage_esdirk
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer


class VanDerPolComponent1(om.ImplicitComponent):
    """
    First component for the Van der Pol oscillator. Implements
    y_1' = y_2.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y2", val=0.0)
        self.add_input("y1_old", val=2.0, tags=["y1", "step_input_var"])
        self.add_input("y1s_i", val=0.0, tags=["y1", "accumulated_stage_var"])
        self.add_output("y1_update", val=0.0, tags=["y1", "stage_output_var"])
        self.add_output("y1", val=2.0)

    def apply_nonlinear(self, inputs, outputs, residuals, *_args):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        residuals["y1_update"] = outputs["y1_update"] - inputs["y2"]
        residuals["y1"] = (
            outputs["y1"]
            - inputs["y1_old"]
            - delta_t * inputs["y1s_i"]
            - delta_t * butcher_diagonal_element * outputs["y1_update"]
        )

    def solve_nonlinear(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["y1_update"] = inputs["y2"]
        outputs["y1"] = (
            inputs["y1_old"]
            + delta_t * inputs["y1s_i"]
            + delta_t * butcher_diagonal_element * outputs["y1_update"]
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # pylint: disable=too-many-branches
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            if "y1_update" in d_residuals:
                if "y1_update" in d_outputs:
                    d_residuals["y1_update"] += d_outputs["y1_update"]
                if "y2" in d_inputs:
                    d_residuals["y1_update"] -= d_inputs["y2"]
            if "y1" in d_residuals:
                if "y1" in d_outputs:
                    d_residuals["y1"] += d_outputs["y1"]
                if "y1_old" in d_inputs:
                    d_residuals["y1"] -= d_inputs["y1_old"]
                if "y1s_i" in d_inputs:
                    d_residuals["y1"] -= delta_t * d_inputs["y1s_i"]
                if "y1_update" in d_outputs:
                    d_residuals["y1"] -= (
                        butcher_diagonal_element * delta_t * d_outputs["y1_update"]
                    )
        elif mode == "rev":
            if "y1_update" in d_residuals:
                if "y1_update" in d_outputs:
                    d_outputs["y1_update"] += d_residuals["y1_update"]
                if "y2" in d_inputs:
                    d_inputs["y2"] -= d_residuals["y1_update"]
            if "y1" in d_residuals:
                if "y1" in d_outputs:
                    d_outputs["y1"] += d_residuals["y1"]
                if "y1_old" in d_inputs:
                    d_inputs["y1_old"] -= d_residuals["y1"]
                if "y1s_i" in d_inputs:
                    d_inputs["y1s_i"] -= delta_t * d_residuals["y1"]
                if "y1_update" in d_outputs:
                    d_outputs["y1_update"] -= (
                        butcher_diagonal_element * delta_t * d_residuals["y1"]
                    )

    def solve_linear(self, d_outputs, d_residuals, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["y1_update"] = d_residuals["y1_update"]
            d_outputs["y1"] = (
                d_residuals["y1"]
                + delta_t * butcher_diagonal_element * d_residuals["y1_update"]
            )
        elif mode == "rev":
            d_residuals["y1_update"] = (
                d_outputs["y1_update"]
                + delta_t * butcher_diagonal_element * d_outputs["y1"]
            )
            d_residuals["y1"] = d_outputs["y1"]


class VanDerPolComponent2(om.ImplicitComponent):
    """
    First component for the Van der Pol oscillator. Implements
    y_2' = ε * (1 - y_1**2)*y_2 - y_1.
    """

    _inv_jac: float

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare(
            "num_parameters",
            types=int,
            default=1,
            desc="""Defines how many epsilons are available over time. These will be
                    placed equidistantly across time. There won't be any interpolation, 
                    changes will be instantaneous.""",
        )
        self.options.declare("simulation_time", types=float)

    def setup(self):
        self.add_input(
            "epsilon",
            val=np.ones(self.options["num_parameters"]),
            tags=["epsilon", "time_independent_input_var"],
        )
        self.add_input("y1", val=0.0)
        self.add_input("y2_old", val=-2.0 / 3.0, tags=["y2", "step_input_var"])
        self.add_input("y2s_i", val=0.0, tags=["y2", "accumulated_stage_var"])
        self.add_output("y2_update", val=0.0, tags=["y2", "stage_output_var"])
        self.add_output("y2", val=-2.0 / 3.0)

    def apply_nonlinear(self, inputs, outputs, residuals, *_args):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        epsilon = inputs["epsilon"][self._get_epsilon_index()]
        residuals["y2_update"] = (
            outputs["y2_update"]
            - epsilon * (1 - inputs["y1"] ** 2) * outputs["y2"]
            + inputs["y1"]
        )
        residuals["y2"] = (
            outputs["y2"]
            - inputs["y2_old"]
            - delta_t * inputs["y2s_i"]
            - delta_t * butcher_diagonal_element * outputs["y2_update"]
        )

    def solve_nonlinear(self, inputs, outputs, *_args):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        epsilon = inputs["epsilon"][self._get_epsilon_index()]
        divisor = (epsilon - inputs["y1"] ** 2) * delta_t * butcher_diagonal_element - 1
        outputs["y2_update"] = (
            inputs["y1"]
            - epsilon
            * (1 - inputs["y1"] ** 2)
            * (inputs["y2_old"] + delta_t * inputs["y2s_i"])
        ) / divisor

        outputs["y2"] = (
            inputs["y2_old"]
            + delta_t * inputs["y2s_i"]
            + delta_t * butcher_diagonal_element * outputs["y2_update"]
        )

    def linearize(self, inputs, outputs, *_args):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        epsilon = inputs["epsilon"][self._get_epsilon_index()]
        divisor = (
            1 - epsilon * (1 - inputs["y1"] ** 2) * delta_t * butcher_diagonal_element
        )
        self._inv_jac = np.zeros((2, 2))
        self._inv_jac[0, 0] = 1.0
        self._inv_jac[0, 1] = epsilon * (1 - inputs["y1"][0] ** 2)
        self._inv_jac[1, 0] = delta_t * butcher_diagonal_element
        self._inv_jac[1, 1] = 1.0
        self._inv_jac /= divisor

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # pylint: disable=too-many-branches
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        epsilon = inputs["epsilon"][self._get_epsilon_index()]
        if mode == "fwd":
            if "y2_update" in d_residuals:
                if "y2_update" in d_outputs:
                    d_residuals["y2_update"] += d_outputs["y2_update"]
                if "y1" in d_inputs:
                    d_residuals["y2_update"] += (
                        1 + 2 * inputs["y1"] * outputs["y2"] * epsilon
                    ) * d_inputs["y1"]
                if "y2" in d_outputs:
                    d_residuals["y2_update"] -= (
                        epsilon * (1 - inputs["y1"] ** 2) * d_outputs["y2"]
                    )
                if "epsilon" in d_inputs:
                    d_residuals["y2_update"] -= (
                        (1 - inputs["y1"] ** 2)
                        * outputs["y2"]
                        * (d_inputs["epsilon"][self._get_epsilon_index()])
                    )
            if "y2" in d_residuals:
                if "y2" in d_outputs:
                    d_residuals["y2"] += d_outputs["y2"]
                if "y2_old" in d_inputs:
                    d_residuals["y2"] -= d_inputs["y2_old"]
                if "y2s_i" in d_inputs:
                    d_residuals["y2"] -= delta_t * d_inputs["y2s_i"]
                if "y2_update" in d_outputs:
                    d_residuals["y2"] -= (
                        butcher_diagonal_element * delta_t * d_outputs["y2_update"]
                    )
        elif mode == "rev":
            if "y2_update" in d_residuals:
                if "y2_update" in d_outputs:
                    d_outputs["y2_update"] += d_residuals["y2_update"]
                if "y1" in d_inputs:
                    d_inputs["y1"] += (
                        1 + 2 * inputs["y1"] * outputs["y2"] * epsilon
                    ) * d_residuals["y2_update"]
                if "y2" in d_outputs:
                    d_outputs["y2"] -= (
                        epsilon * (1 - inputs["y1"] ** 2) * d_residuals["y2_update"]
                    )
                if "epsilon" in d_inputs:
                    d_inputs["epsilon"][self._get_epsilon_index()] -= (
                        (1 - inputs["y1"] ** 2)
                        * outputs["y2"]
                        * d_residuals["y2_update"]
                    )
            if "y2" in d_residuals:
                if "y2" in d_outputs:
                    d_outputs["y2"] += d_residuals["y2"]
                if "y2_old" in d_inputs:
                    d_inputs["y2_old"] -= d_residuals["y2"]
                if "y2s_i" in d_inputs:
                    d_inputs["y2s_i"] -= delta_t * d_residuals["y2"]
                if "y2_update" in d_outputs:
                    d_outputs["y2_update"] -= (
                        butcher_diagonal_element * delta_t * d_residuals["y2"]
                    )

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            d_outputs["y2_update"] = (
                self._inv_jac[0, 0] * d_residuals["y2_update"]
                + self._inv_jac[0, 1] * d_residuals["y2"]
            )
            d_outputs["y2"] = (
                self._inv_jac[1, 0] * d_residuals["y2_update"]
                + self._inv_jac[1, 1] * d_residuals["y2"]
            )
        elif mode == "rev":
            d_residuals["y2_update"] = (
                self._inv_jac[0, 0] * d_outputs["y2_update"]
                + self._inv_jac[1, 0] * d_outputs["y2"]
            )
            d_residuals["y2"] = (
                self._inv_jac[0, 1] * d_outputs["y2_update"]
                + self._inv_jac[1, 1] * d_outputs["y2"]
            )

    def _get_epsilon_index(self):
        time = self.options["integration_control"].stage_time
        simulation_time = self.options[("simulation_time")]
        num_params = self.options["num_parameters"]
        for i in range(num_params - 1):
            if time < simulation_time / num_params * (i + 1):
                return i
        return num_params - 1


class VanDerPolFunctional(om.ExplicitComponent):
    """
    Functional to be used as example for the optimization with the Van der Pol
    oscillator. Uses a continuous approximation of the absolute distance of a
    curve to the unit circle (using |x| ≈ y* tanh(x))
    """

    def setup(self):
        self.add_input("y1")
        self.add_input("y2")
        self.add_output("J", val=0.0, tags=["J", "stage_output_var"])

    def compute(self, inputs, outputs, *_args):

        outputs["J"] = (inputs["y1"] ** 2 + inputs["y2"] ** 2 - 1) * np.tanh(
            inputs["y1"] ** 2 + inputs["y2"] ** 2 - 1
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, *_args):
        tanh_value = np.tanh(inputs["y1"] ** 2 + inputs["y2"] ** 2 - 1)
        tanh_deriv = tanh_value + (inputs["y1"] ** 2 + inputs["y2"] ** 2 - 1) * (
            1 - tanh_value**2
        )
        if mode == "fwd":
            if "J" in d_outputs:
                if "y1" in d_inputs:
                    d_outputs["J"] += 2 * tanh_deriv * inputs["y1"] * d_inputs["y1"]
                if "y2" in d_inputs:
                    d_outputs["J"] += 2 * tanh_deriv * inputs["y2"] * d_inputs["y2"]
        elif mode == "rev":
            if "J" in d_outputs:
                if "y1" in d_inputs:
                    d_inputs["y1"] += 2 * tanh_deriv * inputs["y1"] * d_outputs["J"]
                if "y2" in d_inputs:
                    d_inputs["y2"] += 2 * tanh_deriv * inputs["y2"] * d_outputs["J"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--y1_optimization", action="store_true")
    parser.add_argument("--y2_optimization", action="store_true")
    parser.add_argument("--epsilon_parameters", type=int, default=1)
    parser.add_argument("--delta_t", type=float, default=0.1)
    parser.add_argument("--epsilon_input_file", type=str, default=None)
    parsed_args = parser.parse_args()

    y1_optimization = bool(parsed_args.y1_optimization)
    y2_optimization = bool(parsed_args.y2_optimization)
    epsilon_parameters = parsed_args.epsilon_parameters
    epsilon_input_file = parsed_args.epsilon_input_file

    SIMULATION_TIME = 10.0
    DELTA_T = parsed_args.delta_t
    NUM_STEPS = int(SIMULATION_TIME / DELTA_T)
    butcher_tableau = third_order_four_stage_esdirk
    integration_control = IntegrationControl(0.0, NUM_STEPS, DELTA_T)
    vdp_1 = VanDerPolComponent1(integration_control=integration_control)
    vdp_2 = VanDerPolComponent2(
        integration_control=integration_control,
        num_parameters=epsilon_parameters,
        simulation_time=SIMULATION_TIME,
    )
    vdp_functional = VanDerPolFunctional()

    vdp_inner_prob = om.Problem()
    vdp_inner_prob.model.add_subsystem("vdp1", vdp_1, promotes=["*"])
    vdp_inner_prob.model.add_subsystem("vdp2", vdp_2, promotes=["*"])
    vdp_inner_prob.model.add_subsystem("vdp_functional", vdp_functional, promotes=["*"])

    vdp_inner_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=False, iprint=2, restart_from_successful=True, maxiter=1000
    )
    vdp_inner_prob.model.linear_solver = om.ScipyKrylov(iprint=2)

    vdp_rk_integrator = RungeKuttaIntegrator(
        time_stage_problem=vdp_inner_prob,
        time_integration_quantities=["y1", "y2", "J"],
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        time_independent_input_quantities=["epsilon"],
        checkpointing_type=PyrevolveCheckpointer,
        write_out_distance=1,
        write_file=f"data/vdp_{DELTA_T}_{epsilon_parameters}_"
        f"y1_{y1_optimization}_y2_{y2_optimization}.h5",
    )

    rk_prob = om.Problem()
    rk_prob.model.add_subsystem("rk_integration", vdp_rk_integrator, promotes=["*"])
    rk_prob.model.add_design_var("epsilon", lower=-10, upper=10)
    if y1_optimization:
        rk_prob.model.add_design_var("y1_initial", lower=-3, upper=3)
    if y2_optimization:
        rk_prob.model.add_design_var("y2_initial", lower=-3, upper=3)
    rk_prob.model.add_objective("J_final")
    rk_prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")

    rk_prob.setup()
    rk_prob["y1_initial"] = 2.0
    rk_prob["y2_initial"] = 2.0

    if epsilon_input_file is not None:
        rk_prob["epsilon"] = np.loadtxt(epsilon_input_file)
    else:
        rk_prob["epsilon"].fill(0.3)

    rk_prob.run_driver()

    if y1_optimization:
        print(rk_prob["y1_initial"])
    if y2_optimization:
        print(rk_prob["y2_initial"])
    print(rk_prob["epsilon"])
    print(rk_prob["J_final"])

    np.savetxt(
        f"data/vdp_epsilon_{DELTA_T}_{epsilon_parameters}"
        f"_y1_{y1_optimization}_y2_{y2_optimization}.txt",
        rk_prob["epsilon"],
    )
