"""
Generates a graph of the delta time over time of adaptive Runge-Kutta method.
"""

import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import openmdao.api as om

from rkopenmdao.butcher_tableaux import (
    embedded_heun_euler,
    embedded_second_order_three_stage_esdirk,
    embedded_third_order_four_stage_esdirk,
    embedded_third_order_five_stage_esdirk,
)
from rkopenmdao.error_controllers import *
from rkopenmdao.error_estimator import *
from rkopenmdao.file_writer import TXTFileWriter
from rkopenmdao.integration_control import (
    IntegrationControl,
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator


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
        print(
            delta_t,
            butcher_diagonal_element,
            stage_time,
            stage_time,
            inputs["acc_stages"],
            inputs["x"],
        )
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

write_out_distance = 1
write_file = "write_main2.txt"
other_file = "write_sub.txt"
test_prob = om.Problem()
integration_control = TimeTerminationIntegrationControl(0.2, 10.0, 1.0)

butcher_tableau = embedded_third_order_four_stage_esdirk

test_prob.model.add_subsystem("test_comp", ODE(integration_control=integration_control))
test_controller = ppid
test_estimator = ImprovedErrorEstimator
time_int_prob = om.Problem()
time_int_prob.model.add_subsystem(
    "rk_integration",
    RungeKuttaIntegrator(
        time_stage_problem=test_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        write_out_distance=write_out_distance,
        write_file=write_file,
        error_controller=test_controller,
        error_controller_options={"tol": 1e-6},
        error_estimator_type=test_estimator,
        file_writing_implementation=TXTFileWriter,
        time_integration_quantities=["x"],
        adaptive_time_stepping=True,
    ),
)

time_int_prob.setup()
time_int_prob.run_model()

if rank == 0:
    with open(write_file) as f:
        lines = [line for line in f]

    time = []
    for line in lines:
        js = json.loads(line)
        time.append(js["time"])

    delta_t = [0] * len(time)
    for i in range(len(time) - 1):
        delta_t[i] = time[i + 1] - time[i]
    delta_t[i + 1] = delta_t[i]

    # Generate Figure
    fig = plt.figure()

    plt.xlabel("Time t [s]")  # time axis (x axis)
    plt.ylabel("dTime t [s]")  # delta time axis (y axis)
    plt.grid(True)
    plt.title(butcher_tableau.name)
    plt.plot(time, delta_t, "--o")
    plt.show()
    printname = butcher_tableau.name
    printname = printname.replace(" ", "_")
    printname = printname.replace(",", "")
    printname = printname.lower()
    fig.savefig(f"time_by_dt_{printname}.pdf")
