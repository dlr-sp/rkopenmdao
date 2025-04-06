"""
Generates a graph of the delta time over time of different adaptive Runge-Kutta methods.
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
from .odes import ODE


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

WRITE_OUT_DISTANCE = 1
WRITE_FILE = "write_main2.txt"
test_prob = om.Problem()
integration_control = TimeTerminationIntegrationControl(0.2, 10.0, 1.0)

butcher_tableau = embedded_third_order_four_stage_esdirk

test_prob.model.add_subsystem("test_comp", ODE(integration_control=integration_control))
test_controller = [ppid, integral]
test_estimator = ImprovedErrorEstimator
time_int_prob = om.Problem()
time_int_prob.model.add_subsystem(
    "rk_integration",
    RungeKuttaIntegrator(
        time_stage_problem=test_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        write_out_distance=WRITE_OUT_DISTANCE,
        write_file=WRITE_FILE,
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
