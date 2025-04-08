"""
Generates a graph of the delta time over time of different adaptive Runge-Kutta methods.
"""

from mpi4py import MPI
import numpy as np
import openmdao.api as om

from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_sdirk,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)
from rkopenmdao.error_controllers import ppid, integral
from rkopenmdao.error_estimator import ImprovedErrorEstimator
from rkopenmdao.file_writer import TXTFileWriter
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from .odes import ODE_CFD


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

WRITE_FILE = "adaptive"

integration_control = TimeTerminationIntegrationControl(1.0, 10.0, 0.0)
butcher_tableaux = [
    two_stage_sdirk,
    four_stage_esdirk,
    five_stage_esdirk,
]
test_controller = [ppid, integral]
test_estimator = ImprovedErrorEstimator

for butcher_tableau in butcher_tableaux:
    file_name = f"{WRITE_FILE}_{butcher_tableau.name}"
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(",", "")
    file_name = file_name.lower()

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", ODE_CFD(integration_control=integration_control)
    )
    quantities = ["x"]
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller=test_controller,
            error_controller_options={"tol": 1e-12},
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
            write_file=f"{file_name}.h5",
            write_out_distance=1,
        ),
        promotes=["*"],
    )
    runge_kutta_prob.setup()

    for quantity in quantities:
        runge_kutta_prob[quantity + "_initial"] = 1

    runge_kutta_prob.run_model()
