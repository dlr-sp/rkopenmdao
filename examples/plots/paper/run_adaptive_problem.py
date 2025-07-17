"""
Generates a graph of the delta time over time of different adaptive Runge-Kutta methods.
"""

from mpi4py import MPI
import numpy as np
import openmdao.api as om

from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as second_order_sdirk,
    embedded_second_order_three_stage_esdirk as second_order_esdirk,
    embedded_third_order_three_stage_sdirk as third_order_sdirk,
    embedded_third_order_four_stage_esdirk as third_order_esdirk,
    embedded_fourth_order_four_stage_sdirk as fourth_order_sdirk,
    embedded_fourth_order_five_stage_esdirk as fourth_order_esdirk,
)
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_estimator import SimpleErrorEstimator, ImprovedErrorEstimator
from rkopenmdao.file_writer import TXTFileWriter
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from .odes import ODE_CFD_REAL


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

WRITE_FILE = "adaptive_test"

integration_control = TimeTerminationIntegrationControl(1.0, 10.0, 0.0)
butcher_tableaux = [
    second_order_sdirk,
    second_order_esdirk,
    third_order_sdirk,
    third_order_esdirk,
    fourth_order_sdirk,
    fourth_order_esdirk,
]
test_controller = [integral]
test_estimator = SimpleErrorEstimator

for butcher_tableau in butcher_tableaux:
    file_name = f"{WRITE_FILE}_{butcher_tableau.name}"
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(",", "")
    file_name = file_name.lower()

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", ODE_CFD_REAL(integration_control=integration_control)
    )

    if butcher_tableau.p < 3:
        print(butcher_tableau.name)
        tol = 1e-6
    else:
        tol = 1e-10
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
            error_controller_options={"tol": tol, "safety_factor": 0.8},
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
            write_file=f"{file_name}.h5",
            write_out_distance=1,
        ),
        promotes=["*"],
    )
    runge_kutta_prob.setup()

    for quantity in quantities:
        runge_kutta_prob[quantity + "_initial"] = np.sin(np.pi / 4)

    runge_kutta_prob.run_model()
