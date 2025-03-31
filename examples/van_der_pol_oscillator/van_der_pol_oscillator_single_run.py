from van_der_pol_oscillator_computation import (
    VanDerPolComponent1,
    VanDerPolComponent2,
    VanDerPolFunctional,
)

import numpy as np
import openmdao.api as om

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from rkopenmdao.butcher_tableaux import third_order_four_stage_esdirk
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer

mpl.rcParams["lines.linewidth"] = 1.0

# mpl.rcParams["markers.fillstyle"] = "none"
mpl.rcParams["legend.numpoints"] = 2
mpl.rcParams["lines.markeredgewidth"] = 0.2
mpl.rcParams["lines.markersize"] = 3.0

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 10

# mpl.rcParams["text.usetex"] = True

mpl.rcParams["xtick.labelsize"] = "large"
mpl.rcParams["ytick.labelsize"] = "large"
mpl.rcParams["axes.labelsize"] = "x-large"
mpl.rcParams["legend.fontsize"] = "x-large"
mpl.rcParams["figure.titlesize"] = "x-large"

SIMULATION_TIME = 10.0
DELTA_T = 0.05
NUM_STEPS = int(SIMULATION_TIME / DELTA_T)
butcher_tableau = third_order_four_stage_esdirk
integration_control = IntegrationControl(0.0, NUM_STEPS, DELTA_T)
vdp_1 = VanDerPolComponent1(integration_control=integration_control)
vdp_2 = VanDerPolComponent2(
    integration_control=integration_control,
    num_parameters=1,
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
    # write_out_distance=1,
    # write_file="data/vdp_single_run.h5",
)

rk_prob = om.Problem()
rk_prob.model.add_subsystem("rk_integration", vdp_rk_integrator, promotes=["*"])

rk_prob.setup()
rk_prob["y1_initial"] = -0.32984621
rk_prob["y2_initial"] = -0.944112
rk_prob["epsilon"] = -4.11990118e-5

rk_prob.run_model()
inputs, outputs, _ = rk_prob.model.get_nonlinear_vectors()
d_inputs, d_outputs, _ = rk_prob.model.get_linear_vectors()
for var in d_inputs:
    d_inputs[var].fill(0.0)
for var in d_outputs:
    d_outputs[var].fill(0.0)

d_outputs["J_final"] = 1.0
vdp_rk_integrator.compute_jacvec_product(inputs, d_inputs, d_outputs, "rev")

for var in d_inputs:
    print(var, d_inputs[var])

# fig, ax = plt.subplots(1, 1)
# with h5py.File(
#     "data/vdp_single_run_0.h5",
#     "r",
# ) as f:
#     y1_arr = np.zeros(len(f["y1"]))
#     y2_arr = np.zeros_like(y1_arr)
#     times_arr = np.zeros_like(y1_arr)
#     for i, val in f["y1"].items():
#         y1_arr[int(i)] = val[0]
#     for i, val in f["y2"].items():
#         y2_arr[int(i)] = val[0]
#         times_arr[int(i)] = val.attrs["time"]
#
# ax.set_xlim(-4.1, 4.1)
# ax.set_ylim(-8.1, 8.1)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.plot(y1_arr, y2_arr, color="black")
# ax.plot(y1_arr[0], y2_arr[0], color="black", marker="x")
# ax.set_title(f"x_0 = {y1_arr[0]}, y_0 = {y2_arr[0]}\n ε = {5.0}")
#
# fig.savefig("vdp_single_run.png")
