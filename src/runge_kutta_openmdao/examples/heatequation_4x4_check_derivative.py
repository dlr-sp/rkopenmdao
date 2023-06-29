import numpy as np
import openmdao.api as om

from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau

from runge_kutta_openmdao.heatequation.boundary import BoundaryCondition
from runge_kutta_openmdao.heatequation.domain import Domain
from runge_kutta_openmdao.heatequation.heatequation import HeatEquation
from runge_kutta_openmdao.heatequation.heatequation_stage_component import (
    HeatEquationStageComponent,
)
from runge_kutta_openmdao.heatequation.flux_component import FluxComponent

from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl

from scipy.sparse.linalg import LinearOperator


if __name__ == "__main__":
    total_points_per_direction = 101
    piecewise_points_per_direction = (total_points_per_direction // 4) + 1

    delta_x = 1.0 / (total_points_per_direction - 1)
    domain_dict = {}
    for i in range(4):
        for j in range(4):
            domain_dict[(i, j)] = Domain(
                [0.25 * i, 0.25 * (i + 1)],
                [0.25 * j, 0.25 * (j + 1)],
                piecewise_points_per_direction,
                piecewise_points_per_direction,
            )

    def f(t: float):
        return np.exp(-8 * np.pi**2 * t)

    def f_prime(t: float):
        return -8 * np.pi**2 * g(t)

    def g(x: float):
        return np.cos(2 * np.pi * x)

    def g_prime(x: float):
        return -2 * np.pi * np.sin(2 * np.pi * x)

    def g_prime_prime(x: float):
        return -4 * np.pi**2 * np.cos(2 * np.pi * x)

    boundary_dict = {}
    for i in range(4):
        for j in range(4):
            boundary_dict[(i, j)] = BoundaryCondition(
                left=(lambda t, x, y: -np.sin(np.pi * t) - 1) if i == 0 else None,
                right=(lambda t, x, y: -np.cos(np.pi * t) - 1) if i == 3 else None,
                lower=(lambda t, x, y: 0.0) if j == 0 else None,
                upper=(lambda t, x, y: 0.0) if j == 3 else None,
            )

    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    butcher_tableau = ButcherTableau(
        np.array(
            [
                [gamma, 0.0],
                [1 - gamma, gamma],
            ]
        ),
        np.array([1 - gamma, gamma]),
        np.array([gamma, 1.0]),
    )

    # butcher_tableau = ButcherTableau(
    #     np.array([[0.293, 0.0], [0.414, 0.293]]),
    #     np.array([0.5, 0.5]),
    #     np.array([0.293, 0.707]),
    # )
    # butcher_tableau = ButcherTableau(
    #     np.array([[0.5]]), np.array([1.0]), np.array([0.5])
    # )
    # butcher_tableau = ButcherTableau(
    #     np.array(
    #         [
    #             [0.5, 0.0, 0.0, 0.0],
    #             [0.167, 0.5, 0.0, 0.0],
    #             [-0.5, 0.5, 0.5, 0.0],
    #             [1.5, -1.5, 0.5, 0.5],
    #         ]
    #     ),
    #     np.array([1.5, -1.5, 0.5, 0.5]),
    #     np.array([0.5, 0.667, 0.5, 1.0]),
    # )

    heat_precon = LinearOperator(
        shape=(
            (piecewise_points_per_direction * piecewise_points_per_direction),
            (piecewise_points_per_direction * piecewise_points_per_direction),
        ),
        matvec=lambda x: delta_x**2 / -4 * x,
    )
    heat_equation_dict = {}
    for i in range(4):
        for j in range(4):
            heat_equation_dict[(i, j)] = HeatEquation(
                domain_dict[(i, j)],
                lambda t, x, y: 0.0,
                boundary_dict[(i, j)],
                1.0,
                lambda x, y: 1,
                {"tol": 1e-12, "atol": "legacy", "M": heat_precon},
            )

    integration_control = IntegrationControl(0.0, 1, 1e-5)

    inner_prob = om.Problem()

    heat_group = inner_prob.model.add_subsystem(
        "heat_group", om.ParallelGroup(), min_procs=1, max_procs=16
    )

    for i in range(4):
        for j in range(4):
            shared_boundary = []
            if i != 0:
                shared_boundary.append("left")
            if i != 3:
                shared_boundary.append("right")
            if j != 0:
                shared_boundary.append("lower")
            if j != 3:
                shared_boundary.append("upper")

            heat_subgroup = heat_group.add_subsystem(
                f"heat_subgroup_{i}_{j}", om.Group(), promotes=["*"]
            )

            heat_indep = om.IndepVarComp(f"heat_indep_{i}_{j}")
            heat_indep.add_output(f"heat_{i}_{j}", shape_by_conn=True)

            heat_subgroup.add_subsystem(f"heat_indep_{i}_{j}", heat_indep, promotes=["*"])

            heat_subgroup.add_subsystem(
                f"heat_component_{i}_{j}",
                HeatEquationStageComponent(
                    heat_equation=heat_equation_dict[(i, j)],
                    shared_boundary=shared_boundary,
                    domain_num=4 * i + j,
                    integration_control=integration_control,
                ),
                promotes_inputs=[
                    ("heat", f"heat_{i}_{j}"),
                    ("accumulated_stages", f"accumulated_stages_{i}_{j}"),
                ],
                promotes_outputs=[
                    ("result_stage_heat", f"stage_heat_{i}_{j}"),
                    ("result_stage_slope", f"stage_slope_{i}_{j}"),
                ],
            )

    flux_group = inner_prob.model.add_subsystem(
        "flux_group", om.ParallelGroup(), min_procs=1, max_procs=24
    )

    for i in range(3):
        for j in range(4):
            flux_group.add_subsystem(
                f"flux_comp_{4 *i + j}_{4* (i+1) + j}",
                FluxComponent(
                    delta=delta_x,
                    shape=piecewise_points_per_direction,
                    orientation="vertical",
                ),
            )
            inner_prob.model.connect(
                f"heat_group.stage_heat_{i}_{j}",
                f"flux_group.flux_comp_{4 * i + j}_{4*(i+1) + j}.left_side",
                src_indices=domain_dict[(i, j)].boundary_indices("right") - 1,
            )
            inner_prob.model.connect(
                f"heat_group.stage_heat_{i+1}_{j}",
                f"flux_group.flux_comp_{4 * i + j}_{4*(i+1) + j}.right_side",
                src_indices=domain_dict[(i + 1, j)].boundary_indices("left") + 1,
            )
            inner_prob.model.connect(
                f"flux_group.flux_comp_{4 * i + j}_{4*(i+1) + j}.flux",
                f"heat_group.heat_component_{i}_{j}.boundary_segment_right",
            )
            inner_prob.model.connect(
                f"flux_group.flux_comp_{4 * i + j}_{4*(i+1) + j}.reverse_flux",
                f"heat_group.heat_component_{i+1}_{j}.boundary_segment_left",
            )

    for i in range(4):
        for j in range(3):
            flux_group.add_subsystem(
                f"flux_comp_{4 * i + j}_{4 * i + j + 1}",
                FluxComponent(
                    delta=delta_x,
                    shape=piecewise_points_per_direction,
                    orientation="horizontal",
                ),
            )
            inner_prob.model.connect(
                f"heat_group.stage_heat_{i}_{j}",
                f"flux_group.flux_comp_{4 * i + j}_{4*i + j+1}.lower_side",
                src_indices=domain_dict[(i, j)].boundary_indices("upper")
                - piecewise_points_per_direction,
            )
            inner_prob.model.connect(
                f"heat_group.stage_heat_{i}_{j+1}",
                f"flux_group.flux_comp_{4 * i + j}_{4*i + j+1}.upper_side",
                src_indices=domain_dict[(i, j + 1)].boundary_indices("lower")
                + piecewise_points_per_direction,
            )
            inner_prob.model.connect(
                f"flux_group.flux_comp_{4 * i + j}_{4*i + j+1}.flux",
                f"heat_group.heat_component_{i}_{j}.boundary_segment_upper",
            )
            inner_prob.model.connect(
                f"flux_group.flux_comp_{4 * i + j}_{4*i + j+1}.reverse_flux",
                f"heat_group.heat_component_{i}_{j+1}.boundary_segment_lower",
            )

    # inner_prob.model.nonlinear_solver = om.NonlinearBlockGS()
    newton = inner_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True, iprint=0, maxiter=20, atol=1e-7, rtol=1e-7
    )

    inner_prob.model.linear_solver = om.PETScKrylov(atol=1e-10, rtol=1e-10, ksp_type="gmres")
    # inner_prob.model.linear_solver.precon = om.LinearBlockGS(iprint=-1)
    # inner_prob.model.linear_solver.precon.options["maxiter"] = 2

    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            inner_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_file="dummy.h5",
            quantity_tags=[f"heat_{i}" for i in range(16)],
        ),
        promotes_inputs=["*"],
    )

    # var_comp = om.IndepVarComp("indep")
    # for i in range(4):
    #     for j in range(4):
    #         if MPI:
    #             comm = MPI.COMM_WORLD
    #             rank = comm.rank
    #             sizes, offsets = evenly_distrib_idxs(
    #                 comm.size, heat_equation_dict[(i, j)].initial_vector.size
    #             )
    #         else:
    #             rank = 0
    #             sizes = {rank: heat_equation_dict[(i, j)].initial_vector.size}
    #             offsets = {rank: 0}
    #     var_comp.add_output(
    #         f"heat_{4*i + j}_initial", shape_by_conn=True, distributed=False
    #     )

    # outer_prob.model.add_subsystem("indep", var_comp, promotes=["*"])

    # inner_prob.setup()
    # inner_prob.run_model()
    # metadata = inner_prob.model.get_io_metadata()

    # if inner_prob.comm.rank == 1:
    #     for key, value in metadata.items():
    #         print(key, value)

    outer_prob.setup()
    # for i in range(4):
    #     for j in range(4):
    #         outer_prob.set_val(
    #             f"heat_{4*i + j}_initial",
    #             heat_equation_dict[(i, j)].initial_vector,
    #         )

    outer_prob.run_model()
    print("done running model, starting checking partials")
    outer_prob.check_partials()
