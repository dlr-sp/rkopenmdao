"""Tries out some ODEs with various DIRK schemes to assess their order under different conditions"""

import numpy as np
import openmdao.api as om


import matplotlib.pyplot as plt
import matplotlib as mpl

from rkopenmdao.utils.convergence_test_components import (
    KapsGroup,
    KapsSolution,
    SimpleLinearODE,
    SimpleLinearSolution,
)


from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    implicit_midpoint,
    second_order_two_stage_sdirk,
    third_order_two_stage_sdirk,
    second_order_three_stage_esdirk,
    third_order_three_stage_esdirk,
    third_order_three_stage_sdirk,
    third_order_four_stage_esdirk,
    third_order_four_stage_sdirk,
    third_order_five_stage_esdirk,
    fourth_order_five_stage_esdirk,
    fourth_order_five_stage_sdirk,
    fifth_order_five_stage_sdirk,
    fourth_order_six_stage_esdirk,
    fifth_order_six_stage_esdirk,
    third_order_second_weak_stage_order_four_stage_dirk,
    third_order_third_weak_stage_order_four_stage_dirk,
    fourth_order_third_weak_stage_order_six_stage_dirk,
)


mpl.rcParams["lines.linewidth"] = 1.25

mpl.rcParams["markers.fillstyle"] = "none"
mpl.rcParams["legend.numpoints"] = 2
mpl.rcParams["lines.markeredgewidth"] = 0.5
mpl.rcParams["lines.markersize"] = 6.0

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 12

mpl.rcParams["text.usetex"] = True

mpl.rcParams["xtick.labelsize"] = "medium"
mpl.rcParams["ytick.labelsize"] = "medium"
mpl.rcParams["axes.labelsize"] = "large"
mpl.rcParams["legend.fontsize"] = "medium"
mpl.rcParams["figure.titlesize"] = "large"

problem_dict = {
    # "Simple_linear": (SimpleLinearODE, SimpleLinearSolution, ["y"]),
    "Kaps": (KapsGroup, KapsSolution, ["y_1", "y_2"]),
    # "split HeatEquation": TODO,
}

butcher_tableau_dict = {
    # "explicit_euler": explicit_euler,
    "implicit_euler": implicit_euler,
    "implicit_midpoint": implicit_midpoint,
    "second_order_two_stage_sdirk": second_order_two_stage_sdirk,
    # "third_order_two_stage_sdirk": third_order_two_stage_sdirk,
    # "second_order_three_stage_esdirk": second_order_three_stage_esdirk,
    "third_order_three_stage_esdirk": third_order_three_stage_esdirk,
    "third_order_three_stage_sdirk": third_order_three_stage_sdirk,
    # "third_order_four_stage_esdirk": third_order_four_stage_esdirk,
    # "third_order_four_stage_sdirk": third_order_four_stage_sdirk,
    # "third_order_five_stage_esdirk": third_order_five_stage_esdirk,
    "fourth_order_five_stage_esdirk": fourth_order_five_stage_esdirk,
    "fourth_order_five_stage_sdirk": fourth_order_five_stage_sdirk,
    # "fifth_order_five_stage_sdirk": fifth_order_five_stage_sdirk,
    # "fourth_order_six_stage_esdirk": fourth_order_six_stage_esdirk,
    # "fifth_order_six_stage_esdirk": fifth_order_six_stage_esdirk,
    # "third_order_second_weak_stage_order_four_stage_dirk": third_order_second_weak_stage_order_four_stage_dirk,
    # "third_order_third_weak_stage_order_four_stage_dirk": third_order_third_weak_stage_order_four_stage_dirk,
    # "fourth_order_third_weak_stage_order_six_stage_dirk": fourth_order_third_weak_stage_order_six_stage_dirk,
    # "runge_kutta_four": runge_kutta_four,
}
epsilon_list = [
    1.0,
    # 1e-1,
    # 1e-2,
    1e-3,
    # 1e-4,
    # 1e-5,
    0.0,
]

step_nums = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
delta_t = np.array([float(x) ** -1 for x in step_nums])

for problem_name, (Class, Solution, var_list) in problem_dict.items():
    exact_solution = Solution(1.0)
    errors = np.zeros((len(step_nums), len(var_list), len(butcher_tableau_dict)))
    for epsilon in epsilon_list:
        inclusion_list = []
        for k, (tableau_name, butcher_tableau) in enumerate(
            butcher_tableau_dict.items()
        ):
            if butcher_tableau.butcher_matrix[0, 0] == 0.0 and epsilon == 0.0:
                continue  # can skip esdirk for DAEs, won't work anyway with the current implementation
            else:
                inclusion_list.append(k)
            for i, num_steps in enumerate(step_nums):
                integration_control = IntegrationControl(0.0, num_steps, delta_t[i])
                time_integration_problem = om.Problem()
                if problem_name == "Kaps":
                    time_integration_problem.model.add_subsystem(
                        "test_comp",
                        Class(integration_control=integration_control, epsilon=epsilon),
                    )
                elif problem_name == "Simple_linear":
                    time_integration_problem.model.add_subsystem(
                        "test_comp",
                        Class(integration_control=integration_control),
                    )

                runge_kutta_problem = om.Problem()
                runge_kutta_problem.model.add_subsystem(
                    "rk_integrator",
                    RungeKuttaIntegrator(
                        time_stage_problem=time_integration_problem,
                        butcher_tableau=butcher_tableau,
                        integration_control=integration_control,
                        time_integration_quantities=var_list,
                    ),
                    promotes=["*"],
                )

                runge_kutta_problem.setup()
                for var in var_list:
                    runge_kutta_problem[var + "_initial"].fill(1.0)

                try:
                    runge_kutta_problem.run_model()
                except om.AnalysisError:
                    (
                        inputs,
                        outputs,
                        residuals,
                    ) = time_integration_problem.model.get_nonlinear_vectors()
                    for var in inputs:
                        print(var, time_integration_problem[var])
                    for var in outputs:
                        print(var, time_integration_problem[var])
                        print(
                            var + "residual",
                            residuals[var],
                        )
                    continue

                for j, var in enumerate(var_list):
                    errors[i, j, k] = np.linalg.norm(
                        runge_kutta_problem[var + "_final"] - exact_solution[j]
                    )
        log_errors = np.log10(errors)

        fig, axes = plt.subplots(1, len(var_list), squeeze=False)
        eps_string = r"$\epsilon$"
        fig.suptitle(
            problem_name
            + (f", {eps_string} = {epsilon}" if problem_name == "Kaps" else "")
        )
        for i, var in enumerate(var_list):
            axes[0, i].set_title(var)

            axes[0, i].set_xlabel("step size")
            axes[0, i].set_ylabel("L2 error")
            axes[0, i].set_xlim(min(delta_t), max(delta_t))
            axes[0, i].set_ylim(
                np.min(errors[:, i, inclusion_list]),
                max(np.max(errors[:, i, inclusion_list]), 0.1),
            )

            for k, (tableau_name, butcher_tableau) in enumerate(
                butcher_tableau_dict.items()
            ):
                if butcher_tableau.butcher_matrix[0, 0] != 0.0 or epsilon != 0.0:
                    if i == 0:
                        axes[0, i].loglog(delta_t, errors[:, i, k], label=tableau_name)
                        print(errors[:, i, k])
                    else:
                        axes[0, i].loglog(delta_t, errors[:, i, k])
                        print(errors[:, i, k])

        fig.legend(loc="lower center", ncol=2)

        for i, var in enumerate(var_list):
            axes[0, i].plot(delta_t, delta_t, "k--")
            axes[0, i].plot(delta_t, delta_t**2, "k--")
            axes[0, i].plot(delta_t, delta_t**3, "k--")
            axes[0, i].plot(delta_t, delta_t**4, "k--")
            # axes[0, i].plot(delta_t, 100000 * errors[0, i] * delta_t**5, "--")

            axes[0, i].set_xticks(
                [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09], 8 * [""], minor=True
            )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.35)

        plt.savefig(
            f"convergence_test/{problem_name}{'_' + str(epsilon) if problem_name == 'Kaps' else ''}.pdf"
        )

        plt.close(fig)

        if problem_name != "Kaps":
            continue
