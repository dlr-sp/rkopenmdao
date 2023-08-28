import numpy as np
import openmdao.api as om
import scipy.stats

import matplotlib.pyplot as plt

from .convergence_test_components import (
    KapsGroup,
    KapsSolution,
    SimpleLinearODE,
    SimpleLinearSolution,
)


from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import RungeKuttaIntegrator
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.butcher_tableaux import (
    explicit_euler,
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
    runge_kutta_four,
    third_order_second_weak_stage_order_four_stage_dirk,
    third_order_third_weak_stage_order_four_stage_dirk,
    fourth_order_third_weak_stage_order_six_stage_dirk,
)

problem_dict = {
    "Simple linear": (SimpleLinearODE, SimpleLinearSolution, ["y"]),
    "Kaps": (KapsGroup, KapsSolution, ["y_1", "y_2"]),
    # "split HeatEquation": TODO,
}

butcher_tableau_dict = {
    # "explicit_euler": explicit_euler,
    "implicit_euler": implicit_euler,
    "implicit_midpoint": implicit_midpoint,
    "second_order_two_stage_sdirk": second_order_two_stage_sdirk,
    "third_order_two_stage_sdirk": third_order_two_stage_sdirk,
    "second_order_three_stage_esdirk": second_order_three_stage_esdirk,
    "third_order_three_stage_esdirk": third_order_three_stage_esdirk,
    "third_order_three_stage_sdirk": third_order_three_stage_sdirk,
    "third_order_four_stage_esdirk": third_order_four_stage_esdirk,
    "third_order_four_stage_sdirk": third_order_four_stage_sdirk,
    "third_order_five_stage_esdirk": third_order_five_stage_esdirk,
    "fourth_order_five_stage_esdirk": fourth_order_five_stage_esdirk,
    "fourth_order_five_stage_sdirk": fourth_order_five_stage_sdirk,
    "fifth_order_five_stage_sdirk": fifth_order_five_stage_sdirk,
    "fourth_order_six_stage_esdirk": fourth_order_six_stage_esdirk,
    "fifth_order_six_stage_esdirk": fifth_order_six_stage_esdirk,
    "third_order_second_weak_stage_order_four_stage_dirk": third_order_second_weak_stage_order_four_stage_dirk,
    "third_order_third_weak_stage_order_four_stage_dirk": third_order_third_weak_stage_order_four_stage_dirk,
    "fourth_order_third_weak_stage_order_six_stage_dirk": fourth_order_third_weak_stage_order_six_stage_dirk,
    # "runge_kutta_four": runge_kutta_four,
}
epsilon_list = [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    0.0,
]

step_nums = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
delta_t = np.array([float(x) ** -1 for x in step_nums])

for problem_name, (Class, Solution, var_list) in problem_dict.items():
    exact_solution = Solution(1.0)
    errors = np.zeros((len(step_nums), len(var_list)))
    for tableau_name, butcher_tableau in butcher_tableau_dict.items():
        for epsilon in epsilon_list:
            if butcher_tableau.butcher_matrix[0, 0] == 0.0 and epsilon == 0.0:
                continue  # can skip esdirk for DAEs, won't work anyway
            for i, num_steps in enumerate(step_nums):
                integration_control = IntegrationControl(0.0, num_steps, delta_t[i])
                time_integration_problem = om.Problem()
                if problem_name == "Kaps":
                    time_integration_problem.model.add_subsystem(
                        "test_comp",
                        Class(integration_control=integration_control, epsilon=epsilon),
                    )
                elif problem_name == "Simple linear":
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
                    for var in time_integration_problem.model._inputs:
                        print(var, time_integration_problem[var])
                    for var in time_integration_problem.model._outputs:
                        print(var, time_integration_problem[var])
                        print(
                            var + "residual",
                            time_integration_problem.model._residuals[var],
                        )
                    continue

                for j, var in enumerate(var_list):
                    errors[i, j] = np.linalg.norm(
                        runge_kutta_problem[var + "_final"] - exact_solution[j]
                    )
            log_errors = np.log10(errors)

            fig, axes = plt.subplots(1, len(var_list), squeeze=False)
            fig.suptitle(
                problem_name
                + ", "
                + tableau_name
                + (f", {epsilon}" if problem_name == "Kaps" else "")
            )
            for i, var in enumerate(var_list):
                axes[0, i].set_xscale("log")
                axes[0, i].set_yscale("log")
                axes[0, i].set_title(var)

                axes[0, i].set_xlabel("delta_t")
                axes[0, i].set_ylabel("L2 error")

                axes[0, i].plot(delta_t, errors[:, i])

                axes[0, i].plot(delta_t, 10 * errors[0, i] * delta_t, "--")
                axes[0, i].plot(delta_t, 100 * errors[0, i] * delta_t**2, "--")
                axes[0, i].plot(delta_t, 1000 * errors[0, i] * delta_t**3, "--")
                axes[0, i].plot(delta_t, 10000 * errors[0, i] * delta_t**4, "--")
                axes[0, i].plot(delta_t, 100000 * errors[0, i] * delta_t**5, "--")

            plt.savefig(
                "convergence_test/"
                + problem_name
                + "_"
                + tableau_name
                + (f"_{epsilon}" if problem_name == "Kaps" else "")
                + ".jpg"
            )

            plt.close(fig)

            if problem_name != "Kaps":
                continue
