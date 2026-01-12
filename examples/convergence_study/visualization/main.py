from ..utils.constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
)
from plot_log_error import (
    extract_solution_per_butcher_tableau,
    generate_solution_figure,
    generate_global_error_figure,
)
from plot_convergence import plot_convergence

# Plot convergence figure
plot_convergence(BUTCHER_TABLEAUX, PROBLEM)


# Plot solution and error figures
for butcher_tableau in BUTCHER_TABLEAUX.values():
    data_name, adaptive_data, homogeneous_data = extract_solution_per_butcher_tableau(
        butcher_tableau
    )
    # ----------------------------------------------
    # Generate Solution Figure
    generate_solution_figure(butcher_tableau, adaptive_data, data_name)
    # ----------------------------------------------
    # Generate Global Error Figure
    generate_global_error_figure(
        butcher_tableau, adaptive_data, homogeneous_data, data_name
    )
