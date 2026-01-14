"""
Plot all possible graphs for each scheme:
1. convergence rate,
2. solution graphs and
3. global error graphs.

**TIP: run ./integration_scripts/main.py first.
"""

from ..utils.constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
)
from .plot_convergence import generate_convergence_graph, extract_local_error_data
from .plot_log_error import (
    extract_solution_per_butcher_tableau,
    generate_solution_figure,
    generate_global_error_figure,
)


local_error_data = {}
for butcher_tableau in BUTCHER_TABLEAUX.values():
    # ----------------------------------------------
    # Extract data from hdf5 files
    # ----------------------------------------------
    local_error_data[butcher_tableau.name] = extract_local_error_data(
        butcher_tableau, PROBLEM
    )
    data_name, adaptive_data, homogeneous_data = extract_solution_per_butcher_tableau(
        butcher_tableau
    )

    # ----------------------------------------------
    # Plot solution and error figures
    # ----------------------------------------------
    # Generate Solution Figure
    generate_solution_figure(butcher_tableau, adaptive_data, data_name)

    # Generate Global Error Figure
    generate_global_error_figure(
        butcher_tableau, adaptive_data, homogeneous_data, data_name
    )
# ----------------------------------------------
# Plot convergence figure
# ----------------------------------------------
generate_convergence_graph(local_error_data, BUTCHER_TABLEAUX, PROBLEM)
