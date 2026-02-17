"""
Plot all possible graphs for each scheme:
1. convergence rate,
2. solution graphs and
3. global error graphs.

**TIP: run ./integration_scripts/main.py first.
"""

from .constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
)
from .plot_convergence import generate_convergence_graph, extract_local_error_data
from .plot_errors import (
    extract_solution_per_butcher_tableau,
    generate_solution_figure,
    generate_global_error_figure,
)


# Check whether all files exist, and name the files that do not exist
def check_and_raise_missing_files():
    """Check whether all files exist, and name the files that do not exist"""
    # Group 1: Step-size files (clustered together)
    missing_steps = [
        f"Step-size file for {bt.name} (h={step_size}) missing: {PROBLEM.get_file_path(bt.name, step_size)}"
        for bt in BUTCHER_TABLEAUX.values()
        for step_size in PROBLEM.step_sizes
        if not PROBLEM.file_exists(bt.name, step_size)
    ]

    # Group 2: Homogeneous/adaptive files (clustered together)
    missing_types = [
        f" {_type.capitalize()} file for {bt.name} missing: {PROBLEM.get_file_path(bt.name, _type)}"
        for bt in BUTCHER_TABLEAUX.values()
        for _type in ["avg_homogeneous", "adaptive"]
        if not PROBLEM.file_exists(bt.name, _type)
    ]

    if missing_steps or missing_types:
        error_parts = []
        if missing_steps:
            error_parts.append("MISSING STEP-SIZE FILES:")
            error_parts.extend(missing_steps)
        if missing_types:
            error_parts.append("MISSING SPECIALIZED FILES:")
            error_parts.extend(missing_types)

        error_parts.append("\nHint: Run simulations to generate missing files first.")

        raise FileNotFoundError("\n".join(error_parts))


check_and_raise_missing_files()
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
