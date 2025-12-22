"""
1. Plots the convergence rate of each scheme.

This requires running the "run_homogenous_problem" file."
"""

import h5py
import matplotlib.pyplot as plt

from ..utils.constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
    MARKER,
    COLORS,
    BUTCHER_NAMES,
)
from ..utils.run_rk_problem import generate_path

local_error_data = {}
for butcher_tableau in BUTCHER_TABLEAUX:
    local_error_data[f"{butcher_tableau.name}"] = {}
    for dt in PROBLEM.step_sizes:

        file_name = (
            f"data_{dt:.0E}_{butcher_tableau.name}".replace(" ", "_")
            .replace(",", "")
            .lower()
        )
        file_path = PROBLEM.folder_path / f"{file_name}.h5"
        with h5py.File(
            file_path,
            mode="r",
        ) as f:
            last_step = int(PROBLEM.time_objective / dt)
            local_error_data[f"{butcher_tableau.name}"][str(dt)] = f["error_measure"][
                str(last_step)
            ][0]

# PLOT LOCAL ERROR DATA

fig, ax = plt.subplots()
ax.set_xlabel(r"$\Delta t$")
ax.set_ylabel(r"Local error measure $\epsilon^l$")
plt.grid(True)
for i, scheme in enumerate(BUTCHER_TABLEAUX):
    p = scheme.p
    ax.loglog(
        PROBLEM.step_sizes,
        local_error_data[scheme.name].values(),
        MARKER[i],
        fillstyle="full",
        lw=2,
        color=COLORS[i],
        label=f"{BUTCHER_NAMES[i]}",
    )
ax.set_xlim(PROBLEM.step_sizes[0], PROBLEM.step_sizes[-1])
ax.legend()

save_file = generate_path(
    str(PROBLEM.folder_path / "plots" / "local_error_time_plot.png")
)
fig.savefig(str(save_file))
