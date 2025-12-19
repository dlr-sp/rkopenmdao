import h5py
import matplotlib.pyplot as plt

from rkopenmdao.examples.toy_problems.utils.constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
    MARKER,
    COLORS,
    BUTCHER_NAMES,
)


if __name__ == "__main__":
    local_error_data = {}
    for butcher_tableau in BUTCHER_TABLEAUX:
        local_error_data[f"{butcher_tableau.name}"] = {}
        for dt in PROBLEM.delta_t:
            local_error_data[f"{butcher_tableau.name}"][str(dt)] = []

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
                local_error_data[butcher_tableau.name][str(dt)].append(
                    f["Norm"][str(last_step)][0]
                )

    # PLOT LOCAL ERROR DATA

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$L_2$ norm local error $\epsilon^l$")
    plt.grid(True)
    for i, scheme in enumerate(BUTCHER_TABLEAUX):
        p = scheme.p
        ax.loglog(
            PROBLEM.delta_t,
            local_error_data[scheme.name].values(),
            MARKER[i],
            fillstyle="full",
            lw=2,
            color=COLORS[i],
            label=f"{BUTCHER_NAMES[i]}",
        )
    # Asymptote set local_error_data[scheme.name][j][%SET_HERE%] a string of the smallest time step size
    # plt.loglog(
    #     DELTA_T_LIST,
    #     (local_error_data[scheme.name]["0.01"] / DELTA_T_LIST[0] ** p)
    #     * (DELTA_T_LIST) ** p,
    #    "k--",
    #     lw=1,
    # )
    ax.set_xlim(PROBLEM.delta_t[0], PROBLEM.delta_t[-1])
    ax.legend()

    save_file = PROBLEM.folder_path / "local_error_time_plot.png"
    fig.savefig(save_file)
