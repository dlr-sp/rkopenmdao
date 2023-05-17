import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation, colors
import h5py

if __name__ == "__main__":
    points_per_direction = 51
    points_x = points_per_direction // 2 + 1

    fig = plt.figure()
    axs = []
    for i in range(1, 4):
        axs.append(fig.add_subplot(2, 3, i + 3 * (i % 2 == 0), projection="3d"))

    x = np.linspace(0, 1.0, points_per_direction)
    y = np.linspace(0, 1.0, points_per_direction)
    x, y = np.meshgrid(x, y)

    x_1 = np.linspace(0, 0.5, points_x)
    y_1 = np.linspace(0, 1, points_per_direction)
    x_1, y_1 = np.meshgrid(x_1, y_1)

    x_2 = np.linspace(0.5, 1.0, points_x)
    y_2 = np.linspace(0, 1, points_per_direction)
    x_2, y_2 = np.meshgrid(x_2, y_2)

    for ax in axs:
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("x")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("y")
        ax.set_zlim(0.0, 2.0)
        ax.set_zlabel("u")

    axs[0].set_title("analytic solution")
    axs[1].set_title("monolithic numerical solution")
    axs[2].set_title("OpenMDAO numerical solution")
    ims = []

    checkpoint_distance = 100
    fig.suptitle(
        r"Newton solver, solve_subsystems=True, $\Delta t = 10^{-5}$, $\Delta x = \Delta y = 0.02$"
    )
    fig.subplots_adjust(bottom=0.20)
    tax = fig.add_axes([0.7, 0.05, 0.25, 0.05])
    tax.axis("off")
    cax = fig.add_axes([0.05, 0.05, 0.6, 0.05])
    fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=2.0), cmap=cm.coolwarm),
        cax=cax,
        location="bottom",
    )
    with h5py.File("inner_problem_stage.h5", mode="r") as f_1, h5py.File(
        "analytic.h5", mode="r"
    ) as f_2, h5py.File("monolithic.h5", mode="r") as f_3:
        for i in range(0, 101):
            heat_analytic = np.array(f_2["heat/" + str(i * checkpoint_distance)])
            heat_monolithic = np.array(f_3["heat/" + str(i * checkpoint_distance)])
            heat_nested_1 = np.array(f_1["heat_1/" + str(i * checkpoint_distance)])
            heat_nested_2 = np.array(f_1["heat_2/" + str(i * checkpoint_distance)])

            im1 = axs[0].plot_surface(
                x,
                y,
                heat_analytic.reshape(points_per_direction, points_per_direction),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )
            im2 = axs[1].plot_surface(
                x,
                y,
                heat_monolithic.reshape(points_per_direction, points_per_direction),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )

            im3 = axs[2].plot_surface(
                x_1,
                y_1,
                heat_nested_1.reshape(points_per_direction, points_x),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )
            im4 = axs[2].plot_surface(
                x_2,
                y_2,
                heat_nested_2.reshape(points_per_direction, points_x),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )
            im5 = tax.text(
                x=0, y=0, s="time: " + str(i * checkpoint_distance * 1e-5)[:5]
            )
            ims.append([im1, im2, im3, im4, im5])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=False)
    ani.save("HeatEquOpenMDAOAnimation.gif")
