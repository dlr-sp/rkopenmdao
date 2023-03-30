import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation, colors
import h5py

if __name__ == "__main__":
    points_per_direction = 21
    points_x = points_per_direction // 2 + 1

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    x = np.linspace(0, 1.0, points_per_direction)
    y = np.linspace(0, 1.0, points_per_direction)
    x, y = np.meshgrid(x, y)

    x_1 = np.linspace(0, 0.5, points_x)
    y_1 = np.linspace(0, 1, points_per_direction)
    x_1, y_1 = np.meshgrid(x_1, y_1)

    x_2 = np.linspace(0.5, 1.0, points_x)
    y_2 = np.linspace(0, 1, points_per_direction)
    x_2, y_2 = np.meshgrid(x_2, y_2)

    ax1.set_zlim(0.0, 2.0)
    ax2.set_zlim(0.0, 2.0)
    ims = []

    checkpoint_distance = 10

    with h5py.File("inner_problem_stage.h5", mode="r") as f_1, h5py.File(
        "analytic.h5", mode="r"
    ) as f_2, h5py.File("monolithic.h5", mode="r") as f_3:
        for i in range(0, 101):
            heat_analytic = np.array(f_2["heat/" + str(i * checkpoint_distance)])
            heat_monolithic = np.array(f_3["heat/" + str(i * checkpoint_distance)])
            heat_nested_1 = np.array(f_1["heat_1/" + str(i * checkpoint_distance)])
            heat_nested_2 = np.array(f_1["heat_2/" + str(i * checkpoint_distance)])

            im1 = ax1.plot_surface(
                x,
                y,
                heat_analytic.reshape(points_per_direction, points_per_direction),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )
            im2 = ax2.plot_surface(
                x,
                y,
                heat_monolithic.reshape(points_per_direction, points_per_direction),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )

            im3 = ax3.plot_surface(
                x_1,
                y_1,
                heat_nested_1.reshape(points_per_direction, points_x),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )
            im4 = ax3.plot_surface(
                x_2,
                y_2,
                heat_nested_2.reshape(points_per_direction, points_x),
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=+2.0,
                linewidth=0,
                antialiased=False,
            )
            ims.append([im1, im2, im3, im4])

    fig.subplots_adjust(bottom=0.8)
    cax = fig.add_axes([0.1, 0.1, 0.8, 0.1])

    fig.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=2.0), cmap=cm.coolwarm),
        cax=cax,
    )

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=False)
    ani.save("HeatEquOpenMDAO_stage_inner_problem.mp4")
