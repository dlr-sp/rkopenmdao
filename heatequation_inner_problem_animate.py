import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import h5py

if __name__ == "__main__":
    points_x = 6
    points_per_direction = 11

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    x_1 = np.linspace(0, 0.5, points_x)
    y_1 = np.linspace(0, 1, points_per_direction)
    x_1, y_1 = np.meshgrid(x_1, y_1)

    x_2 = np.linspace(0.5, 1.0, points_x)
    y_2 = np.linspace(0, 1, points_per_direction)
    x_2, y_2 = np.meshgrid(x_2, y_2)

    ax1.set_zlim(-1.0, 1.0)
    ax2.set_zlim(-1.0, 1.0)
    ims = []

    with h5py.File("inner_problem_stage.h5", mode="r") as f:
        for i in range(0, 1001, 10):
            heat_1 = np.array(f["heat_1/" + str(i)])
            heat_2 = np.array(f["heat_2/" + str(i)])
            im1 = ax1.plot_surface(
                x_1,
                y_1,
                heat_1.reshape(points_per_direction, points_x),
                cmap=cm.coolwarm,
                vmin=-1.0,
                vmax=+1.0,
                linewidth=0,
                antialiased=False,
            )
            im2 = ax2.plot_surface(
                x_2,
                y_2,
                heat_2.reshape(points_per_direction, points_x),
                cmap=cm.coolwarm,
                vmin=-1.0,
                vmax=+1.0,
                linewidth=0,
                antialiased=False,
            )
            ims.append([im1, im2])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=False)
    ani.save("HeatEquOpenMDAO_stage_inner_problem.mp4")
