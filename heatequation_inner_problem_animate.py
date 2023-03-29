import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation

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

    with open("inner_problem_stage.txt", mode="r", encoding="utf-8") as f:
        line_num = 1
        for line in f:
            if line_num == 1:
                pass
            else:
                time_str, heat_1_str, heat_2_str = line.split(",")
                time = float(time_str)
                heat_1 = np.fromstring(heat_1_str[1:-1], sep=" ")
                heat_2 = np.fromstring(heat_2_str[1:-2], sep=" ")
                abs_max = max(heat_1.max(), -heat_1.min(), heat_2.max(), -heat_2.min())
                if abs_max > 1.0:
                    print(time, line_num, abs_max)
                if (line_num - 1) % 10 == 1:
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
            line_num += 1

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=False)
    ani.save("HeatEquOpenMDAO_stage_inner_problem.mp4")
