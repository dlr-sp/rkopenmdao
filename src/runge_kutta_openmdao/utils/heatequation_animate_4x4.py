import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation, colors
import h5py

if __name__ == "__main__":
    total_points_per_direction = 101
    piecewise_points_per_direction = total_points_per_direction // 4 + 1

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    coords = []

    for i in range(4):
        for j in range(4):
            x = np.linspace(0.25 * i, 0.25 * (i + 1), piecewise_points_per_direction)
            y = np.linspace(0.25 * j, 0.25 * (j + 1), piecewise_points_per_direction)
            coords.append(np.meshgrid(x, y))

    ims = []

    checkpoint_distance = 1000

    tax = fig.add_axes([0.7, 0.05, 0.25, 0.05])
    tax.axis("off")

    with h5py.File("0_heat_equ_4x4.h5", mode="r") as f_0, h5py.File(
        "1_heat_equ_4x4.h5", mode="r"
    ) as f_1, h5py.File("2_heat_equ_4x4.h5", mode="r") as f_2, h5py.File(
        "3_heat_equ_4x4.h5", mode="r"
    ) as f_3:
        for k in range(0, 201):
            local_ims = []

            for i in range(4):
                for j in range(4):
                    if f"heat_{j+i*4}" in f_0:
                        current_heat = np.array(
                            f_0[f"heat_{j+i*4}/" + str(k * checkpoint_distance)]
                        )
                    elif f"heat_{j+i*4}" in f_1:
                        current_heat = np.array(
                            f_1[f"heat_{j+i*4}/" + str(k * checkpoint_distance)]
                        )
                    elif f"heat_{j+i*4}" in f_2:
                        current_heat = np.array(
                            f_2[f"heat_{j+i*4}/" + str(k * checkpoint_distance)]
                        )
                    elif f"heat_{j+i*4}" in f_3:
                        current_heat = np.array(
                            f_3[f"heat_{j+i*4}/" + str(k * checkpoint_distance)]
                        )
                    local_ims.append(
                        ax.plot_surface(
                            coords[j + i * 4][0],
                            coords[j + i * 4][1],
                            current_heat.reshape(
                                piecewise_points_per_direction,
                                piecewise_points_per_direction,
                            ),
                            cmap=cm.coolwarm,
                            vmin=0.0,
                            vmax=+2.0,
                            linewidth=0,
                            antialiased=False,
                        )
                    )
            local_ims.append(
                tax.text(x=0, y=0, s="time: " + str(i * checkpoint_distance * 1e-4)[:4])
            )

            ims.append(local_ims)
    print("finished generating singular images, now put them together in animation")
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=False)
    ani.save("HeatEquOpenMDAOAnimation4x4.gif")
