if rank == 0:
    with open(f"{file_name}.txt") as f:
        lines = [line for line in f]

    time = []
    for line in lines:
        js = json.loads(line)
        time.append(js["time"])

    delta_t = [0] * len(time)
    for i in range(len(time) - 1):
        delta_t[i] = time[i + 1] - time[i]
    delta_t[i + 1] = delta_t[i]

    # Generate Figure
    fig = plt.figure()

    plt.xlabel("Time t [s]")  # time axis (x axis)
    plt.ylabel("dTime t [s]")  # delta time axis (y axis)
    plt.grid(True)
    plt.title(butcher_tableau.name)
    plt.plot(time, delta_t, "-")
    plt.show()
    fig.savefig(f"{file_name}.pdf")
