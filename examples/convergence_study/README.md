# 'convergence_study/' – Usage example for RKOpenMDAO

This example shows how to use RKOpenMDAO to solve a simple ODE and PDE problems.

---

## Table of Contents
- [What’s inside?](#whats-inside)
- [How to run an example](#how-to-run-an-example)
- [The equations](#the-equations)

---

## What's inside?

The `convergence_study/` directory contains the following folders.
All examples assume you have installed the core package (see the main repository’s README for installation instructions).

| Folder                 | Purpose                          | Quick run command                                   |
|------------------------|----------------------------------|-----------------------------------------------------|
| `integration_scripts/` | Running the unsteady simulations | `python -m integration_scripts.main 'problem_name'` |
| `visualization/`       | Plotting results with Matplotlib | `python -m visualization.main 'problem_name'`       |


In the `integration_scripts` folder, you will find the following scripts:
- `main.py`: Runs all the simulations and saves the results.
- `adaptive.py`: Runs the adaptive simulation.
- `avg_homogeneous.py`: Runs the homogeneous simulation with an average step size computed from the adaptive simulation 
files.
- `homogeneous.py`: Runs the homogeneous simulation with a set of step sizes.

In the `visualization` folder, you will find the following scripts:
- `main.py`: Plots the results from the simulations.
- `plot_convergence.py`: Plots the convergence of the homogeneous time stepping simulations.
- `plot_log_error.py`: Plots the logarithmic error of the adaptive simulations with comparison to the average step size
homogeneous simulation.

---
## How to run an example
In order to run an example, you need to navigate and remain in the `convergence_study`, and  
run the desired script using the `-m` flag, e.g. `python -m integration_scripts.main 'problem_name'`.
The flag `problem_name` can be one of the following: `prothero_robinson`, `kaps`. The description of each problem 
can be found in [The equations](#the-equations).
In addition, the flag `--stiffness` can be utilized to change the stiffness parameter of the equations, using a float value.

TIP: if the homogeneous problem is run first, the adaptive problem will use the average local tolerance of the 
convergence study for each Runge-Kutta scheme.

---
## The equations
In this convergence study, we solve the following analytical problems:
### Prothero-Robinson
This problem is a single disciplinary system:
$$`x' = \lambda * (x - \Phi(t)) + \frac{d\Phi(t)}{dt}`$$
Where $\Phi(t)$ is a function of $t$:
$$`\Phi(t) = sin(\frac{\pi}{4}+t)`$$
The problems' initial value is $`x(0)=\Phi(0)`$,
and $\lambda$ is a constant that affects the stiffness of the equation.

### Kaps' Problem
This problem is a two-disciplinary system:
$$`\epsilon y_1'(t) = -\left(1+2\epsilon\right)y_1(t) + y_2^2(t)`$$
$$`y_2'(t) = y_1(t) - y_2(t) - y_2^2(t)`$$
with initial conditions $`y_1(0)=y_2(0)=0`$, and $\epsilon$ is a constant that affects the stiffness of the equation.




