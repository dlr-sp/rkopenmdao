from typing import Union
from typing import Callable
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from examples.heatequation.src import fdm_matrix, domain, boundary, inhomogenity
from src.runge_kutta_openmdao.runge_kutta import runge_kutta


class HeatEquation:
    """
    Class summarizing the necessary information about an Heat equation
    and its FDM descretization
    """

    def __init__(
        self,
        heat_domain: domain.Domain,
        heat_inhomogenity_function: Callable[[float, float, float], float],
        heat_boundary_condition: boundary.BoundaryCondition,
        heat_diffusivity: float,
        initial_value: Union[np.ndarray, Callable[[float, float], float]],
        solver_options: dict,
        start_time=0.0,
    ):
        self.domain = heat_domain
        self.diffusivity = heat_diffusivity
        self.boundary_condition = heat_boundary_condition
        self.fdm_matrix = fdm_matrix.FdmMatrix(self.domain, self.diffusivity)
        self.inhomogenity_vector = inhomogenity.InhomogenityVector(
            self.domain,
            self.diffusivity,
            heat_inhomogenity_function,
            self.boundary_condition,
        )
        self.initial_vector = np.zeros(self.domain.n_x * self.domain.n_y)
        if isinstance(initial_value, np.ndarray):
            self.initial_vector = initial_value
        else:
            for j in range(self.domain.n_y):
                for i in range(self.domain.n_x):
                    index = self.domain.ij_to_index(i, j)
                    coordinates = self.domain.ij_coordinates(i, j)
                    self.initial_vector[index] = initial_value(
                        coordinates[0], coordinates[1]
                    )
        self.time = start_time
        self.solver_options = solver_options

    def set_time(self, new_time: float):
        self.time = new_time
        self.inhomogenity_vector.update_pde_inhomogenity(self.time)

    def stationary_func(self, arg: np.ndarray):
        return (
            self.fdm_matrix.mat_vec_prod(arg) + self.inhomogenity_vector.return_vector()
        )

    def stationary_d_func(self, d_arg: np.ndarray):
        return self.fdm_matrix.mat_vec_prod(d_arg)

    def heat_equation_time_stage_residual(
        self,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
        old_value: np.ndarray,
        accumulated_stages: np.ndarray,
        current_guess: np.ndarray,
    ):
        value = (
            old_value
            + delta_t * accumulated_stages
            + delta_t * butcher_diagonal_element * current_guess
        )

        self.inhomogenity_vector.update_boundary_inhomogenity(stage_time)
        self.inhomogenity_vector.update_pde_inhomogenity(stage_time)
        return (
            current_guess
            - self.fdm_matrix.mat_vec_prod(value)
            - self.inhomogenity_vector.return_vector()
        )

    def d_stage_d_old_value(self):
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: -self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: -self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def d_stage_d_stage(self, delta_t: float, butcher_diagonal_element: float):
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: vec
            - delta_t * butcher_diagonal_element * self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: vec
            - delta_t
            * butcher_diagonal_element
            * self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def d_stage_d_accumulated_stages(
        self,
        delta_t: float,
    ):
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: -delta_t * self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: -delta_t * self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def d_stage_d_boundary_segment(self, delta: float, segment: str):
        def boundary_mat_vec(vector):
            result = np.zeros(self.domain.n_x * self.domain.n_y)
            if segment in ("left", "right"):
                result[self.domain.boundary_indices(segment)] = -2.0 * vector / delta
            elif segment in ("upper", "lower"):
                result[self.domain.boundary_indices(segment)] = -2.0 * vector / delta
            return result

        def boundary_mat_vec_transpose(vector):
            result = np.zeros(
                self.domain.n_y if segment in ("left", "right") else self.domain.n_x
            )
            if segment in ("left", "right"):
                result = -2.0 * vector[self.domain.boundary_indices(segment)] / delta
            elif segment in ("upper", "lower"):
                result = -2.0 * vector[self.domain.boundary_indices(segment)] / delta
            return result

        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_y if segment in ("left", "right") else self.domain.n_x,
            ),
            matvec=boundary_mat_vec,
            rmatvec=boundary_mat_vec_transpose,
        )

    def solve_heat_equation_time_stage_residual(
        self,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
        old_value: np.ndarray,
        accumulated_stages: np.ndarray,
        current_guess=None,
    ):
        # In general, it could be done like this:
        #
        # runge_kutta_stage = runge_kutta.RungeKuttaStage(
        #     butcher_tableau, self.stationary_func, delta_t
        # )
        # return runge_kutta_stage.solve_stage_residual(
        #     stage_num, old_value, self.time, current_guess, **previous_stages
        # )
        # However, for the heat equation this is linear, so there is no need for a nonlinear solver

        rhs = old_value + delta_t * accumulated_stages
        rhs = self.fdm_matrix.mat_vec_prod(rhs)

        self.inhomogenity_vector.update_boundary_inhomogenity(stage_time)
        self.inhomogenity_vector.update_pde_inhomogenity(stage_time)
        rhs += self.inhomogenity_vector.return_vector()

        lhs_matrix = LinearOperator(
            (self.domain.n_x * self.domain.n_y, self.domain.n_x * self.domain.n_y),
            matvec=lambda vector: vector
            - delta_t * butcher_diagonal_element * self.fdm_matrix.mat_vec_prod(vector),
        )

        result, _ = gmres(lhs_matrix, rhs, x0=current_guess, **self.solver_options)
        return result

    def solve_d_stage_d_stage(
        self,
        delta_t: float,
        butcher_diagonal_element: float,
        mode: str,
        rhs: np.ndarray,
        guess=None,
    ):
        matrix = self.d_stage_d_stage(delta_t, butcher_diagonal_element)
        result, _ = gmres(
            matrix if mode == "fwd" else matrix.transpose(),
            rhs,
            x0=guess,
            **self.solver_options
        )
        return result

    def heat_equation_time_step(
        self,
        time: float,
        delta_t: float,
        old_value: np.ndarray,
        butcher_tableau: runge_kutta.ButcherTableau,
    ):
        previous_stages = {}
        new_value = old_value.copy()
        for i in range(butcher_tableau.number_of_stages()):
            new_stage = self.solve_heat_equation_time_stage_residual(
                time,
                delta_t,
                butcher_tableau,
                i,
                old_value.copy(),
                current_guess=None,
                **previous_stages
            )
            previous_stages[str(i)] = new_stage
            new_value += delta_t * butcher_tableau.butcher_weight_vector[i] * new_stage
        return new_value

    def solve_heat_equation(
        self,
        butcher_tableau: runge_kutta.ButcherTableau,
        delta_t: float,
        number_of_steps: int,
    ):
        current_vector = self.initial_vector.copy()
        current_time = self.time
        solution = {current_time: current_vector.copy()}
        print(current_time, current_vector)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        x = np.linspace(0, 1, self.domain.n_x)
        y = np.linspace(0, 1, self.domain.n_y)
        x, y = np.meshgrid(x, y)

        ims = []

        ax.set_zlim(-2.0, 2.0)
        im = ax.plot_surface(
            x,
            y,
            current_vector.reshape(self.domain.n_x, self.domain.n_y),
            cmap=cm.coolwarm,
            vmin=-3.0,
            vmax=+3.0,
            linewidth=0,
            antialiased=False,
        )
        ims.append([im])

        for i in range(number_of_steps):
            try:
                current_vector = self.heat_equation_time_step(
                    current_time, delta_t, current_vector, butcher_tableau
                )
                current_time += delta_t
                solution[current_time] = current_vector
                if i % 10 == 0:
                    im = ax.plot_surface(
                        x,
                        y,
                        current_vector.reshape(self.domain.n_x, self.domain.n_y),
                        cmap=cm.coolwarm,
                        vmin=-3.0,
                        vmax=+3.0,
                        linewidth=0,
                        antialiased=False,
                        animated=True,
                    )
                    ims.append([im])
                    print(
                        current_time,
                        current_vector,
                    )
            except ValueError:
                break
        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=1000
        )
        ani.save("HeatEqu.mp4")
        return solution
