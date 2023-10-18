from typing import Union
from typing import Callable
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
import h5py

from rkopenmdao.butcher_tableau import ButcherTableau

from . import fdm_matrix, domain, boundary, inhomogeneity


class HeatEquation:
    """
    Class summarizing the necessary information about a Heat equation
    and its FDM discretization
    """

    def __init__(
        self,
        heat_domain: domain.Domain,
        heat_inhomogeneity_function: Callable[[float, float, float], float],
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
        self.inhomogeneity_vector = inhomogeneity.InhomogeneityVector(
            self.domain,
            self.diffusivity,
            heat_inhomogeneity_function,
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

        # we use a 1d array instead of a 2d one because it is a diagonal matrix and it can be
        # emulated via the hadamard product
        self.mass_matrix = np.ones_like(self.initial_vector)
        self.mass_matrix[0 : self.domain.n_x] /= 2
        self.mass_matrix[
            (self.domain.n_y - 1) * self.domain.n_x : self.domain.n_y * self.domain.n_x
        ] /= 2

        self.mass_matrix[0 : self.domain.n_x * self.domain.n_y : self.domain.n_x] /= 2
        self.mass_matrix[
            self.domain.n_x - 1 : self.domain.n_x * self.domain.n_y : self.domain.n_x
        ] /= 2

    def set_time(self, new_time: float):
        """Sets time and updates homogeneity term accordingly."""
        self.time = new_time
        self.inhomogeneity_vector.update_pde_inhomogeneity(self.time)

    def heat_equation_time_stage_residual(
        self,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
        old_value: np.ndarray,
        accumulated_stages: np.ndarray,
        current_slope: np.ndarray,
    ):
        """Instationary residual of the heat equation."""
        value = (
            old_value
            + delta_t * accumulated_stages
            + delta_t * butcher_diagonal_element * current_slope
        )

        self.inhomogeneity_vector.update_boundary_inhomogeneity(stage_time)
        self.inhomogeneity_vector.update_pde_inhomogeneity(stage_time)
        guess_stage = (
            self.fdm_matrix.mat_vec_prod(value)
            + self.mass_matrix * self.inhomogeneity_vector.return_vector()
        )
        new_guess_residual = self.mass_matrix * current_slope - guess_stage

        return new_guess_residual

    def d_stage_d_old_value(self):
        """Derivative of instationary residual wrt. to old state."""
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: -self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: -self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def d_stage_d_stage(self, delta_t: float, butcher_diagonal_element: float):
        """Derivative of instationary residual wrt. to current stage variable."""
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: self.mass_matrix * vec
            - delta_t * butcher_diagonal_element * self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: self.mass_matrix * vec
            - delta_t
            * butcher_diagonal_element
            * self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def d_stage_d_accumulated_stages(
        self,
        delta_t: float,
    ):
        """Derivative of instationary residual wrt. to accumulated previous stage variables."""
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: -delta_t * self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: -delta_t * self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def d_stage_d_boundary_segment(self, delta: float, segment: str):
        """Derivative of instationary residual wrt. to the requested boundary segment."""

        def boundary_mat_vec(vector):
            result = np.zeros(self.domain.n_x * self.domain.n_y)
            result[self.domain.boundary_indices(segment)] = -2.0 * vector / delta
            return self.mass_matrix * result

        def boundary_mat_vec_transpose(vector):
            result = np.zeros(
                self.domain.n_y if segment in ("left", "right") else self.domain.n_x
            )
            result = -2.0 * vector[self.domain.boundary_indices(segment)] / delta
            return self.mass_matrix[self.domain.boundary_indices(segment)] * result

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
        guessed_stage=None,
    ):
        """Solves the instationary residual for the stage variable."""
        if butcher_diagonal_element != 0:
            rhs = old_value + delta_t * accumulated_stages
            rhs = self.fdm_matrix.mat_vec_prod(rhs)

            self.inhomogeneity_vector.update_boundary_inhomogeneity(stage_time)
            self.inhomogeneity_vector.update_pde_inhomogeneity(stage_time)
            rhs += self.mass_matrix * self.inhomogeneity_vector.return_vector()
            lhs_matrix = LinearOperator(
                (self.domain.n_x * self.domain.n_y, self.domain.n_x * self.domain.n_y),
                matvec=lambda vector: self.mass_matrix * vector
                - delta_t
                * butcher_diagonal_element
                * self.fdm_matrix.mat_vec_prod(vector),
            )

            guess = (
                self.guess_stage(
                    stage_time,
                    delta_t,
                    butcher_diagonal_element,
                    old_value,
                    accumulated_stages,
                )
                if guessed_stage is None
                else guessed_stage
            )

            result, iterstatus = gmres(
                lhs_matrix,
                rhs,
                x0=guess,
                **self.solver_options,
            )
            return result
        else:
            result = self.fdm_matrix.mat_vec_prod(
                old_value + delta_t * accumulated_stages
            )
            self.inhomogeneity_vector.update_boundary_inhomogeneity(stage_time)
            self.inhomogeneity_vector.update_pde_inhomogeneity(stage_time)
            result += self.inhomogeneity_vector.return_vector()

            return result

    def guess_stage(
        self,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
        old_value: np.ndarray,
        accumulated_stages: np.ndarray,
    ):
        """Provides a guess what the current stage variable based on old information."""
        stage_guess = old_value.copy()
        stage_guess += delta_t * accumulated_stages
        stage_guess = self.fdm_matrix.mat_vec_prod(stage_guess)
        if butcher_diagonal_element != 0:
            stage_guess += (
                delta_t
                * butcher_diagonal_element
                * self.fdm_matrix.mat_vec_prod(stage_guess)
            )
        self.inhomogeneity_vector.update_boundary_inhomogeneity(stage_time)
        self.inhomogeneity_vector.update_pde_inhomogeneity(stage_time)
        return stage_guess

    def guess_d_stage(self, delta_t, butcher_diagonal_element):
        """Provides a guess what the derivative wrt. the current stage variable based on old information."""
        return LinearOperator(
            shape=(
                self.domain.n_x * self.domain.n_y,
                self.domain.n_x * self.domain.n_y,
            ),
            matvec=lambda vec: vec
            + delta_t * butcher_diagonal_element * self.fdm_matrix.mat_vec_prod(vec),
            rmatvec=lambda vec: vec
            + delta_t
            * butcher_diagonal_element
            * self.fdm_matrix.mat_vec_prod_transpose(vec),
        )

    def solve_d_stage_d_stage(
        self,
        delta_t: float,
        butcher_diagonal_element: float,
        mode: str,
        rhs_slope: np.ndarray,
        guessed_result=None,
    ):
        """Solves (M+delta_t*a_ii*A) dk_i = dR_k_i (fwd-mode) or (M+delta_t*a_ii*A)^T dR_k_i = dk_i (rev-mode)"""
        if butcher_diagonal_element != 0:
            matrix = self.d_stage_d_stage(delta_t, butcher_diagonal_element)
            approx_inverse = self.guess_d_stage(delta_t, butcher_diagonal_element)
            slope_result, iterstatus = gmres(
                matrix if mode == "fwd" else matrix.transpose(),
                rhs_slope,
                x0=approx_inverse.matvec(rhs_slope)
                if mode == "fwd"
                else approx_inverse.rmatvec(rhs_slope)
                if guessed_result is None
                else guessed_result,
                **self.solver_options,
            )
            return slope_result
        else:
            return rhs_slope

    def heat_equation_time_step(
        self,
        time: float,
        delta_t: float,
        old_value: np.ndarray,
        butcher_tableau: ButcherTableau,
    ):
        """Computes one time step of the RK-scheme for the heat equation."""
        previous_stages = {}
        accumulated_stages = np.zeros_like(old_value)
        for i in range(butcher_tableau.number_of_stages()):
            for j in range(i):
                accumulated_stages += (
                    butcher_tableau.butcher_matrix[i, j] * previous_stages[j]
                )

            new_stage = self.solve_heat_equation_time_stage_residual(
                time + butcher_tableau.butcher_time_stages[i] * delta_t,
                delta_t,
                butcher_tableau.butcher_matrix[i, i],
                old_value.copy(),
                accumulated_stages,
                guessed_stage=None,
            )
            previous_stages[i] = new_stage
            accumulated_stages.fill(0.0)

        accumulated_stages += old_value
        for i in range(butcher_tableau.number_of_stages()):
            accumulated_stages += (
                delta_t * butcher_tableau.butcher_weight_vector[i] * previous_stages[i]
            )

        return accumulated_stages

    def solve_heat_equation(
        self,
        butcher_tableau: ButcherTableau,
        delta_t: float,
        number_of_steps: int,
        output_file="data.h5",
        checkpoint_distance=10,
    ):
        """Solves the heat equation with the RK-scheme for the given butcher tableau."""
        current_vector = self.initial_vector.copy()
        current_time = self.time
        with h5py.File(output_file, mode="w") as f:
            f.create_dataset("heat/0", data=current_vector)

        for i in range(1, number_of_steps + 1):
            try:
                current_vector = self.heat_equation_time_step(
                    current_time, delta_t, current_vector, butcher_tableau
                )
                current_time += delta_t
                if i % checkpoint_distance == 0:
                    with h5py.File(output_file, mode="r+") as f:
                        f.create_dataset("heat/" + str(i), data=current_vector)

            except ValueError:
                break
