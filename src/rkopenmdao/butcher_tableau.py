"""Contains classes for the representation of Butcher tableaux."""
from __future__ import annotations
import warnings

import numpy as np


def _get_column_widths(co_arrays):
    len_max = []
    col_max = None
    for co_array in co_arrays:
        len_max.append(max([len(ai) for ai in co_array.reshape(-1)]))
        col_max = max(len_max)
    return len_max, col_max


class ButcherTableau:
    """
    Representation of a Butcher tableau for Runge-Kutta methods.

    Attributes
    ----------

    butcher_matrix : np.ndarray
        The runge kutta matrix of the Butcher tableau.
    butcher_weight_vector : np.ndarray
        The weight vector of the Butcher tableau.
    butcher_time_stages: np.ndarray, optional
        the time stages of the Butcher tableau.
    p: int, optional
        The order of the global truncation error of the Butcher tableau.
    name: str, optional
        the name of the Butcher tableau.

    Methods
    -------
    min_p_order()
        Returns the minimum order of the Butcher tableau.
    number_of_stages()
        Returns the number of stages of the Butcher tableau.
    is_embedded()
        Returns whether the Butcher tableau is embedded.
    is_explicit()
        Returns whether the Butcher tableau is explicit.
    is_diagonally_implicit()
        Returns whether this tableau is diagonally implicit.
    is_implicit()
        Returns whether this tableau is implicit.
    """

    def __init__(
        self,
        butcher_matrix: np.ndarray,
        butcher_weight_vector: np.ndarray,
        butcher_time_stages: np.ndarray | None = None,
        p=None,
        name='Runge-Kutta method',
    ):
        
        if len(np.shape(butcher_matrix)) == 2:
            self.butcher_matrix = butcher_matrix
        else:
            self.butcher_matrix = np.array([butcher_matrix])
        self.butcher_weight_vector = butcher_weight_vector
        if np.abs(np.sum(self.butcher_weight_vector) - 1.0) > 1e-5:
            warnings.warn("Averaging weights do not sum up to 1")
        if butcher_time_stages is None:
            self.butcher_time_stages = np.sum(self.butcher_matrix, 1)
        else:
            self.butcher_time_stages = butcher_time_stages
            if not np.all(np.sum(self.butcher_matrix, 1) - butcher_time_stages) < 1e-5:
                warnings.warn("Spatial shift matrix (A) rows do not "
                              "sum up to the temporal shift vector (c) indices value")
        self.stages = butcher_weight_vector.size
        self.name = name
        self._p = p
        if (
            self.butcher_matrix.shape != (self.stages, self.stages)
            or self.butcher_weight_vector.shape != self.butcher_time_stages.shape
        ):
            raise AssertionError("Sizes of matrix and vectors doesn't match")
        
    @property
    def p(self):
        if self._p is None:
            raise ValueError("Order is unknown, please specify the order")
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    def min_p_order(self):
        return self._p

    @property
    def is_embedded(self):
        """returns whether the class has also embedded weights to the Runge-Kutta scheme"""
        return False

    def number_of_stages(self):
        """
        Returns the number of stages of the tableau
        Return
        ------
        int
            The number of stages of the butcher tableau.
        """
        return self.stages

    def is_explicit(self) -> bool:
        """
        Returns whether this tableau is explicit.
        Return
        ------
        bool
            True if the Butcher tableau is explicit.
        """
        for i in range(self.stages):
            for j in range(i, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_diagonally_implicit(self) -> bool:
        """
        Returns whether this tableau is diagonally implicit.
        This requires at least one non-zero diagonal element in the Butcher matrix,
        i.e. we don't count explicit schemes as diagonally implicit.
        Return
        ------
        bool
            True if the Butcher tableau is diagonally implicit.
        """
        if self.is_explicit():
            return False
        for i in range(self.stages):
            for j in range(i + 1, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_implicit(self) -> bool:
        """
        Returns whether this tableau is implicit.

        Return
        ------
        bool
            True if implicit
        """
        if self.is_explicit() or self.is_diagonally_implicit():
            return False
        return True

    def __len__(self) -> int:
        """
        Returns the number of rows of the Butcher matrix.

        Returns
        -------
        int
            Number of rows of the Butcher matrix.
        """
        return np.size(self.butcher_matrix, 0)

    def __str__(self):
        """
        Returns the Butcher table as a string

        Returns:
            Butcher table as a string.
        """
        butcher_time_stages = np.array([str(element) for element in self.butcher_time_stages])
        butcher_matrix = np.array([[str(element) for element in lst] for lst in self.butcher_matrix])
        butcher_weight_vector = np.array([str(element) for element in self.butcher_weight_vector])
        len_max, col_max = _get_column_widths([butcher_matrix, butcher_weight_vector, butcher_time_stages])

        s = self.name + '\n'
        for i in range(len(self)):
            s += butcher_time_stages[i].ljust(col_max + 1) + '|'
            for j in range(len(self)):
                s += butcher_matrix[i, j].ljust(col_max + 1)
            s = s.rstrip() + '\n'
        s += '_' * (col_max + 1) + '|' + ('_' * (col_max + 1) * len(self)) + '\n'
        s += ' ' * (col_max + 1) + '|'
        for j in range(len(self)):
            s += butcher_weight_vector[j].ljust(col_max + 1)
        return s.rstrip()


class EmbeddedButcherTableau(ButcherTableau):
    """
    Representation of a Butcher tableau with internal error control
    (i.e. has a second weight vector for representing a second scheme with lower order).

    Attributes
    ----------

    butcher_matrix : np.ndarray
        The runge kutta matrix of the Butcher tableau.
    butcher_weight_vector : np.ndarray
        The weight vector of the Butcher tableau.
    butcher_adaptive_weights : np.ndarray
        The embedded weights of the Butcher tableau.
    butcher_time_stages: np.ndarray, optional
        the time stages of the Butcher tableau.
    p: int, optional
        The order of the global truncation error of the main Butcher tableau.
    phat: int, optional
        The order of the global truncation error of the second Butcher tableau.
    name: str, optional
        the name of the Butcher tableau.
    """

    def __init__(
        self,
        butcher_matrix: np.ndarray,
        butcher_weight_vector: np.ndarray,
        butcher_adaptive_weights: np.ndarray,
        butcher_time_stages: np.ndarray | None = None,
        p: int = None,
        phat: int = None,
        name: str = 'Adaptive Runge-Kutta method'
    ):
        super().__init__(butcher_matrix, butcher_weight_vector, butcher_time_stages, p, name)
        self._phat = phat
        self.butcher_adaptive_weights = butcher_adaptive_weights
        if self.butcher_adaptive_weights.shape != self.butcher_weight_vector.shape:
            raise AssertionError("Size of adaptive stage doesn't match.")

    @classmethod
    def from_butchertableau(cls,
                            butcher_tableau: ButcherTableau,
                            butcher_adaptive_weights: np.ndarray,
                            phat: int = None,
                            name: str = None
                            ):
        """

        Constructs an EmeddedButcherTableau with an existing ButcherTableau and adaptive weights.
        Parameters
        ----------
        butcher_tableau: ButcherTableau
            The main Butcher Tableau.
        butcher_adaptive_weights: np.ndarray
            A vector of adaptive weights.
        phat: int, optional
            The order of the global truncation error of the embedded Butcher tableau.
        name: str, optional
            The name of the Butcher tableau.
        """
        if name is None:
            name = "Adaptive " + butcher_tableau.name

        return cls(butcher_tableau.butcher_matrix,
                   butcher_tableau.butcher_weight_vector,
                   butcher_adaptive_weights,
                   butcher_tableau.butcher_time_stages, butcher_tableau.p, phat, name)

    @property
    def phat(self):
        if self._phat is None:
            raise ValueError("Order is unknown, please specify the order")
        return self._phat

    @phat.setter
    def phat(self, phat):
        self._phat = phat

    @property
    def is_embedded(self):
        return True

    def min_p_order(self) -> int:
        """
        Returns
        -------
        int
            minimum order of the between the main and second Butcher tableau.
        """
        return min(self._p, self._phat)

    @property
    def main_method(self):
        return ButcherTableau(self.butcher_matrix,
                              self.butcher_weight_vector,
                              self.butcher_time_stages,
                              p=self._p,
                              name="Main: " + self.name
                              )

    @property
    def embedded_method(self):
        return ButcherTableau(self.butcher_matrix,
                              self.butcher_adaptive_weights,
                              self.butcher_time_stages,
                              p=self._phat,
                              name="Embedded: " + self.name
                              )