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
        if np.sum(self.butcher_weight_vector) != 1:
            warnings.warn("Averaging weights do not sum up to 1")
        if butcher_time_stages is None:
            self.butcher_time_stages = np.sum(self.butcher_matrix, 1)
        else:
            self.butcher_time_stages = butcher_time_stages
            if np.sum(self.butcher_matrix, 1) != butcher_time_stages:
                warnings.warn("Spatial shift matrix (A) rows do not "
                              "sum up to the temporal shift vector (c) indices value")
        self.stages = butcher_weight_vector.size
        self.name = name
        self.p = p
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

    @property
    def is_embedded(self):
        return False

    def number_of_stages(self):
        """Returns the number of stages of the tableau"""
        return self.stages

    def is_explicit(self) -> bool:
        """Checks whether this tableau is explicit or not."""
        for i in range(self.stages):
            for j in range(i, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_diagonally_implicit(self) -> bool:
        """
        Checks whether this tableau is diagonally implicit or not.
        This requires at least one non-zero diagonal element in the Butcher matrix,
        i.e. we don't count explicit schemes as diagonally implicit.
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
        Checks whether this tableau is fully implicit or not.
        """
        if self.is_explicit() or self.is_diagonally_implicit():
            return False
        return True

    def __len__(self):
        return np.size(self.butcher_matrix, 0)

    def __str__(self):
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
    """

    def __init__(
        self,
        butcher_matrix: np.ndarray,
        butcher_weight_vector: np.ndarray,
        butcher_adaptive_weights: np.ndarray,
        butcher_time_stages: np.ndarray | None = None,
        p=None,
        phat=None,
        name='Adaptive Runge-Kutta method'
    ):
        super().__init__(butcher_matrix, butcher_weight_vector, butcher_time_stages, p, name)
        self.phat = phat
        self.butcher_adaptive_weights = butcher_adaptive_weights
        if self.butcher_adaptive_weights.shape != self.butcher_weight_vector.shape:
            raise AssertionError("Size of adaptive stage doesn't match.")

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

    @property
    def main_method(self):
        return ButcherTableau(self.butcher_matrix,
                              self.butcher_weight_vector,
                              self.butcher_time_stages,
                              p=self._p,
                              name="Main method"
                              )

    @property
    def embedded_method(self):
        return ButcherTableau(self.butcher_matrix,
                              self.butcher_weight_vector,
                              self.butcher_time_stages,
                              p=self._phat,
                              name="Embedded method"
                              )
