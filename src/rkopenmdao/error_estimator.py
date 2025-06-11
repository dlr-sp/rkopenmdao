"""Contains the classes for the error Estimator of the error controller."""

# pylint: disable = c-extension-no-member

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from mpi4py import MPI
import numpy as np

from .metadata_extractor import TimeIntegrationMetadata


@dataclass
class ErrorEstimator(ABC):
    """
    Solves a norm for the difference of two vectors Delta = U - U_embedded
    """

    order: int = 2
    exclude: list[str] = field(default_factory=list)
    quantity_metadata: TimeIntegrationMetadata = None
    comm: MPI.Comm = field(default_factory=MPI.COMM_WORLD)

    @abstractmethod
    def __call__(self, u, embedded_u) -> float:
        pass

    @abstractmethod
    def __str__(self):
        pass

    def _process_quantity(self, quantity, vector):
        """Solves the Norm with MPI if the quantity is distributed,
        else Sequential solver is used."""
        if quantity.array_metadata.distributed:
            return _mpi_partial_norm(vector, self.order, self.comm)
        else:
            return _non_mpi_partial_norm(vector, self.order)

    def _global_value_estimator(self, global_value, additional_value):
        """Returns the max/min or sums of values, depending on the order."""
        if self.order == np.inf:
            return max(global_value, additional_value)
        if self.order == -np.inf:
            return min(global_value, additional_value)
        return global_value + additional_value

    def _normalize(self, global_value):
        """Normalization calculator"""
        if self.order >= 2 and self.order != np.inf:
            return global_value ** (1 / self.order)
        return global_value


@dataclass
class SimpleErrorEstimator(ErrorEstimator):
    """
    A simple norm solver, that calculates the norm of the difference of two vectors:
    Delta = U - U_embedded

    Attributes
    ----------
    order : {non-zero int, inf, -inf}, optional
        Order of the norm. inf means numpy's inf object. The default is 2.
    quantity_metadata : TimeIntegrationMetadata, optional
        Time integrated quantities
    comm: MPI.COMM_WORLD, optional
        MPI communication
    Methods
    -------
    __call__(u, embedded_u)
        Solves a norm for the difference of two vectors Delta = U - U_embedded.
    __str__()
        prints the Lp/Lebesgue space.
    """

    def __call__(self, u: np.ndarray, embedded_u: np.ndarray) -> float:
        """
        Parameters
        ----------
        u : np.ndarray
            Solution vector of the problem
        embedded_u : np.ndarray
            Solution vector of the embedded problem

        Returns
        -------
        float
            Norm of the difference of two vectors
        """
        global_value = 0
        delta = u - embedded_u
        for quantity in self.quantity_metadata.time_integration_quantity_list:
            if quantity.name not in self.exclude:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                global_value = self._global_value_estimator(
                    global_value,
                    self._process_quantity(quantity, delta.copy()[start:end]),
                )
        global_value = self._normalize(global_value)
        return global_value

    def __str__(self):
        """Prints the Lp/Lebesgue space"""
        return f"L_{self.order}"


@dataclass
class ImprovedErrorEstimator(ErrorEstimator):
    """
    An improved norm solver that calculates the norm of the difference of two
    vectors:
    Delta = (U - U_embedded)/(u + eta/eps)

    Attributes
    ----------
    order : {non-zero int, inf, -inf}, optional
        Order of the norm. inf means numpy's inf object. The default is 2.
    eta : float, optional
        A small positive absolute tolerance which added to avoid division by zero.
    eps: float, optional
        relative error tolerance

    Methods
    -------
    __call__(u, embedded_u):
        Solves a norm for the difference of two vectors
        Delta = (U - U_embedded)/(|U| + eta/eps)
    __str__():
        prints the Lp/Lebesgue space and the attributes
    """

    eta: float = 1e-6
    eps: float = 1e-6

    def __call__(self, u: np.ndarray, embedded_u: np.ndarray) -> float:
        """
        Solves a norm for the difference of two vectors:
        delta = (U - U_embedded)/(|U| + eta/eps)
        Parameters
        ----------
        u : np.ndarray
            Solution vector of the problem
        embedded_u : np.ndarray
            Solution vector of the embedded problem

        Returns
        -------
        float
            Norm of delta.
        """
        global_temp_value = 0
        global_value = 0

        for quantity in self.quantity_metadata.time_integration_quantity_list:
            if quantity.name not in self.exclude:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                global_temp_value = self._global_value_estimator(
                    global_temp_value,
                    self._process_quantity(quantity, u.copy()[start:end]),
                )
        global_temp_value = self._normalize(global_temp_value)

        delta = u - embedded_u
        for quantity in self.quantity_metadata.time_integration_quantity_list:
            if quantity.name not in self.exclude:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                global_value = self._global_value_estimator(
                    global_value,
                    self._process_quantity(quantity, delta.copy()[start:end]),
                )
        global_value /= global_temp_value + self.eta / self.eps
        return global_value

    def __str__(self):
        """prints the Lp/Lebesgue space and the attributes"""
        return f"L_{self.order} norm:  eta = {self.eta}, eps = {self.eps}."


def _mpi_partial_norm(val: np.ndarray, order, comm: MPI.Comm) -> float:
    """Norm calculator using MPI interface."""
    if order == np.inf:
        local_norm = np.abs(val).max()
        partial_norm = comm.allreduce(local_norm, op=MPI.MAX)
    elif order == -np.inf:
        local_norm = np.abs(val).min()
        partial_norm = comm.allreduce(local_norm, op=MPI.MIN)
    elif order == 0:
        local_sum_order = (val != 0).astype(val.real.dtype).sum()
        partial_norm = comm.allreduce(local_sum_order, op=MPI.SUM)
    elif order == 1:
        # Special case for speedup
        local_sum_order = np.sum(np.abs(val))
        partial_norm = comm.allreduce(local_sum_order, op=MPI.SUM)
    elif order == 2:
        # Special case for speedup (complex numbers)
        local_sum_order = np.sum((val.conj() * val).real)
        partial_norm = comm.allreduce(local_sum_order, op=MPI.SUM)
    else:
        local_sum_order = np.sum(np.abs(val) ** order)
        partial_norm = comm.allreduce(local_sum_order, op=MPI.SUM)
    return partial_norm


def _non_mpi_partial_norm(val: np.ndarray, order) -> float:
    """Norm Calculator without MPI Interface"""
    if order == np.inf:
        partial_norm = np.abs(val).max()
    elif order == -np.inf:
        partial_norm = np.abs(val).min()
    elif order == 0:
        partial_norm = (val != 0).astype(val.real.dtype).sum()
    elif order == 1:
        # Special case for speedup
        partial_norm = np.sum(np.abs(val))
    elif order == 2:
        partial_norm = np.sum((val.conj() * val).real)
    else:
        partial_norm = np.sum(np.abs(val) ** order)
    return partial_norm
