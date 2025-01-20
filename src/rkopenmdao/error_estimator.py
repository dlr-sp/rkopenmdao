"""Contains the classes for the error Estimator of the error controller."""

# pylint: disable = c-extension-no-member

from abc import ABC, abstractmethod
import numpy as np
from mpi4py import MPI
from .metadata_extractor import TimeIntegrationMetadata


class ErrorEstimator(ABC):
    """
    Solves a norm for the difference of two vectors Delta = U - U_embedded
    """

    @abstractmethod
    def __init__(
        self,
        order: int,
        quantity_metadata: TimeIntegrationMetadata,
        comm: MPI.COMM_WORLD,
    ):
        pass

    @abstractmethod
    def __call__(self, u, embedded_u):
        pass


class SimpleErrorEstimator(ErrorEstimator):
    """
    A simple norm solver, in which a numpy.linalg is utilized to calculate the norm
    difference of two vectors:
    Delta = U - U_embedded

    Attributes
    ----------
    order : {non-zero int, inf, -inf}, optional
        Order of the norm. inf means numpy's inf object. The default is 2.

    Methods
    -------
    __call__(u, embedded_u)
        Solves a norm for the difference of two vectors Delta = U - U_embedded.
    __str__()
        prints the Lp/Lebesgue space.
    """

    def __init__(
        self,
        order=2,
        quantity_metadata: TimeIntegrationMetadata = None,
        comm=MPI.COMM_WORLD,
    ):
        """
        Parameters
        ----------
        order: {non-zero int, inf, -inf}, optional
            Order of the norm. inf means numpy's inf object. The default is 2.
        comm: MPI.COMM_WORLD, optional
            MPI communicator "MPI_COMM_WORLD"
        """
        self.order = order
        self.comm = comm

    def __call__(self, u: np.ndarray, embedded_u: np.ndarray) -> float:
        """
        Parameters
        ----------
        u : np.ndarray
            Solution vector of the problem
        embedded_u : np.ndarray
            Solution vector of the embedded problem

        Return
        -------
        float
            Norm of the difference of two vectors
        """
        delta = u - embedded_u
        return _mpi_norm(delta, self.order, self.comm)

    def __str__(self):
        """Prints the Lp/Lebesgue space"""
        return f"L_{self.order}"


class ImprovedErrorEstimator(ErrorEstimator):
    """
    An improved norm solver, in which a numpy.linalg is utilized to calculate the norm
    difference of two vectors:
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

    def __init__(
        self,
        order=2,
        quantity_metadata: TimeIntegrationMetadata = None,
        comm=MPI.COMM_WORLD,
        eta=1e-6,
        eps=1e-6,
    ):
        """
        Parameters
        ----------
        order: {non-zero int, inf, -inf}, optional
            Order of the norm. inf means numpy's inf object. The default is 2.
        comm: MPI.COMM_WORLD, optional
            MPI communicator "MPI_COMM_WORLD"
        eta: float, optional
            A small positive absolute tolerance which added to avoid division by zero.
        eps: float, optional
            relative error tolerance
        """
        self.order = order
        self.eta = eta
        self.eps = eps
        self.comm = comm

    def __call__(self, u: np.ndarray, embedded_u: np.ndarray) -> float:
        """
        Solves a norm for the difference of two vectors
        Delta = (U - U_embedded)/(|U| + eta/eps)
        Parameters
        ----------
        u : np.ndarray
            Solution vector of the problem
        embedded_u : np.ndarray
            Solution vector of the embedded problem

        Return
        -------
        float
            Norm of Delta
        """

        u_norm = _mpi_norm(u.copy(), self.order, self.comm)
        delta = u - embedded_u
        return _mpi_norm(delta, self.order, self.comm) / (u_norm + self.eta / self.eps)

    def __str__(self):
        """prints the Lp/Lebesgue space and the attributes"""
        return f"L_{self.order} norm:  eta = {self.eta}, eps = {self.eps}."


def _mpi_norm(val: np.ndarray, order, comm: MPI.Comm) -> float:
    """Norm calculator using MPI interface."""
    if order == np.inf:
        local_norm = np.abs(val).max()
        norm = comm.allreduce(local_norm, op=MPI.MAX)
    elif order == -np.inf:
        local_norm = np.abs(val).min()
        norm = comm.allreduce(local_norm, op=MPI.MIN)
    elif order == 0:
        local_sum_order = (val != 0).astype(val.real.dtype).sum()
        norm = comm.allreduce(local_sum_order, op=MPI.SUM)
    elif order == 1:
        # Special case for speedup
        local_sum_order = np.sum(np.abs(val))
        norm = comm.allreduce(local_sum_order, op=MPI.SUM)
    elif order == 2:
        # Special case for speedup (complex numbers)
        local_sum_order = np.sum((val.conj() * val).real)
        norm = comm.allreduce(local_sum_order, op=MPI.SUM) ** 0.5
    else:
        local_sum_order = np.sum(np.abs(val) ** order)
        norm = comm.allreduce(local_sum_order, op=MPI.SUM) ** (1 / order)
    return norm
