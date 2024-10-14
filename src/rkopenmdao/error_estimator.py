from abc import ABC, abstractmethod
import numpy as np
from mpi4py import MPI


class ErrorEstimator(ABC):
    """
    Solves a norm for the difference of two vectors Delta = U - U_embedded
    """
    @abstractmethod
    def __init__(self, quantity_metadata: dict):
        pass

    @abstractmethod
    def __call__(self, u, embedded_u):
        pass


class SimpleErrorEstimator(ErrorEstimator):
    """
    A simple norm solver, in which a numpy.linalg is utilized to calculate the norm difference of two vectors:
    Delta = U - U_embedded

    Attributes
    ----------
    ord : {non-zero int, inf, -inf}, optional
        Order of the norm. inf means numpy's inf object. The default is 2.

    Methods
    -------
    __call__(u, embedded_u)
        Solves a norm for the difference of two vectors Delta = U - U_embedded.
    __str__()
        prints the Lp/Lebesgue space.
    """
    def __init__(self, ord=2, comm=MPI.COMM_WORLD):
        """
        Parameters
        ----------
        ord: {non-zero int, inf, -inf}, optional
            Order of the norm. inf means numpy's inf object. The default is 2.
        """
        self.ord = ord
        self.comm = comm
        self._parallel_setup()

    def _parallel_setup(self):
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

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
        return _mpi_norm(delta, self.ord, self.comm)

    def __str__(self):
        """Prints the Lp/Lebesgue space"""
        return f"L_{self.ord}"


class ImprovedErrorEstimator(ErrorEstimator):
    """
    An improved norm solver, in which a numpy.linalg is utilized to calculate the norm difference of two vectors:
    Delta = (U - U_embedded)/(u + eta/eps)

    Attributes
    ----------
    ord : {non-zero int, inf, -inf}, optional
        Order of the norm. inf means numpy's inf object. The default is 2.
    eta : float, optional
        A small positive absolute tolerance which added to avoid division by zero.
    eps: float, optional
        relative error tolerance

    Methods
    -------
    __call__(u, embedded_u):
        Solves a norm for the difference of two vectors Delta = (U - U_embedded)/(|U| + eta/eps)
    __str__():
        prints the Lp/Lebesgue space and the attributes
        """
    def __init__(self, ord=2, eta=1e-3, eps=1e-3):
        """
        Parameters
        ----------
        ord: {non-zero int, inf, -inf}, optional
            Order of the norm. inf means numpy's inf object. The default is 2.
        eta: float, optional
            A small positive absolute tolerance which added to avoid division by zero.
        eps: float, optional
            relative error tolerance
        """
        self.ord = ord
        self.eta = eta
        self.eps = eps
        self._parallel_setup()

    def _parallel_setup(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nprocs = self.comm.Get_size()

    def __call__(self, u: np.ndarray, embedded_u: np.ndarray) -> float:
        """
        Solves a norm for the difference of two vectors Delta = (U - U_embedded)/(|U| + eta/eps)
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

        u_norm = _mpi_norm(u, self.ord, self.comm)
        delta = (u - embedded_u) / (u_norm + self.eta / self.eps)
        return _mpi_norm(delta, self.ord, self.comm)

    def __str__(self):
        """prints the Lp/Lebesgue space and the attributes"""
        return f"L_{self.ord} norm:  eta = {self.eta}, eps = {self.eps}."


def _mpi_norm(val: np.ndarray, ord, comm: MPI.Comm) -> float:
    if ord == np.inf:
        local_norm = np.abs(val).max()
        norm = comm.reduce(local_norm, op=MPI.MAX, root=0)
    elif ord == -np.inf:
        local_norm = np.abs(val).min()
        norm = comm.reduce(local_norm, op=MPI.MIN, root=0)
    elif ord == 0:
        local_sum_order = (val != 0).astype(val.real.dtype).sum()
        norm = comm.reduce(local_sum_order, op=MPI.SUM, root=0)
    elif ord == 1:
        # Special case for speedup
        local_sum_order = np.sum(np.abs(val))
        norm = comm.reduce(local_sum_order, op=MPI.SUM, root=0)
    elif ord == 2:
        # Special case for speedup (complex numbers)
        local_sum_order = np.sum((val.conj() * val).real)
        norm = comm.reduce(local_sum_order, op=MPI.SUM, root=0) ** .5
    else:
        local_sum_order = np.sum(np.abs(val) ** ord)
        norm = comm.reduce(local_sum_order, op=MPI.SUM, root=0) ** (1 / ord)
    return norm
