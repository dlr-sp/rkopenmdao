from abc import ABC, abstractmethod
from numpy import linalg as LA
import numpy as np


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
    ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        Order of the norm. inf means numpy's inf object. The default is None.

    Methods
    -------
    __call__(u, embedded_u)
        Solves a norm for the difference of two vectors Delta = U - U_embedded.
    __str__()
        prints the Lp/Lebesgue space.
    """
    def __init__(self, ord=None):
        """
        Parameters
        ----------
        ord: {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
            Order of the norm. inf means numpy's inf object. The default is None
        """
        self.ord = ord

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
        return LA.norm(delta, ord=self.ord)

    def __str__(self):
        """Prints the Lp/Lebesgue space"""
        return f"L_{self.ord}"


class ImprovedErrorEstimator(ErrorEstimator):
    """
    An improved norm solver, in which a numpy.linalg is utilized to calculate the norm difference of two vectors:
    Delta = (U - U_embedded)/(u + pos_abs_tol/eps)

    Attributes
    ----------
    ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        Order of the norm. inf means numpy's inf object. The default is None
    eta : float, optional
        A small positive absolute tolerance which added to avoid division by zero.
    eps: float, optional
        relative error tolerance

    Methods
    -------
    __call__(u, embedded_u):
        Solves a norm for the difference of two vectors Delta = (U - U_embedded)/(u + pos_abs_tol/eps)
    __str__():
        prints the Lp/Lebesgue space and the attributes
        """
    def __init__(self, ord=None, pos_abs_tol=1e-3, eps=1e-3):
        self.ord = ord
        self.eta = pos_abs_tol
        self.eps = eps

    def __call__(self, u: np.ndarray, embedded_u: np.ndarray) -> float:
        """
        Solves a norm for the difference of two vectors Delta = (U - U_embedded)/(u + pos_abs_tol/eps)
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
        delta = (u - embedded_u)/(u+self.eta/self.eps)
        return LA.norm(delta, ord=self.ord)

    def __str__(self):
        """prints the Lp/Lebesgue space and the attributes"""
        return f"L_{self.ord} norm:  eta = {self.eta}, eps = {self.eps}."
