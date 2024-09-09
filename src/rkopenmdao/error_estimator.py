from abc import ABC, abstractmethod
from numpy import linalg as LA
import numpy as np


def _convert_list_to_arr(arr):
    if isinstance(arr, list):
        arr = np.array(arr)
    return arr


class ErrorEstimator(ABC):
    @abstractmethod
    def __init__(self, quantity_metadata: dict):
        pass
    @abstractmethod
    def __call__(self, u, embedded_u):
        pass


class SimpleErrorEstimator(ErrorEstimator):
    """
    Solves a norm for the difference of two vectors Delta = U - U_embedded
    """
    def __init__(self, ord=None):
        self.ord = ord

    def __call__(self, u, embedded_u):
        u = _convert_list_to_arr(u)
        embedded_u = _convert_list_to_arr(embedded_u)
        delta = u - embedded_u
        return LA.norm(delta, ord=self.ord, axis=self.axis, keepdims=self.keepdims)

    def __str__(self):
        return f"L_{self.ord}"


class ImprovedErrorEstimator(ErrorEstimator):
    def __init__(self, ord=None, pos_abs_tol=1e-3, eps=1e-3):
        self.ord = ord
        self.pos_abs_tol = pos_abs_tol
        self.eps = eps

    def __call__(self, u, embedded_u):
        u = _convert_list_to_arr(u)
        embedded_u = _convert_list_to_arr(embedded_u)
        delta = (u - embedded_u)/(u+self.pos_abs_tol/self.eps)
        return LA.norm(delta, ord=self.ord)

    def __str__(self):
        if self.keepdims is False:
            return f"L_{self.ord} norm \neta = {self.pos_abs_tol}, \teps = {self.eps}."

a = ImprovedErrorEstimator(ord=np.inf)
print(a([1,2,3],[1,3,2]))