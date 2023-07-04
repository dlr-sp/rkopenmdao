from typing import List, Union

import numpy as np


class FunctionalCoefficients:
    def list_quantities(self) -> List[str]:
        raise NotImplementedError(
            "Method 'list_quantities' must be implemented in child class by user."
        )

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            "Method 'get_coefficient' must be implemented in child class by user."
        )


class EmptyFunctionalCoefficients(FunctionalCoefficients):
    def list_quantities(self) -> List[str]:
        return []

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        return 0.0
