from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain

import h5py
from mpi4py.MPI import Comm
import numpy as np

from .metadata_extractor import TimeIntegrationMetadata


@dataclass
class FileWriterInterface(ABC):
    """Interface for general file writers in RKOpenMDAO."""

    file_name: str
    time_integration_metadata: TimeIntegrationMetadata
    comm: Comm

    @abstractmethod
    def setup_file(self):
        """Creates file and prepares other necessary one-time work."""

    @abstractmethod
    def write_step(
        self,
        step: int,
        time: float,
        time_integration_data: np.ndarray,
        postprocessing_data: np.ndarray,
    ):
        """Writes out step to file"""


@dataclass
class Hdf5FileWriter(FileWriterInterface):

    def setup_file(self):
        if self.comm.rank == 0:
            written_out_quantities = chain(
                self.time_integration_metadata.time_integration_quantity_list,
                self.time_integration_metadata.postprocessing_quantity_list,
            )
            with h5py.File(self.file_name, mode="w") as f:
                for quantity in written_out_quantities:
                    f.create_group(quantity.name)

    def write_step(
        self,
        step: int,
        time: float,
        time_integration_data: np.ndarray,
        postprocessing_data: np.ndarray,
    ):
        written_out_quantities = chain(
            self.time_integration_metadata.time_integration_quantity_list,
            self.time_integration_metadata.postprocessing_quantity_list,
        )
        with h5py.File(self.file_name, mode="r+", driver="mpio", comm=self.comm) as f:
            for quantity in written_out_quantities:
                quantity_group = f[quantity.name]
                dataset = quantity_group.create_dataset(
                    str(step),
                    shape=quantity.array_metadata.global_shape,
                    dtype=np.float64,
                )
                dataset.attrs["time"] = time
                if quantity.array_metadata.local:
                    start_array = quantity.array_metadata.start_index
                    end_array = quantity.array_metadata.end_index
                    start_tuple = np.unravel_index(
                        quantity.array_metadata.global_start_index,
                        quantity.array_metadata.global_shape,
                    )
                    end_tuple = np.unravel_index(
                        quantity.array_metadata.global_end_index - 1,
                        quantity.array_metadata.global_shape,
                    )
                    access_list = []
                    for start_index, end_index in zip(start_tuple, end_tuple):
                        access_list.append(slice(start_index, end_index + 1))
                    if quantity.type == "time_integration":
                        f[quantity.name][str(step)][tuple(access_list)] = (
                            time_integration_data[start_array:end_array].reshape(
                                quantity.array_metadata.shape
                            )
                        )
                    elif quantity.type == "postprocessing":
                        f[quantity.name][str(step)][tuple(access_list)] = (
                            postprocessing_data[start_array:end_array].reshape(
                                quantity.array_metadata.shape
                            )
                        )
