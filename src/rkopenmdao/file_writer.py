"""Defines interface for file writers, as well as the HDF5 file writer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Tuple

import h5py

# pylint doesn't pick it up, but it's there
# pylint: disable = no-name-in-module
from mpi4py.MPI import Comm
import numpy as np

from .metadata_extractor import Quantity, TimeIntegrationMetadata


@dataclass
class FileWriterInterface(ABC):
    """Interface for general file writers in RKOpenMDAO."""

    file_name: str
    time_integration_metadata: TimeIntegrationMetadata
    comm: Comm

    @abstractmethod
    def write_step(
        self,
        step: int,
        time: float,
        time_integration_data: np.ndarray,
        error_measure: float = None,
    ) -> None:
        """Writes out step to file"""


@dataclass
class Hdf5FileWriter(FileWriterInterface):
    """File writer using h5py to write out to HDF5-files."""

    def __post_init__(self):
        if self.comm.rank == 0:
            with h5py.File(self.file_name, mode="w") as f:
                for (
                    quantity
                ) in self.time_integration_metadata.time_integration_quantity_list:
                    f.create_group(quantity.name)
                f.create_group("error_measure")

    def write_step(
        self,
        step: int,
        time: float,
        time_integration_data: np.ndarray,
        error_measure: float = None,
    ) -> None:
        with h5py.File(self.file_name, mode="r+", driver="mpio", comm=self.comm) as f:
            for (
                quantity
            ) in self.time_integration_metadata.time_integration_quantity_list:
                dataset = f[quantity.name].create_dataset(
                    str(step),
                    shape=quantity.array_metadata.global_shape,
                    dtype=np.float64,
                )
                dataset.attrs["time"] = time
                if 0 not in quantity.array_metadata.shape:
                    write_indices = self.get_write_indices(quantity)
                    start_array = quantity.array_metadata.start_index
                    end_array = quantity.array_metadata.end_index
                    f[quantity.name][str(step)][write_indices] = time_integration_data[
                        start_array:end_array
                    ].reshape(quantity.array_metadata.shape)
            if error_measure:
                norm_data = f["error_measure"].create_dataset(
                    str(step),
                    shape=(1,),
                    dtype=np.float64,
                )
                norm_data.attrs["time"] = time
                f["error_measure"][str(step)][0] = error_measure

    @staticmethod
    def get_write_indices(quantity: Quantity) -> tuple:
        """Gets indices of where to write the local part of the quantity
        into the respective dataset."""
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
        return tuple(access_list)


class TXTFileWriter(FileWriterInterface):
    """
    File writer to write out to txt-files:
    Does not support parallelism.
    """

    def write_step(
        self,
        step: int,
        time: float,
        time_integration_data: np.ndarray,
        error_measure: float = None,
    ) -> None:
        if self.comm.rank == 0:
            mode = "a"
            if step == 0:
                mode = "w"
            if error_measure:
                data_dict = {"step": step, "time": time, "error_measure": error_measure}
            else:
                data_dict = {"step": step, "time": time}

            with open(self.file_name, mode, encoding="utf-8") as file_out:
                for (
                    quantity
                ) in self.time_integration_metadata.time_integration_quantity_list:
                    if quantity.array_metadata.local:
                        start_array = quantity.array_metadata.start_index
                        end_array = quantity.array_metadata.end_index
                        data_dict[quantity.name] = (
                            time_integration_data[start_array:end_array]
                            .reshape(quantity.array_metadata.shape)
                            .tolist()
                        )
                file_out.write(json.dumps(data_dict) + "\n")


def read_hdf5_file(
    file: str, quantities: list[str], solution: callable
) -> Tuple[dict, dict, dict]:
    """extracts the time, error and result data from an HDF5 file"""
    # Initialize dictionaries
    time = {}
    error_data = {}
    result = {}
    # Open the HDF5 file in read-only mode
    with h5py.File(
        file,
        mode="r",
    ) as f:
        # coefficient values
        group = f[quantities[0]]
        # Extract time metadata
        for key in group.keys():
            time[int(key)] = group[key].attrs["time"]
        # Extract solution and compute Error wrt. analytical solution
        for i, quantity in enumerate(quantities):
            group = f[quantity]
            error_data[quantity] = {}  # initialize error data for each quantity
            result[quantity] = {}
            # Now works for matrices
            for key in group.keys():
                result[quantity][int(key)] = np.zeros(group[key].shape)
                for row in range(group[key].shape[0]):
                    result[quantity][int(key)][row] = group[key][row]

                if len(quantities) > 1:
                    quan_result = [result[quantity][0]] * len(quantities)
                    error_data[quantity][int(key)] = np.abs(
                        solution(
                            time[int(key)],
                            quan_result,
                            time[0],
                        )[i]
                        - result[quantity][int(key)]
                    )
                    print(error_data[quantity][int(key)])
                else:
                    error_data[quantity][int(key)] = np.abs(
                        solution(time[int(key)], result[quantity][0], time[0])
                        - result[quantity][int(key)]
                    )
    return time, error_data, result


def read_last_local_error(file_path):
    with h5py.File(
        file_path,
        mode="r",
    ) as f:
        # Extract time metadata
        # due to floating point precision error a small epsilon is added
        last_step = max([int(x) for x in f["error_measure"].keys()])
        return f["error_measure"][str(last_step)][0]
