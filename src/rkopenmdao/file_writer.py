"""Defines interface for file writers, as well as the HDF5 file writer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
import json

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
        postprocessing_data: np.ndarray,
    ) -> None:
        """Writes out step to file"""


@dataclass
class Hdf5FileWriter(FileWriterInterface):
    """File writer using h5py to write out to HDF5-files."""

    def __post_init__(self):
        if self.comm.rank == 0:
            written_out_quantities = chain(
                self.time_integration_metadata.time_integration_quantity_list,
                self.time_integration_metadata.postprocessing_quantity_list,
            )
            with h5py.File(self.file_name, mode="w") as f:
                for quantity in written_out_quantities:
                    f.create_group(quantity.name)
                f.create_group("Norm")

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

    def write_step(
        self,
        step: int,
        time: float,
        time_integration_data: np.ndarray,
        postprocessing_data: np.ndarray,
        norm: float = None,
    ) -> None:
        data_map = {
            "time_integration": time_integration_data,
            "postprocessing": postprocessing_data,
        }
        with h5py.File(self.file_name, mode="r+", driver="mpio", comm=self.comm) as f:
            for quantity in chain(
                self.time_integration_metadata.time_integration_quantity_list,
                self.time_integration_metadata.postprocessing_quantity_list,
            ):
                dataset = f[quantity.name].create_dataset(
                    str(step),
                    shape=quantity.array_metadata.global_shape,
                    dtype=np.float64,
                )
                dataset.attrs["time"] = time
                if quantity.array_metadata.local:
                    write_indices = self.get_write_indices(quantity)
                    start_array = quantity.array_metadata.start_index
                    end_array = quantity.array_metadata.end_index
                    f[quantity.name][str(step)][write_indices] = data_map[
                        quantity.type
                    ][start_array:end_array].reshape(quantity.array_metadata.shape)
            if norm:
                norm_data = f["Norm"].create_dataset(
                    str(step),
                    shape=(1,),
                    dtype=np.float64,
                )
                norm_data.attrs["time"] = time
                f["Norm"][str(step)][0] = norm


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
        postprocessing_data: np.ndarray,
        norm: float = None,
    ) -> None:
        if self.comm.rank == 0:
            mode = "a"
            if step == 0:
                mode = "w"
            data_map = {
                "time_integration": time_integration_data,
                "postprocessing": postprocessing_data,
            }
            data_dict = {"step": step, "time": time}

            with open(self.file_name, mode, encoding="utf-8") as file_out:
                for quantity in chain(
                    self.time_integration_metadata.time_integration_quantity_list,
                    self.time_integration_metadata.postprocessing_quantity_list,
                ):
                    if quantity.array_metadata.local:
                        start_array = quantity.array_metadata.start_index
                        end_array = quantity.array_metadata.end_index
                        data_dict[quantity.name] = (
                            data_map[quantity.type][start_array:end_array]
                            .reshape(quantity.array_metadata.shape)
                            .tolist()
                        )
                file_out.write(json.dumps(data_dict) + "\n")
