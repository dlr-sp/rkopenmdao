"""Defines interface for file writers, as well as the HDF5 file writer."""

from dataclasses import dataclass
from typing import Tuple


import h5py

# pylint doesn't pick it up, but it's there
# pylint: disable = no-name-in-module
from mpi4py.MPI import Comm, COMM_WORLD
import numpy as np

from rkopenmdao.callback import Callback
from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE
from rkopenmdao.metadata_extractor import Quantity
from rkopenmdao.time_discretization.runge_kutta_discretization_state import (
    EmbeddedRungeKuttaDiscretizationState,
)
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
)
from rkopenmdao.time_integration_state import TimeIntegrationState


@dataclass
class OpenMDAOHDF5Callback(Callback):
    """
    File writing callback for problems involving an OpenMDAOODE.

    For each written time step, contains a dataset per quantity in the problem.

    Parameters
    ----------
    filename: str
        Name of the file to be created.
    write_out_period: int = 1
        Period on which time steps the quantity datasets get written.
    comm: Comm = COMM_WORLD
        MPI communicator which needs to match the one used by the problem inside the
        OpenMDAOODE.
    """

    filename: str
    write_out_period: int = 1
    comm: Comm = COMM_WORLD

    def before_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: OpenMDAOODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> None:
        """
        Creates datasets for data of the to be computed in the current step, if
        necessary wrt. to self.write_out_period. In the first iteration, also creates
        datasets for the initial values.

        Parameters
        ----------
        iteration: int
            The current iteration of the time integration.
        time_integration_state: TimeIntegrationState
            Current state of the time integration. Only used currently to differentiate
            between time integrations with and without error control.
        ode: OpenMDAOODE
            OpenMDAO ODE implementation that represents a multidisciplinary problem.
        discretization_scheme: TimeDiscretizationSchemeInterface
            Currently unused. Probably will be necessary to be used once the use of
            arbitrary time discretizations is fully supported by the time integration.
        """
        mode = "w" if iteration == 1 else "r+"
        iter_strs = []
        # In the first iteration, also add dataset for the initial values.
        if iteration == 1:
            iter_strs.append("0")
        # Create datasets if write out is requested via self.write_out_period.
        if iteration % self.write_out_period == 0:
            iter_strs.append(f"{iteration}")
        with h5py.File(self.filename, mode=mode, driver="mpio", comm=self.comm) as hf:
            for iter_str in iter_strs:
                hf.create_group(iter_str)
                hf[iter_str].create_dataset("time", shape=(1,), dtype=np.float64)
                for (
                    quantity
                ) in ode.time_integration_metadata.time_integration_quantity_list:
                    hf[iter_str].create_dataset(
                        quantity.name,
                        shape=quantity.array_metadata.global_shape,
                        dtype=np.float64,
                    )
                if isinstance(
                    time_integration_state.discretization_state,
                    EmbeddedRungeKuttaDiscretizationState,
                ):
                    hf[iter_str].create_dataset(
                        "error_measure",
                        shape=(1,),
                        dtype=np.float64,
                    )

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: OpenMDAOODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> None:
        """
        Writes into the datasets previously created by `before_iteration`, if necessary
        wrt. to self.write_out_period. In the first iteration, also writes out the
        initial values.

        Parameters
        ----------
        iteration: int
            The current iteration of the time integration.
        time_integration_state: TimeIntegrationState
            Current state of the time integration.
        ode: OpenMDAOODE
            OpenMDAO ODE implementation that represents a multidisciplinary problem.
        discretization_scheme: TimeDiscretizationSchemeInterface
            Currently unused. Probably will be necessary to be used once the use of
            arbitrary time discretizations is fully supported by the time integration.
        """
        with h5py.File(self.filename, mode="r+", driver="mpio", comm=self.comm) as hf:
            # Write out initial values.
            if iteration == 1:
                hf["0"]["time"][
                    0
                ] = time_integration_state.discretization_state.start_time
                for (
                    quantity
                ) in ode.time_integration_metadata.time_integration_quantity_list:
                    if 0 not in quantity.array_metadata.shape:
                        write_indices = self.get_write_indices(quantity)
                        start_array = quantity.array_metadata.start_index
                        end_array = quantity.array_metadata.end_index
                        hf["0"][quantity.name][write_indices] = (
                            time_integration_state.discretization_state.start_state[
                                start_array:end_array
                            ].reshape(quantity.array_metadata.shape)
                        )
                if isinstance(
                    time_integration_state.discretization_state,
                    EmbeddedRungeKuttaDiscretizationState,
                ):
                    hf["0"]["error_measure"][0] = 0
            # Write out the result of timestep, if requested.
            if iteration % self.write_out_period == 0:
                hf[f"{iteration}"]["time"][
                    0
                ] = time_integration_state.discretization_state.final_time
                for (
                    quantity
                ) in ode.time_integration_metadata.time_integration_quantity_list:
                    if 0 not in quantity.array_metadata.shape:
                        write_indices = self.get_write_indices(quantity)
                        start_array = quantity.array_metadata.start_index
                        end_array = quantity.array_metadata.end_index
                        hf[f"{iteration}"][quantity.name][write_indices] = (
                            time_integration_state.discretization_state.final_state[
                                start_array:end_array
                            ].reshape(quantity.array_metadata.shape)
                        )
                if isinstance(
                    time_integration_state.discretization_state,
                    EmbeddedRungeKuttaDiscretizationState,
                ):
                    hf[f"{iteration}"]["error_measure"][0] = (
                        time_integration_state.error_history[0]
                    )

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


def read_hdf5_file(
    file: str, quantities: list[str], solution: callable
) -> Tuple[dict, dict, dict]:
    """
    Extracts time, error and result data from an HDF5 file created by
    OpenMDAOHDF5Callback.
    """
    # Initialize dictionaries
    time = {}
    error_data = {}
    result = {}
    # Open the HDF5 file in read-only mode
    with h5py.File(
        file,
        mode="r",
    ) as f:
        # Extract time metadata
        for key, group in f.items():
            time[int(key)] = group["time"][0]
        # Extract solution and compute Error wrt. analytical solution
        for i, quantity in enumerate(quantities):
            error_data[quantity] = {}  # initialize error data for each quantity
            result[quantity] = {}
            # Now works for matrices
            for key, group in f.items():
                result[quantity][int(key)] = np.zeros(group[quantity].shape)
                for row in range(group[quantity].shape[0]):
                    result[quantity][int(key)][row] = group[quantity][row]

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
                else:
                    error_data[quantity][int(key)] = np.abs(
                        solution(time[int(key)], result[quantity][0], time[0])
                        - result[quantity][int(key)]
                    )
    return time, error_data, result


def read_last_local_error(file_path: str):
    """
    Read the last local error measure from a file created by OpenMDAOHDF5Callback.

    Parameters
    ----------
    file_path: str
        Path to the file the error is read from.
    """
    with h5py.File(
        file_path,
        mode="r",
    ) as f:
        last_step = max(int(x) for x in f.keys())
        return f[str(last_step)]["error_measure"][0]
