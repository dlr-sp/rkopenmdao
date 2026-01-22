"""Methods for extracting metadata from the inner OpenMDAO problems of the
RungeKuttaIntegrator used for organizing its own data structures."""

# pylint: disable=protected-access
# pylint: disable = c-extension-no-member
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import openmdao.api as om

from mpi4py import MPI

from .errors import SetupError


@dataclass
class TranslationMetadata(ABC):
    """Abstract interface for translation metadata"""


@dataclass
class TimeIntegrationTranslationMetadata(TranslationMetadata):
    """Translation metadata for time integration quantities."""

    step_input_var: str | None = None
    accumulated_stage_var: str | None = None
    stage_output_var: str | None = None


@dataclass
class TimeIndependentInputTranslationMetadata(TranslationMetadata):
    """Translation metadata for time independent input quantities"""

    time_independent_input_var: str | None = None


@dataclass
class ArrayMetadata:
    """Metadata concerning properties of arrays of quantities."""

    shape: tuple = (0,)
    global_shape: tuple = (0,)
    local: bool = False
    distributed: bool = False
    start_index: int = 0
    end_index: int = 0
    global_start_index: int = 0
    global_end_index: int = 0


@dataclass
class Quantity:
    """Base class for all quantity types. Contains all the necessary information about a
    quantity."""

    name: str
    type: str
    array_metadata: ArrayMetadata
    translation_metadata: TranslationMetadata


@dataclass
class TimeIntegrationQuantity(Quantity):
    """Class for time integrated quantities."""

    translation_metadata: TimeIntegrationTranslationMetadata


@dataclass
class TimeIndependentQuantity(Quantity):
    """Class for time independent input quantities."""

    translation_metadata: TimeIndependentInputTranslationMetadata


@dataclass
class TimeIntegrationMetadata:
    """Collection of metadata over all quantities of a time integration."""

    time_integration_array_size: int = 0
    time_independent_input_size: int = 0
    time_integration_quantity_list: list[TimeIntegrationQuantity] = field(
        default_factory=list
    )
    time_independent_input_quantity_list: list[TimeIndependentQuantity] = field(
        default_factory=list
    )


def extract_time_integration_metadata(
    stage_problem: om.Problem, time_integration_quantity_list: list
) -> TimeIntegrationMetadata:
    """Extracts metadata from the time stage problem and returns it inside a
    TimeIntegrationMetadata object."""
    local_quantities = stage_problem.model.get_io_metadata(
        iotypes="output",
        metadata_keys=["tags"],
        tags=time_integration_quantity_list,
        get_remote=False,
    )
    global_quantities = stage_problem.model.get_io_metadata(
        iotypes="output",
        metadata_keys=["tags", "global_shape"],
        tags=time_integration_quantity_list,
        get_remote=True,
    )
    array_size = 0
    quantity_list = []

    time_integration_set = set(time_integration_quantity_list)
    for var, data in global_quantities.items():
        tags = time_integration_set & set(data["tags"])

        if len(tags) != 1:
            raise SetupError(
                f"Variable {var} either has two time integration quantity tags, or "
                f"'stage_output_var' was used as quantity tag. Both are forbidden. "
                f"Tags of {var} intersected with time integration quantities: "
                f"{tags}."
            )
        quantity_name = tags.pop()
        if var in local_quantities:
            array_size, quantity = _extract_time_integration_quantity(
                quantity_name,
                stage_problem,
                array_size,
            )
        else:
            quantity = TimeIntegrationQuantity(
                type="time_integration",
                name=quantity_name,
                array_metadata=ArrayMetadata(global_shape=data["global_shape"]),
                translation_metadata=TimeIntegrationTranslationMetadata(),
            )
        quantity_list.append(quantity)
    runge_kutta_metadata = TimeIntegrationMetadata(
        time_integration_array_size=array_size,
        time_integration_quantity_list=quantity_list,
    )
    return runge_kutta_metadata


def _create_local_array_metadata(
    stage_problem: om.Problem,
    start_index: int,
    end_index: int,
    detailed_var: str,
    detailed_data: dict,
) -> ArrayMetadata:
    if detailed_data["shape"] != detailed_data["global_shape"]:
        try:
            sizes = stage_problem.model._var_sizes["output"][
                :,
                stage_problem.model._var_allprocs_abs2idx[detailed_var],
            ]
        except KeyError:  # Have to deal with an older openMDAO version
            sizes = stage_problem.model._var_sizes["nonlinear"]["output"][
                :,
                stage_problem.model._var_allprocs_abs2idx["nonlinear"][detailed_var],
            ]
        global_start_index = start = np.sum(sizes[: stage_problem.comm.rank])
        global_end_index = start + sizes[stage_problem.comm.rank]
    else:
        global_start_index = 0
        global_end_index = np.prod(detailed_data["shape"])
    return ArrayMetadata(
        shape=detailed_data["shape"],
        global_shape=detailed_data["global_shape"],
        local=True,
        start_index=start_index,
        end_index=end_index,
        global_start_index=global_start_index,
        global_end_index=global_end_index,
    )


def _extract_time_integration_quantity(
    quantity_name: str,
    stage_problem: om.Problem,
    array_size: int,
) -> tuple[int, TimeIntegrationQuantity]:
    detailed_local_quantity = stage_problem.model.get_io_metadata(
        metadata_keys=["tags", "shape", "global_shape"],
        tags=[quantity_name],
        get_remote=False,
    )
    found_stage_output_var = 0
    found_step_input_var = 0
    found_accumulated_stage_var = 0
    translation_metadata = TimeIntegrationTranslationMetadata()
    for detailed_var, detailed_data in detailed_local_quantity.items():
        if "stage_output_var" in detailed_data["tags"]:
            found_stage_output_var += 1
            translation_metadata.stage_output_var = detailed_var
            start_index = array_size
            array_size += np.prod(detailed_data["shape"])
            end_index = array_size
            array_metadata = _create_local_array_metadata(
                stage_problem=stage_problem,
                start_index=start_index,
                end_index=end_index,
                detailed_var=detailed_var,
                detailed_data=detailed_data,
            )
        elif "step_input_var" in detailed_data["tags"]:
            found_step_input_var += 1
            translation_metadata.step_input_var = detailed_var
        elif "accumulated_stage_var" in detailed_data["tags"]:
            found_accumulated_stage_var += 1
            translation_metadata.accumulated_stage_var = detailed_var

    if found_stage_output_var > 1:
        raise SetupError(
            f"For quantity {quantity_name}, there is more than one inner variable "
            f"tagged with 'stage_output_var'."
        )
    if found_step_input_var > 1:
        raise SetupError(
            f"For quantity {quantity_name}, there is more than one inner variable "
            f"tagged with 'step_input_var'."
        )
    if found_accumulated_stage_var > 1:
        raise SetupError(
            f"For quantity {quantity_name}, there is more than one inner variable "
            f"tagged with 'accumulated_stage_var'."
        )
    if found_stage_output_var < 1:
        raise SetupError(
            f"For quantity {quantity_name}, there is no inner variable tagged with "
            f"'stage_output_var'."
        )
    if (found_step_input_var, found_accumulated_stage_var) not in [
        (0, 0),
        (1, 1),
    ]:
        raise SetupError(
            f"For quantity {quantity_name}, there is either a variable tagged for"
            f" 'step_input_var, but not 'accumulated_stage_var', or vice versa. "
            f"Either none or both have to be present."
        )

    # We check that there is a stage_output_var, so array_metadata must exist
    # pylint: disable=possibly-used-before-assignment
    quantity = TimeIntegrationQuantity(
        name=quantity_name,
        type="time_integration",
        array_metadata=array_metadata,
        translation_metadata=translation_metadata,
    )

    return array_size, quantity


def add_time_independent_input_metadata(
    stage_problem: om.Problem,
    time_independent_input_quantity_list: list,
    runge_kutta_metadata: TimeIntegrationMetadata,
):
    """Extracts metadata of independent inputs from the stage problem and adds it to the
    passed TimeIntegrationMetadata object."""
    local_quantities = stage_problem.model.get_io_metadata(
        iotypes="input",
        metadata_keys=["tags"],
        tags=time_independent_input_quantity_list,
        get_remote=False,
    )
    global_quantities = stage_problem.model.get_io_metadata(
        iotypes="input",
        metadata_keys=["tags", "global_shape"],
        tags=time_independent_input_quantity_list,
        get_remote=True,
    )

    time_independent_input_set = set(time_independent_input_quantity_list)
    for var, data in global_quantities.items():
        tags = data["tags"] & time_independent_input_set

        if len(tags) != 1:
            raise SetupError(
                f"Variable {var} either has two time independent quantity tags, "
                f"or 'time_independent_input_var' was used as quantity tag. Both are "
                f"forbidden. Tags of {var} intersected with time independent input "
                f"quantities: {tags}."
            )
        quantity_name = tags.pop()

        if var in local_quantities:

            runge_kutta_metadata.time_independent_input_size, quantity = (
                _extract_time_independent_quantity(
                    quantity_name,
                    stage_problem,
                    runge_kutta_metadata.time_independent_input_size,
                )
            )
        else:
            quantity = TimeIndependentQuantity(
                type="independent_input",
                name=quantity_name,
                array_metadata=ArrayMetadata(global_shape=data["global_shape"]),
                translation_metadata=TimeIndependentInputTranslationMetadata(),
            )
        runge_kutta_metadata.time_independent_input_quantity_list.append(quantity)


def _extract_time_independent_quantity(
    quantity_name: str,
    stage_problem: om.Problem,
    array_size,
) -> tuple[int, TimeIndependentQuantity]:
    detailed_local_quantity = stage_problem.model.get_io_metadata(
        metadata_keys=["tags", "shape", "global_shape"],
        tags=[quantity_name],
        get_remote=False,
    )
    found_time_independent_input_var = 0
    translation_metadata = TimeIndependentInputTranslationMetadata()
    for detailed_var, detailed_data in detailed_local_quantity.items():
        if "time_independent_input_var" in detailed_data["tags"]:
            found_time_independent_input_var += 1
            translation_metadata.time_independent_input_var = detailed_var
            start_index = array_size
            array_size += np.prod(detailed_data["shape"])
            end_index = array_size
            array_metadata = _create_local_array_metadata(
                stage_problem=stage_problem,
                start_index=start_index,
                end_index=end_index,
                detailed_var=detailed_var,
                detailed_data=detailed_data,
            )

    if found_time_independent_input_var > 1:
        raise SetupError(
            f"For quantity {quantity_name}, there is more than one inner variable "
            f"tagged with 'time_independent_input_var'."
        )

    if found_time_independent_input_var < 1:
        raise SetupError(
            f"For quantity {quantity_name}, there is no inner variable tagged with "
            f"'time_independent_input_var'."
        )
    # We check that there is a time_independent_input_var, so array_metadata must exist
    # pylint: disable=possibly-used-before-assignment
    quantity = TimeIndependentQuantity(
        name=quantity_name,
        type="independent_input",
        array_metadata=array_metadata,
        translation_metadata=translation_metadata,
    )

    return array_size, quantity


def add_distributivity_information(
    stage_problem: om.Problem, runge_kutta_metadata: TimeIntegrationMetadata
) -> None:
    """Adds information about the distributed structure of data to the passed
    TimeIntegrationMetadata object."""
    for quantity in chain(
        runge_kutta_metadata.time_integration_quantity_list,
        runge_kutta_metadata.time_independent_input_quantity_list,
    ):
        shape_matches_locally = (
            quantity.array_metadata.shape == quantity.array_metadata.global_shape
        )
        shapes_match_everywhere = stage_problem.comm.allreduce(
            shape_matches_locally, MPI.LAND
        )
        quantity.array_metadata.distributed = not shapes_match_everywhere
        # local = np.array(quantity.array_metadata.local)
        # everywhere_local = np.zeros_like(local)
        # # pylint: disable=c-extension-no-member
        # stage_problem.comm.Allreduce(local, everywhere_local, MPI.LAND)
        # if everywhere_local:
        #     # If a variable is local everywhere, it is distributed if and only if the
        #     # local and global shape don't match.
        #     quantity.array_metadata.distributed = (
        #         quantity.array_metadata.shape != quantity.array_metadata.global_shape
        #     )
        # else:
        #     # If a variable is not local, it must be somewhere else, and therefore is
        #     # distributed
        #     quantity.array_metadata.distributed = True


# TODO: we will later want to differentiate the algebraic part of DAEs (to take
#  advantage of the FSAL property of some butcher tableaux/RK schemes)
# def add_algebraic_metadata(
#     stage_problem: om.Problem,
#     time_integration_quantity_list: list,
#     quantity_metadata: dict,
#     translation_metadata: dict,
# ):
#     pass
