"""Methods for extracting metadata from the inner OpenMDAO problems of the
RungeKuttaIntegrator used for organizing its own data structures."""

# pylint: disable=protected-access
from typing import Tuple

import numpy as np
import openmdao.api as om

from mpi4py import MPI


def extract_time_integration_metadata(
    stage_problem: om.Problem, time_integration_quantity_list: list
) -> Tuple[int, dict, dict]:
    """Extracts metadata from the time stage problem and returns the size of the
    integrators internal array, a dict to translate to the inner problem, and a dict
    containing variable information."""
    local_quantities = stage_problem.model.get_io_metadata(
        iotypes="output",
        metadata_keys=["tags"],
        tags=time_integration_quantity_list,
        get_remote=False,
    )
    global_quantities = stage_problem.model.get_io_metadata(
        iotypes="output",
        metadata_keys=["tags"],
        tags=time_integration_quantity_list,
        get_remote=True,
    )
    translation_metadata = {
        quantity: {
            "step_input_var": None,
            "accumulated_stage_var": None,
            "stage_output_var": None,
            "postproc_input_var": None,
        }
        for quantity in time_integration_quantity_list
    }
    quantity_metadata = {
        quantity: {
            "type": "time_integration",
            "shape": 0,
            "global_shape": 0,
            "local": False,
            "distributed": False,
            "start_index": 0,
            "end_index": 0,
            "functionally_integrated": False,
            "functional_start_index": 0,
            "functional_end_index": 0,
            "global_start_index": 0,
            "global_end_index": 0,
        }
        for quantity in time_integration_quantity_list
    }
    array_size = 0

    time_integration_set = set(time_integration_quantity_list)
    for var, data in global_quantities.items():
        if var in local_quantities:
            tags = time_integration_set & set(data["tags"])
            assert len(tags) == 1, (
                f"Variable {var} either has two time integration quantity tags, "
                "or 'stage_output_var' was used as quantity tag. Both are forbidden. "
                f"Tags of {var} intersected with time integration quantities: {tags}."
            )

            quantity = tags.pop()
            quantity_metadata[quantity]["local"] = True
            array_size, translation_metadata, quantity_metadata = (
                _extract_detailed_time_integration_info(
                    quantity,
                    stage_problem,
                    array_size,
                    translation_metadata,
                    quantity_metadata,
                )
            )

    return (
        array_size,
        translation_metadata,
        quantity_metadata,
    )


def _extract_detailed_time_integration_info(
    quantity: str,
    stage_problem: om.Problem,
    array_size: int,
    translation_metadata: dict,
    quantity_metadata: dict,
) -> Tuple[int, dict, dict]:
    detailed_local_quantity = stage_problem.model.get_io_metadata(
        metadata_keys=["tags", "shape", "global_shape"],
        tags=[quantity],
        get_remote=False,
    )
    found_stage_output_var = 0
    found_step_input_var = 0
    found_accumulated_stage_var = 0
    for detailed_var, detailed_data in detailed_local_quantity.items():
        if "stage_output_var" in detailed_data["tags"]:
            found_stage_output_var += 1
            translation_metadata[quantity]["stage_output_var"] = detailed_var
            quantity_metadata[quantity]["shape"] = detailed_data["shape"]
            quantity_metadata[quantity]["global_shape"] = detailed_data["global_shape"]
            quantity_metadata[quantity]["start_index"] = array_size
            array_size += np.prod(detailed_data["shape"])
            quantity_metadata[quantity]["end_index"] = array_size
            if detailed_data["shape"] != detailed_data["global_shape"]:
                try:
                    sizes = stage_problem.model._var_sizes["output"][
                        :,
                        stage_problem.model._var_allprocs_abs2idx[detailed_var],
                    ]
                except KeyError:  # Have to deal with an older openMDAO version
                    sizes = stage_problem.model._var_sizes["nonlinear"]["output"][
                        :,
                        stage_problem.model._var_allprocs_abs2idx["nonlinear"][
                            detailed_var
                        ],
                    ]
                quantity_metadata[quantity]["global_start_index"] = start = np.sum(
                    sizes[: stage_problem.comm.rank]
                )
                quantity_metadata[quantity]["global_end_index"] = (
                    start + sizes[stage_problem.comm.rank]
                )
            else:
                quantity_metadata[quantity]["global_start_index"] = 0
                quantity_metadata[quantity]["global_end_index"] = np.prod(
                    detailed_data["shape"]
                )
        elif "step_input_var" in detailed_data["tags"]:
            found_step_input_var += 1
            translation_metadata[quantity]["step_input_var"] = detailed_var
        elif "accumulated_stage_var" in detailed_data["tags"]:
            found_accumulated_stage_var += 1
            translation_metadata[quantity]["accumulated_stage_var"] = detailed_var
    assert found_stage_output_var <= 1, (
        f"For quantity {quantity}, there is more than one inner variable tagged"
        f" with 'stage_output_var'."
    )

    assert found_step_input_var <= 1, (
        f"For quantity {quantity}, there is more than one inner variable tagged"
        f" with 'step_input_var'."
    )

    assert found_accumulated_stage_var <= 1, (
        f"For quantity {quantity}, there is more than one inner variable tagged"
        f" with 'accumulated_stage_var'."
    )

    assert found_stage_output_var >= 1, (
        f"For quantity {quantity}, there is no inner variable tagged with "
        f"'stage_output_var'."
    )

    assert (found_step_input_var, found_accumulated_stage_var) in [
        (0, 0),
        (1, 1),
    ], (
        f"For quantity {quantity}, there is either a variable tagged for"
        f" 'step_input_var, but not 'accumulated_stage_var', or vice versa. "
        f"Either none or both have to be present."
    )
    return array_size, translation_metadata, quantity_metadata


def add_postprocessing_metadata(
    postproc_problem: om.Problem,
    time_integration_quantity_list: list,
    postprocessing_quantity_list: list,
    quantity_metadata: dict,
    translation_metadata: dict,
) -> Tuple[int, dict, dict]:
    """Extracts metadata from the postprocessing problem and adds it to the passed
    metadata dicts. Returns them, as well as the size for the internal postprocessing
    state array."""
    postproc_input_vars = postproc_problem.model.get_io_metadata(
        iotypes="input",
        metadata_keys=["tags"],
        tags=["postproc_input_var"],
        get_remote=False,
    )
    time_integration_set = set(time_integration_quantity_list)
    postprocessing_set = set(postprocessing_quantity_list)
    assert time_integration_set.isdisjoint(
        postprocessing_set
    ), "time integration and postprocessing quantities have to be disjoint"

    translation_metadata.update(
        {
            quantity: {
                "postproc_output_var": None,
            }
            for quantity in postprocessing_quantity_list
        }
    )
    quantity_metadata.update(
        {
            quantity: {
                "type": "postprocessing",
                "shape": 0,
                "global_shape": 0,
                "local": False,
                "distributed": False,
                "start_index": 0,
                "end_index": 0,
                "functionally_integrated": False,
                "functional_start_index": 0,
                "functional_end_index": 0,
                "global_start_index": 0,
                "global_end_index": 0,
            }
            for quantity in postprocessing_quantity_list
        }
    )

    for var, data in postproc_input_vars.items():
        tags = data["tags"] & time_integration_set
        assert len(tags) == 1, (
            f"Variable {var} either has two time integration quantity tags, or "
            f"'postproc_input_var' was used as quantity tag. Both are forbidden. Tags "
            f"of {var} intersected with time integration quantities: {tags}."
        )

        quantity = tags.pop()

        translation_metadata = _extract_detailed_postprocessing_input_info(
            quantity, postproc_problem, translation_metadata
        )

    local_postproc_quantities = postproc_problem.model.get_io_metadata(
        iotypes="output",
        metadata_keys=["tags"],
        tags=postprocessing_quantity_list,
        get_remote=False,
    )
    global_postproc_quantities = postproc_problem.model.get_io_metadata(
        iotypes="output",
        metadata_keys=["tags"],
        tags=postprocessing_quantity_list,
        get_remote=True,
    )
    array_size = 0

    for var, data in global_postproc_quantities.items():
        if var in local_postproc_quantities:
            tags = postprocessing_set & set(data["tags"])
            assert len(tags) == 1, (
                f"Variable {var} either has two postprocessing quantity tags, or "
                f"'postproc_output_var' was used as quantity tag. Both are forbidden. "
                f"Tags of {var} intersected with time integration quantities: {tags}."
            )

            quantity = tags.pop()
            quantity_metadata[quantity]["local"] = True

            array_size, translation_metadata, quantity_metadata = (
                _extract_detailed_postprocessing_output_info(
                    quantity,
                    postproc_problem,
                    array_size,
                    translation_metadata,
                    quantity_metadata,
                )
            )

    return (
        array_size,
        translation_metadata,
        quantity_metadata,
    )


def _extract_detailed_postprocessing_input_info(
    quantity: str, postproc_problem: om.Problem, translation_metadata
) -> dict:
    detailed_local_quantity = postproc_problem.model.get_io_metadata(
        metadata_keys=["tags"],
        tags=[quantity],
        get_remote=False,
    )
    found_postproc_input_var = 0
    # At least one variable will be found in here, so found_postproc_input_var will
    # be greater than zero
    for detailed_var, detailed_data in detailed_local_quantity.items():
        if "postproc_input_var" in detailed_data["tags"]:
            found_postproc_input_var += 1
            translation_metadata[quantity]["postproc_input_var"] = detailed_var
    assert found_postproc_input_var == 1, (
        f"More than one variable with quantity tag {quantity} for "
        f"'postproc_input_var' in postprocessing_problem."
    )
    return translation_metadata


def _extract_detailed_postprocessing_output_info(
    quantity: str,
    postproc_problem: om.Problem,
    array_size: int,
    translation_metadata: dict,
    quantity_metadata: dict,
) -> Tuple[int, dict, dict]:
    detailed_local_quantity = postproc_problem.model.get_io_metadata(
        metadata_keys=["tags", "shape", "global_shape"],
        tags=[quantity],
        get_remote=False,
    )
    found_postproc_output_var = 0
    for detailed_var, detailed_data in detailed_local_quantity.items():
        if "postproc_output_var" in detailed_data["tags"]:
            found_postproc_output_var += 1
            translation_metadata[quantity]["postproc_output_var"] = detailed_var
            quantity_metadata[quantity]["shape"] = detailed_data["shape"]
            quantity_metadata[quantity]["global_shape"] = detailed_data["global_shape"]
            quantity_metadata[quantity]["start_index"] = array_size
            array_size += np.prod(detailed_data["shape"])
            quantity_metadata[quantity]["end_index"] = array_size
            if detailed_data["shape"] != detailed_data["global_shape"]:
                try:
                    sizes = postproc_problem.model._var_sizes["output"][
                        :,
                        postproc_problem.model._var_allprocs_abs2idx[detailed_var],
                    ]
                except KeyError:  # Have to deal with an older openMDAO version
                    sizes = postproc_problem.model._var_sizes["nonlinear"]["output"][
                        :,
                        postproc_problem.model._var_allprocs_abs2idx["nonlinear"][
                            detailed_var
                        ],
                    ]
                quantity_metadata[quantity]["global_start_index"] = start = np.sum(
                    sizes[: postproc_problem.comm.rank]
                )
                quantity_metadata[quantity]["global_end_index"] = (
                    start + sizes[postproc_problem.comm.rank]
                )
            else:
                quantity_metadata[quantity]["global_start_index"] = 0
                quantity_metadata[quantity]["global_end_index"] = np.prod(
                    detailed_data["shape"]
                )
    assert found_postproc_output_var <= 1, (
        f"For quantity {quantity}, there is more than one inner variable tagged"
        f" with 'postproc_output_var'."
    )

    assert found_postproc_output_var >= 1, (
        f"For quantity {quantity}, there is no inner variable tagged with "
        f"'postproc_output_vars'."
    )
    return array_size, translation_metadata, quantity_metadata


def add_functional_metadata(
    functional_quantities: list, quantity_metadata: dict
) -> Tuple[int, dict]:
    """Adds metadata in the quantity metadata dict based on which functionals are part
    of a functional. Also returns the size of the internal functional state array."""
    assert set(functional_quantities).issubset(set(quantity_metadata.keys())), (
        "Some functional requires a quantity that is part of neither the time "
        "integration nor postprocessing."
    )

    array_size = 0
    for quantity in functional_quantities:
        quantity_metadata[quantity]["functionally_integrated"] = True
        if quantity_metadata[quantity]["local"]:
            quantity_metadata[quantity]["functional_start_index"] = array_size
            array_size += np.prod(quantity_metadata[quantity]["shape"])
            quantity_metadata[quantity]["functional_end_index"] = array_size
    return array_size, quantity_metadata


def add_distributivity_information(
    stage_problem: om.Problem, quantity_metadata: dict
) -> None:
    """Adds information about the distributed structure of data the the quantity
    metadata dict."""
    for metadata in quantity_metadata.values():
        local = np.array([metadata["local"]])
        everywhere_local = np.zeros_like(local)
        # pylint: disable=c-extension-no-member
        stage_problem.comm.Allreduce(local, everywhere_local, MPI.LAND)
        if everywhere_local:
            metadata["distributed"] = metadata["shape"] != metadata["global_shape"]
        else:
            metadata["distributed"] = True


# TODO: we will later want to differentiate the algebraic part of DAEs (to take
#  advantage of the FSAL property of some butcher tableaux/RK schemes)
# def add_algebraic_metadata(
#     stage_problem: om.Problem,
#     time_integration_quantity_list: list,
#     quantity_metadata: dict,
#     translation_metadata: dict,
# ):
#     pass

# TODO: we will later want the ability to passthrough parameters to the outside problem
#  for optimization
# def add_passthrough_metadata(
#     stage_problem: om.Problem,
#     time_integration_quantity_list: list,
#     quantity_metadata: dict,
#     translation_metadata: dict,
# ):
#     pass
