"""Tests for extraction of metadata from OpenMDAO problems."""

# pylint: disable=c-extension-no-member

from mpi4py import MPI
import openmdao.api as om
import pytest

from rkopenmdao.metadata_extractor import (
    extract_time_integration_metadata,
    add_postprocessing_metadata,
    add_functional_metadata,
    add_distributivity_information,
)


class MetadataTestComponent(om.ExplicitComponent):
    """Helper class to create a component with specific in/output properties. No
    computations are done."""

    def initialize(self):
        self.options.declare(
            "input_dict", types=dict, default={}, desc="Inputs with tags"
        )
        self.options.declare(
            "output_dict", types=dict, default={}, desc="Outputs with tags"
        )

    def setup(self):
        for var, metadata in self.options["input_dict"].items():
            self.add_input(
                var,
                tags=metadata["tags"],
                shape=metadata["shape"],
                distributed=metadata["distributed"],
            )
        for var, metadata in self.options["output_dict"].items():
            self.add_output(
                var,
                tags=metadata["tags"],
                shape=metadata["shape"],
                distributed=metadata["distributed"],
            )


def basic_test_problem():
    """Sets up the basic test problem that is reused for many tests here."""
    input_dict = {
        "x_old": {
            "tags": ["step_input_var", "x"],
            "shape": (5, 2),
            "distributed": False,
        },
        "x_acc_stages": {
            "tags": ["accumulated_stage_var", "x"],
            "shape": (5, 2),
            "distributed": False,
        },
    }
    output_dict = {
        "x_update": {
            "tags": ["stage_output_var", "x"],
            "shape": (5, 2),
            "distributed": False,
        },
    }
    test_comp = MetadataTestComponent(input_dict=input_dict, output_dict=output_dict)
    prob = om.Problem()
    prob.model.add_subsystem("test_comp", test_comp, promotes=["*"])
    prob.setup()
    return prob


def test_metadata_non_parallel_correct():
    """Tests for the sequential case whether time integration metadata is correctly
    set."""
    prob = basic_test_problem()

    array_size, translation_metadata, quantity_metadata = (
        extract_time_integration_metadata(prob, ["x"])
    )

    assert array_size == 10
    assert translation_metadata == {
        "x": {
            "step_input_var": "test_comp.x_old",
            "accumulated_stage_var": "test_comp.x_acc_stages",
            "stage_output_var": "test_comp.x_update",
            "postproc_input_var": None,
        }
    }
    assert quantity_metadata == {
        "x": {
            "type": "time_integration",
            "shape": (5, 2),
            "global_shape": (5, 2),
            "local": True,
            "distributed": False,
            "start_index": 0,
            "end_index": 10,
            "functionally_integrated": False,
            "functional_start_index": 0,
            "functional_end_index": 0,
            "global_start_index": 0,
            "global_end_index": 10,
        }
    }


@pytest.mark.parametrize(
    "input_dict, output_dict, quantity_list, error_message",
    [
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_update_2": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x"],
            "For quantity x, there is more than one inner variable tagged with "
            "'stage_output_var'.",
        ),
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_old_2": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x"],
            "For quantity x, there is more than one inner variable tagged with "
            "'step_input_var'.",
        ),
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages_2": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x"],
            "For quantity x, there is more than one inner variable tagged with "
            "'accumulated_stage_var'.",
        ),
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var_error", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x"],
            "For quantity x, there is no inner variable tagged with "
            "'stage_output_var'.",
        ),
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x"],
            "For quantity x, there is either a variable tagged for 'step_input_var, "
            "but not 'accumulated_stage_var', or vice versa. Either none or both have "
            "to be present.",
        ),
        (
            {
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x"],
            "For quantity x, there is either a variable tagged for 'step_input_var, "
            "but not 'accumulated_stage_var', or vice versa. Either none or both have "
            "to be present.",
        ),
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x", "y"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["x", "y"],
            "Variable test_comp.x_update either has two time integration quantity tags,"
            " or 'stage_output_var' was used as quantity tag. Both are forbidden. Tags "
            "of test_comp.x_update intersected with time integration quantities: "
            "({'y', 'x'}|{'x', 'y'}).",
        ),
        (
            {
                "x_old": {
                    "tags": ["step_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x_acc_stages": {
                    "tags": ["accumulated_stage_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "x_update": {
                    "tags": ["stage_output_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            ["stage_output_var", "x"],
            "Variable test_comp.x_update either has two time integration quantity tags,"
            " or 'stage_output_var' was used as quantity tag. Both are forbidden. Tags "
            "of test_comp.x_update intersected with time integration quantities: "
            "({'x', 'stage_output_var'}|{'stage_output_var', 'x'}).",
        ),
    ],
)
def test_metadata_non_parallel_incorrect(
    input_dict, output_dict, quantity_list, error_message
):
    """Tests various incorrect cases for the setup of the time integration metadata."""
    test_comp = MetadataTestComponent(input_dict=input_dict, output_dict=output_dict)
    prob = om.Problem()
    prob.model.add_subsystem("test_comp", test_comp, promotes=["*"])
    prob.setup()
    with pytest.raises(AssertionError, match=error_message):
        extract_time_integration_metadata(prob, quantity_list)


def test_metadata_functional_correct():
    """Tests whether functional metadata is correctly added to the metadata dict."""
    prob = basic_test_problem()
    _, _, quantity_metadata = extract_time_integration_metadata(prob, ["x"])
    functional_size, quantity_metadata = add_functional_metadata(
        ["x"], quantity_metadata
    )
    assert functional_size == 10
    assert quantity_metadata == {
        "x": {
            "type": "time_integration",
            "shape": (5, 2),
            "global_shape": (5, 2),
            "local": True,
            "distributed": False,
            "start_index": 0,
            "end_index": 10,
            "functionally_integrated": True,
            "functional_start_index": 0,
            "functional_end_index": 10,
            "global_start_index": 0,
            "global_end_index": 10,
        }
    }


def test_metadata_functional_incorrect():
    """Tests an incorrect case for the addition of functional metadata."""
    prob = basic_test_problem()
    _, _, quantity_metadata = extract_time_integration_metadata(prob, ["x"])
    with pytest.raises(
        AssertionError,
        match="Some functional requires a quantity that is part of neither the time "
        "integration nor postprocessing.",
    ):
        add_functional_metadata(["y"], quantity_metadata)


def test_metadata_postprocessing_correct():
    """Tests the correct case for addition of postprocessing metadata to the metadata
    dicts."""
    prob = basic_test_problem()
    input_dict = {
        "x": {
            "tags": ["postproc_input_var", "x"],
            "shape": (5, 2),
            "distributed": False,
        },
    }
    output_dict = {
        "y": {
            "tags": ["postproc_output_var", "y"],
            "shape": (1, 2),
            "distributed": False,
        },
    }
    postproc_comp = MetadataTestComponent(
        input_dict=input_dict, output_dict=output_dict
    )
    postproc_prob = om.Problem()
    postproc_prob.model.add_subsystem("postproc_test", postproc_comp, promotes=["*"])
    _, translation_metadata, quantity_metadata = extract_time_integration_metadata(
        prob, ["x"]
    )
    postproc_prob.setup()

    postproc_array_size, translation_metadata, quantity_metadata = (
        add_postprocessing_metadata(
            postproc_prob, ["x"], ["y"], quantity_metadata, translation_metadata
        )
    )

    assert postproc_array_size == 2
    assert translation_metadata == {
        "x": {
            "step_input_var": "test_comp.x_old",
            "accumulated_stage_var": "test_comp.x_acc_stages",
            "stage_output_var": "test_comp.x_update",
            "postproc_input_var": "postproc_test.x",
        },
        "y": {
            "postproc_output_var": "postproc_test.y",
        },
    }
    assert quantity_metadata == {
        "x": {
            "type": "time_integration",
            "shape": (5, 2),
            "global_shape": (5, 2),
            "local": True,
            "distributed": False,
            "start_index": 0,
            "end_index": 10,
            "functionally_integrated": False,
            "functional_start_index": 0,
            "functional_end_index": 0,
            "global_start_index": 0,
            "global_end_index": 10,
        },
        "y": {
            "type": "postprocessing",
            "shape": (1, 2),
            "global_shape": (1, 2),
            "local": True,
            "distributed": False,
            "start_index": 0,
            "end_index": 2,
            "functionally_integrated": False,
            "functional_start_index": 0,
            "functional_end_index": 0,
            "global_start_index": 0,
            "global_end_index": 2,
        },
    }


@pytest.mark.parametrize(
    "input_dict, output_dict, quantity_list, error_message",
    [
        (
            {
                "x": {
                    "tags": ["postproc_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
                "x'": {
                    "tags": ["postproc_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "y": {
                    "tags": ["postproc_output_var", "y"],
                    "shape": (1, 2),
                    "distributed": False,
                },
            },
            ["y"],
            "More than one variable with quantity tag x for 'postproc_input_var' in "
            "postprocessing_problem.",
        ),
        (
            {
                "x": {
                    "tags": ["postproc_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "y": {
                    "tags": ["postproc_output_var", "y", "z"],
                    "shape": (1, 2),
                    "distributed": False,
                },
            },
            ["y", "z"],
            "Variable postproc_test.y either has two postprocessing quantity tags, or "
            "'postproc_output_var' was used as quantity tag. Both are forbidden. Tags "
            "of postproc_test.y intersected with time integration quantities: "
            "({'y', 'z'}|{'z', 'y'}).",
        ),
        (
            {
                "x": {
                    "tags": ["postproc_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "y": {
                    "tags": [
                        "postproc_output_var",
                        "y",
                    ],
                    "shape": (1, 2),
                    "distributed": False,
                },
            },
            ["postproc_output_var", "y"],
            "Variable postproc_test.y either has two postprocessing quantity tags, or "
            "'postproc_output_var' was used as quantity tag. Both are forbidden. Tags "
            "of postproc_test.y intersected with time integration quantities:"
            " ({'y', 'postproc_output_var'}|{'postproc_output_var', 'y'}).",
        ),
        (
            {
                "x": {
                    "tags": ["postproc_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "y": {
                    "tags": [
                        "postproc_output_var",
                        "y",
                    ],
                    "shape": (1, 2),
                    "distributed": False,
                },
                "y'": {
                    "tags": [
                        "postproc_output_var",
                        "y",
                    ],
                    "shape": (1, 2),
                    "distributed": False,
                },
            },
            ["y"],
            "For quantity y, there is more than one inner variable tagged with "
            "'postproc_output_var'.",
        ),
        (
            {
                "x": {
                    "tags": ["postproc_input_var", "x"],
                    "shape": (5, 2),
                    "distributed": False,
                },
            },
            {
                "y": {
                    "tags": [
                        "postproc_output_var_error",
                        "y",
                    ],
                    "shape": (1, 2),
                    "distributed": False,
                },
            },
            ["y"],
            "For quantity y, there is no inner variable tagged with "
            "'postproc_output_vars'.",
        ),
    ],
)
def test_metadata_postprocessing_incorrect(
    input_dict, output_dict, quantity_list, error_message
):
    """Tests various incorrect cases for the addition of postprocessing metadata to the
    dicts."""
    prob = basic_test_problem()
    postproc_comp = MetadataTestComponent(
        input_dict=input_dict, output_dict=output_dict
    )
    postproc_prob = om.Problem()
    postproc_prob.model.add_subsystem("postproc_test", postproc_comp, promotes=["*"])
    _, translation_metadata, quantity_metadata = extract_time_integration_metadata(
        prob, ["x"]
    )
    postproc_prob.setup()

    with pytest.raises(AssertionError, match=error_message):
        add_postprocessing_metadata(
            postproc_prob, ["x"], quantity_list, quantity_metadata, translation_metadata
        )


@pytest.mark.mpi
def test_metadata_distributed_var_correct():
    """Tests whether addition of metadata for distributed variables is handled
    correctly."""
    input_dict = {
        "x_old": {
            "tags": ["step_input_var", "x"],
            "shape": 10 + MPI.COMM_WORLD.rank,
            "distributed": True,
        },
        "x_acc_stages": {
            "tags": ["accumulated_stage_var", "x"],
            "shape": 10 + MPI.COMM_WORLD.rank,
            "distributed": True,
        },
    }
    output_dict = {
        "x_update": {
            "tags": ["stage_output_var", "x"],
            "shape": 10 + MPI.COMM_WORLD.rank,
            "distributed": True,
        },
    }
    indep = om.IndepVarComp()
    indep.add_output("x_old", shape_by_conn=True, distributed=True)
    indep.add_output("x_acc_stages", shape_by_conn=True, distributed=True)
    test_comp = MetadataTestComponent(input_dict=input_dict, output_dict=output_dict)
    prob = om.Problem()
    prob.model.add_subsystem("indep", indep, promotes=["*"])
    prob.model.add_subsystem("test_comp", test_comp, promotes=["*"])
    prob.setup()

    array_size, translation_metadata, quantity_metadata = (
        extract_time_integration_metadata(prob, ["x"])
    )
    add_distributivity_information(prob, quantity_metadata)

    assert array_size == 10 + MPI.COMM_WORLD.rank
    assert translation_metadata == {
        "x": {
            "step_input_var": "test_comp.x_old",
            "accumulated_stage_var": "test_comp.x_acc_stages",
            "stage_output_var": "test_comp.x_update",
            "postproc_input_var": None,
        }
    }
    assert quantity_metadata == {
        "x": {
            "type": "time_integration",
            "shape": (10 + MPI.COMM_WORLD.rank,),
            "global_shape": (21,),
            "local": True,
            "distributed": True,
            "start_index": 0,
            "end_index": 10 + MPI.COMM_WORLD.rank,
            "functionally_integrated": False,
            "functional_start_index": 0,
            "functional_end_index": 0,
            "global_start_index": 0 if MPI.COMM_WORLD.rank == 0 else 10,
            "global_end_index": 10 if MPI.COMM_WORLD.rank == 0 else 21,
        }
    }


@pytest.mark.mpi
def test_metadata_parallel_group_correct():
    """Tests whether addition of metadata for variables coming from parallel groups is
    handled correctly."""
    input_dict_1 = {
        "x_old": {
            "tags": ["step_input_var", "x"],
            "shape": (2, 3),
            "distributed": False,
        },
        "x_acc_stages": {
            "tags": ["accumulated_stage_var", "x"],
            "shape": (2, 3),
            "distributed": False,
        },
    }
    output_dict_1 = {
        "x_update": {
            "tags": ["stage_output_var", "x"],
            "shape": (2, 3),
            "distributed": False,
        },
    }
    input_dict_2 = {
        "y_old": {
            "tags": ["step_input_var", "y"],
            "shape": (3, 2),
            "distributed": False,
        },
        "y_acc_stages": {
            "tags": ["accumulated_stage_var", "y"],
            "shape": (3, 2),
            "distributed": False,
        },
    }
    output_dict_2 = {
        "y_update": {
            "tags": ["stage_output_var", "y"],
            "shape": (3, 2),
            "distributed": False,
        },
    }
    par_group = om.ParallelGroup()
    par_group.add_subsystem(
        "test_comp_1",
        MetadataTestComponent(input_dict=input_dict_1, output_dict=output_dict_1),
        promotes=["*"],
    )
    par_group.add_subsystem(
        "test_comp_2",
        MetadataTestComponent(input_dict=input_dict_2, output_dict=output_dict_2),
        promotes=["*"],
    )
    prob = om.Problem()
    prob.model.add_subsystem("par_group", par_group, promotes=["*"])
    prob.setup()

    array_size, translation_metadata, quantity_metadata = (
        extract_time_integration_metadata(prob, ["x", "y"])
    )
    add_distributivity_information(prob, quantity_metadata)

    assert array_size == 6
    if prob.comm.rank == 0:
        assert translation_metadata == {
            "x": {
                "step_input_var": "par_group.test_comp_1.x_old",
                "accumulated_stage_var": "par_group.test_comp_1.x_acc_stages",
                "stage_output_var": "par_group.test_comp_1.x_update",
                "postproc_input_var": None,
            },
            "y": {
                "step_input_var": None,
                "accumulated_stage_var": None,
                "stage_output_var": None,
                "postproc_input_var": None,
            },
        }
        assert quantity_metadata == {
            "x": {
                "type": "time_integration",
                "shape": (2, 3),
                "global_shape": (2, 3),
                "local": True,
                "distributed": True,
                "start_index": 0,
                "end_index": 6,
                "functionally_integrated": False,
                "functional_start_index": 0,
                "functional_end_index": 0,
                "global_start_index": 0,
                "global_end_index": 6,
            },
            "y": {
                "type": "time_integration",
                "shape": 0,
                "global_shape": 0,
                "local": False,
                "distributed": True,
                "start_index": 0,
                "end_index": 0,
                "functionally_integrated": False,
                "functional_start_index": 0,
                "functional_end_index": 0,
                "global_start_index": 0,
                "global_end_index": 0,
            },
        }
    else:
        assert translation_metadata == {
            "x": {
                "step_input_var": None,
                "accumulated_stage_var": None,
                "stage_output_var": None,
                "postproc_input_var": None,
            },
            "y": {
                "step_input_var": "par_group.test_comp_2.y_old",
                "accumulated_stage_var": "par_group.test_comp_2.y_acc_stages",
                "stage_output_var": "par_group.test_comp_2.y_update",
                "postproc_input_var": None,
            },
        }
        assert quantity_metadata == {
            "x": {
                "type": "time_integration",
                "shape": 0,
                "global_shape": 0,
                "local": False,
                "distributed": True,
                "start_index": 0,
                "end_index": 0,
                "functionally_integrated": False,
                "functional_start_index": 0,
                "functional_end_index": 0,
                "global_start_index": 0,
                "global_end_index": 0,
            },
            "y": {
                "type": "time_integration",
                "shape": (3, 2),
                "global_shape": (3, 2),
                "local": True,
                "distributed": True,
                "start_index": 0,
                "end_index": 6,
                "functionally_integrated": False,
                "functional_start_index": 0,
                "functional_end_index": 0,
                "global_start_index": 0,
                "global_end_index": 6,
            },
        }
