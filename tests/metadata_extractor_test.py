"""Tests for extraction of metadata from OpenMDAO problems."""

# pylint: disable=c-extension-no-member

from mpi4py import MPI
import openmdao.api as om
import pytest

from rkopenmdao.metadata_extractor import (
    extract_time_integration_metadata,
    add_time_independent_input_metadata,
    add_postprocessing_metadata,
    add_functional_metadata,
    add_distributivity_information,
    TimeIntegrationQuantity,
    PostprocessingQuantity,
    TimeIndependentQuantity,
    ArrayMetadata,
    TimeIntegrationTranslationMetadata,
    PostprocessingTranslationMetadata,
    TimeIndependentInputTranslationMetadata,
)

from rkopenmdao.errors import SetupError


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
    return prob


def test_metadata_non_parallel_correct():
    """Tests for the sequential case whether time integration metadata is correctly
    set."""
    prob = basic_test_problem()
    prob.setup()

    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])

    assert time_integration_metadata.time_integration_array_size == 10
    assert time_integration_metadata.time_integration_quantity_list == [
        TimeIntegrationQuantity(
            "x",
            "time_integration",
            ArrayMetadata(
                shape=(5, 2),
                global_shape=(5, 2),
                local=True,
                start_index=0,
                end_index=10,
                global_start_index=0,
                global_end_index=10,
            ),
            TimeIntegrationTranslationMetadata(
                step_input_var="test_comp.x_old",
                accumulated_stage_var="test_comp.x_acc_stages",
                stage_output_var="test_comp.x_update",
            ),
        )
    ]


@pytest.mark.parametrize(
    "input_dict, output_dict, time_integration_quantity_list, error_message",
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
    input_dict, output_dict, time_integration_quantity_list, error_message
):
    """Tests various incorrect cases for the setup of the time integration metadata."""
    test_comp = MetadataTestComponent(input_dict=input_dict, output_dict=output_dict)
    prob = om.Problem()
    prob.model.add_subsystem("test_comp", test_comp, promotes=["*"])
    prob.setup()
    with pytest.raises(SetupError, match=error_message):
        extract_time_integration_metadata(prob, time_integration_quantity_list)


def test_metadata_time_independent_inputs_correct():
    """Tests whether independent input metadata is correctly added."""
    prob = basic_test_problem()
    prob.model.add_subsystem(
        "input_comp",
        MetadataTestComponent(
            input_dict={
                "w": {
                    "tags": ["time_independent_input_var", "w"],
                    "shape": (2, 2),
                    "distributed": False,
                },
            },
            output_dict={},
        ),
    )
    prob.setup()
    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    add_time_independent_input_metadata(prob, ["w"], time_integration_metadata)
    assert time_integration_metadata.time_independent_input_size == 4
    assert time_integration_metadata.time_independent_input_quantity_list == [
        TimeIndependentQuantity(
            "w",
            "independent_input",
            ArrayMetadata(
                shape=(2, 2),
                global_shape=(2, 2),
                local=True,
                start_index=0,
                end_index=4,
                global_start_index=0,
                global_end_index=4,
            ),
            TimeIndependentInputTranslationMetadata(
                time_independent_input_var="input_comp.w",
            ),
        ),
    ]


@pytest.mark.parametrize(
    "input_dict, time_independent_input_quantity_list, error_message",
    [
        (
            {
                "w": {
                    "tags": ["time_independent_input_var", "v", "w"],
                    "shape": (2, 2),
                    "distributed": False,
                },
            },
            ["v", "w"],
            "Variable input_comp.w either has two time independent quantity tags, "
            "or 'time_independent_input_var' was used as quantity tag. Both are "
            "forbidden. Tags of input_comp.w intersected with time independent input "
            "quantities: ({'v', 'w'}|{'w', 'v'})",
        ),
        (
            {
                "w": {
                    "tags": ["time_independent_input_var", "w"],
                    "shape": (2, 2),
                    "distributed": False,
                },
            },
            ["time_independent_input_var", "w"],
            "Variable input_comp.w either has two time independent quantity tags, "
            "or 'time_independent_input_var' was used as quantity tag. Both are "
            "forbidden. Tags of input_comp.w intersected with time independent input "
            "quantities: "
            "({'w', 'time_independent_input_var'}|{'time_independent_input_var', 'w'})",
        ),
        (
            {
                "w": {
                    "tags": ["time_independent_input_var", "w"],
                    "shape": (2, 2),
                    "distributed": False,
                },
                "ww": {
                    "tags": ["time_independent_input_var", "w"],
                    "shape": (2, 2),
                    "distributed": False,
                },
            },
            ["w"],
            "For quantity w, there is more than one inner variable tagged"
            " with 'time_independent_input_var'.",
        ),
        (
            {
                "w": {
                    "tags": ["time_independent_input_var_err", "w"],
                    "shape": (2, 2),
                    "distributed": False,
                },
            },
            ["w"],
            "For quantity w, there is no inner variable tagged with "
            "'time_independent_input_var'.",
        ),
    ],
)
def test_metadata_time_independent_inputs_incorrect(
    input_dict, time_independent_input_quantity_list, error_message
):
    """Tests various incorrect cases for the addition of metadata for independent
    inputs."""
    prob = basic_test_problem()
    prob.model.add_subsystem(
        "input_comp",
        MetadataTestComponent(
            input_dict=input_dict,
            output_dict={},
        ),
    )
    prob.setup()
    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    with pytest.raises(SetupError, match=error_message):
        add_time_independent_input_metadata(
            prob, time_independent_input_quantity_list, time_integration_metadata
        )


def test_metadata_functional_correct():
    """Tests whether functional metadata is correctly added to the metadata dict."""
    prob = basic_test_problem()
    prob.setup()
    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    add_functional_metadata(["x"], time_integration_metadata)
    assert time_integration_metadata.functional_array_size == 10
    assert time_integration_metadata.time_integration_quantity_list == [
        TimeIntegrationQuantity(
            "x",
            "time_integration",
            ArrayMetadata(
                shape=(5, 2),
                global_shape=(5, 2),
                local=True,
                start_index=0,
                end_index=10,
                functionally_integrated=True,
                functional_start_index=0,
                functional_end_index=10,
                global_start_index=0,
                global_end_index=10,
            ),
            TimeIntegrationTranslationMetadata(
                step_input_var="test_comp.x_old",
                accumulated_stage_var="test_comp.x_acc_stages",
                stage_output_var="test_comp.x_update",
            ),
        )
    ]


def test_metadata_functional_incorrect():
    """Tests an incorrect case for the addition of functional metadata."""
    prob = basic_test_problem()
    prob.setup()
    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    with pytest.raises(
        SetupError,
        match="Some functional requires a quantity that is part of neither the time "
        "integration nor postprocessing.",
    ):
        add_functional_metadata(["y"], time_integration_metadata)


def test_metadata_postprocessing_correct():
    """Tests the correct case for addition of postprocessing metadata to the metadata
    dicts."""
    prob = basic_test_problem()
    prob.setup()
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
    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    postproc_prob.setup()

    add_postprocessing_metadata(postproc_prob, ["x"], ["y"], time_integration_metadata)

    assert time_integration_metadata.postprocessing_array_size == 2
    assert time_integration_metadata.postprocessing_quantity_list == [
        PostprocessingQuantity(
            "y",
            "postprocessing",
            ArrayMetadata(
                shape=(1, 2),
                global_shape=(1, 2),
                local=True,
                start_index=0,
                end_index=2,
                global_start_index=0,
                global_end_index=2,
            ),
            PostprocessingTranslationMetadata(postproc_output_var="postproc_test.y"),
        ),
    ]


@pytest.mark.parametrize(
    "input_dict, output_dict, postprocessing_quantity_list, error_message",
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
    input_dict, output_dict, postprocessing_quantity_list, error_message
):
    """Tests various incorrect cases for the addition of postprocessing metadata to the
    dicts."""
    prob = basic_test_problem()
    prob.setup()
    postproc_comp = MetadataTestComponent(
        input_dict=input_dict, output_dict=output_dict
    )
    postproc_prob = om.Problem()
    postproc_prob.model.add_subsystem("postproc_test", postproc_comp, promotes=["*"])
    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    postproc_prob.setup()

    with pytest.raises(SetupError, match=error_message):
        add_postprocessing_metadata(
            postproc_prob,
            ["x"],
            postprocessing_quantity_list,
            time_integration_metadata,
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

    time_integration_metadata = extract_time_integration_metadata(prob, ["x"])
    add_distributivity_information(prob, time_integration_metadata)

    assert (
        time_integration_metadata.time_integration_array_size
        == 10 + MPI.COMM_WORLD.rank
    )
    assert time_integration_metadata.time_integration_quantity_list == [
        TimeIntegrationQuantity(
            "x",
            "time_integration",
            ArrayMetadata(
                shape=(10 + MPI.COMM_WORLD.rank,),
                global_shape=(21,),
                local=True,
                distributed=True,
                start_index=0,
                end_index=10 + MPI.COMM_WORLD.rank,
                global_start_index=0 if MPI.COMM_WORLD.rank == 0 else 10,
                global_end_index=10 if MPI.COMM_WORLD.rank == 0 else 21,
            ),
            TimeIntegrationTranslationMetadata(
                step_input_var="test_comp.x_old",
                accumulated_stage_var="test_comp.x_acc_stages",
                stage_output_var="test_comp.x_update",
            ),
        )
    ]


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

    time_integration_metadata = extract_time_integration_metadata(prob, ["x", "y"])
    add_distributivity_information(prob, time_integration_metadata)

    assert time_integration_metadata.time_integration_array_size == 6
    if prob.comm.rank == 0:
        assert time_integration_metadata.time_integration_quantity_list == [
            TimeIntegrationQuantity(
                "x",
                "time_integration",
                ArrayMetadata(
                    shape=(2, 3),
                    global_shape=(2, 3),
                    local=True,
                    distributed=True,
                    start_index=0,
                    end_index=6,
                    global_start_index=0,
                    global_end_index=6,
                ),
                TimeIntegrationTranslationMetadata(
                    step_input_var="par_group.test_comp_1.x_old",
                    accumulated_stage_var="par_group.test_comp_1.x_acc_stages",
                    stage_output_var="par_group.test_comp_1.x_update",
                ),
            ),
            TimeIntegrationQuantity(
                "y",
                "time_integration",
                ArrayMetadata(distributed=True),
                TimeIntegrationTranslationMetadata(),
            ),
        ]
    else:
        assert time_integration_metadata.time_integration_quantity_list == [
            TimeIntegrationQuantity(
                "x",
                "time_integration",
                ArrayMetadata(distributed=True),
                TimeIntegrationTranslationMetadata(),
            ),
            TimeIntegrationQuantity(
                "y",
                "time_integration",
                ArrayMetadata(
                    shape=(3, 2),
                    global_shape=(3, 2),
                    local=True,
                    distributed=True,
                    start_index=0,
                    end_index=6,
                    global_start_index=0,
                    global_end_index=6,
                ),
                TimeIntegrationTranslationMetadata(
                    step_input_var="par_group.test_comp_2.y_old",
                    accumulated_stage_var="par_group.test_comp_2.y_acc_stages",
                    stage_output_var="par_group.test_comp_2.y_update",
                ),
            ),
        ]
