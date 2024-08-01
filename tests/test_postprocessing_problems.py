"""Contains several problems used to test the postprocessing capabilities of the
RungeKuttaIntegrator."""

import numpy as np
import openmdao.api as om


# pylint: disable=arguments-differ


class NegatingComponent(om.ExplicitComponent):
    """Component that simply negates its inputs."""

    def initialize(self):
        self.options.declare(
            "quantity_list",
            default=[],
            types=list,
            desc="list of tuples of quantities and their shape.",
        )

    def setup(self):
        for quantity, shape in self.options["quantity_list"]:
            self.add_input(quantity, shape=shape, tags=[quantity, "postproc_input_var"])
            self.add_output(
                "negated_" + quantity,
                shape=shape,
                tags=["negated_" + quantity, "postproc_output_var"],
            )

    def compute(self, inputs, outputs):
        for quantity, _ in self.options["quantity_list"]:
            outputs["negated_" + quantity] = -inputs[quantity]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for quantity, _ in self.options["quantity_list"]:
                d_outputs["negated_" + quantity] -= d_inputs[quantity]
        elif mode == "rev":
            for quantity, _ in self.options["quantity_list"]:
                d_inputs[quantity] -= d_outputs["negated_" + quantity]


def create_negating_problem(quantity_list):
    """Creates a problem containing only a NegatingComponent"""
    negating_problem = om.Problem()
    negating_problem.model.add_subsystem(
        "negate", NegatingComponent(quantity_list=quantity_list)
    )
    return negating_problem


def negating_function(array):
    """Function that negates its inputs (as analytical solution to the above)"""
    return -array


class AccumulatingComponent(om.ExplicitComponent):
    """Component that accumulates its input."""

    def initialize(self):
        self.options.declare(
            "quantity_list",
            default=[],
            types=list,
            desc="list of tuples of quantities and their shape.",
        )

    def setup(self):
        for quantity, shape in self.options["quantity_list"]:
            self.add_input(quantity, shape=shape, tags=[quantity, "postproc_input_var"])

        self.add_output(
            "accumulated", shape=1, tags=["accumulated", "postproc_output_var"]
        )

    def compute(self, inputs, outputs):
        outputs["accumulated"] = 0
        for quantity, _ in self.options["quantity_list"]:
            outputs["accumulated"] += np.sum(inputs[quantity])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for quantity, _ in self.options["quantity_list"]:
                d_outputs["accumulated"] += np.sum(d_inputs[quantity])
        elif mode == "rev":
            for quantity, _ in self.options["quantity_list"]:
                d_inputs[quantity] += d_outputs["accumulated"]


def create_accumulating_problem(quantity_list):
    """Creates a problem containing only a AccumulatingComponent"""
    accumulating_problem = om.Problem()
    accumulating_problem.model.add_subsystem(
        "accumulate", AccumulatingComponent(quantity_list=quantity_list)
    )
    return accumulating_problem


def accumulating_function(array):
    """Function that accumulates its inputs (as analytical solution to the above)"""
    return np.sum(array)


class SquaringComponent(om.ExplicitComponent):
    """Component that squares its inputs."""

    def initialize(self):
        self.options.declare(
            "quantity_list",
            default=[],
            types=list,
            desc="list of tuples of quantities and their shape.",
        )

    def setup(self):
        for quantity, shape in self.options["quantity_list"]:
            self.add_input(quantity, shape=shape, tags=[quantity, "postproc_input_var"])
            self.add_output(
                "squared_" + quantity,
                shape=shape,
                tags=["squared_" + quantity, "postproc_output_var"],
            )

    def compute(self, inputs, outputs):
        for quantity, _ in self.options["quantity_list"]:
            outputs["squared_" + quantity] = inputs[quantity] ** 2

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for quantity, _ in self.options["quantity_list"]:
                d_outputs["squared_" + quantity] += (
                    2 * inputs[quantity] * d_inputs[quantity]
                )
        elif mode == "rev":
            for quantity, _ in self.options["quantity_list"]:
                d_inputs[quantity] += (
                    2 * inputs[quantity] * d_outputs["squared_" + quantity]
                )


def create_squaring_problem(quantity_list):
    """Creates a problem containing only a SquaringComponent"""
    squaring_problem = om.Problem()
    squaring_problem.model.add_subsystem(
        "square", SquaringComponent(quantity_list=quantity_list)
    )
    return squaring_problem


def squaring_function(array):
    """Function that squares its inputs (as analytical solution to the above)"""
    return array * array


class PhaseComponent(om.ExplicitComponent):
    """Component that applies a pseudo-phasefield out of its inputs."""

    def initialize(self):
        self.options.declare(
            "quantity_list",
            default=[],
            types=list,
            desc="list of tuples of quantities and their shape.",
        )

    def setup(self):
        for quantity, shape in self.options["quantity_list"]:
            self.add_input(quantity, shape=shape, tags=[quantity, "postproc_input_var"])
            self.add_output(
                "phase_" + quantity,
                shape=shape,
                tags=["phase_" + quantity, "postproc_output_var"],
            )

    def compute(self, inputs, outputs):
        for quantity, _ in self.options["quantity_list"]:
            outputs["phase_" + quantity] = np.tanh(inputs[quantity])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for quantity, _ in self.options["quantity_list"]:
                d_outputs["phase_" + quantity] += (
                    1 - np.tanh(inputs[quantity]) ** 2
                ) * d_inputs[quantity]
        elif mode == "rev":
            for quantity, _ in self.options["quantity_list"]:
                d_inputs[quantity] += (1 - np.tanh(inputs[quantity]) ** 2) * d_outputs[
                    "phase_" + quantity
                ]


def create_phase_problem(quantity_list):
    """Creates a problem containing only a PhaseComponent"""
    phase_problem = om.Problem()
    phase_problem.model.add_subsystem(
        "phase", PhaseComponent(quantity_list=quantity_list)
    )
    return phase_problem


def phase_function(array):
    """Function that applies a pseudo-phasefield its inputs (as analytical solution to
    the above)."""
    return np.tanh(array)
