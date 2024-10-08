"""Tests for the flux component that bridges two heat equations."""

import itertools
import pytest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from ..flux_component import FluxComponent


@pytest.mark.heatequ
@pytest.mark.heatequ_flux_comp
@pytest.mark.parametrize(
    "delta, vector_1, vector_2, orientation, expected",
    (
        [0.1, np.ones(11), np.ones(11), "vertical", np.zeros(11)],
        [0.05, np.zeros(21), np.ones(21), "horizontal", 10 * np.ones(21)],
        [0.02, np.ones(51), -np.ones(51), "vertical", -50 * np.ones(51)],
    ),
)
def test_flux_component_computation(delta, vector_1, vector_2, orientation, expected):
    """Tests the computation of the flux component."""
    assert vector_1.size == vector_2.size
    flux_comp = FluxComponent(delta=delta, shape=vector_1.size, orientation=orientation)
    flux_comp.setup()
    test_vec = {}
    if orientation == "vertical":
        test_vec["left_side"] = vector_1
        test_vec["right_side"] = vector_2
    else:
        test_vec["lower_side"] = vector_1
        test_vec["upper_side"] = vector_2

    test_result = {
        "flux": np.zeros_like(vector_1),
        "reverse_flux": np.zeros_like(vector_2),
    }

    test_residual = {
        "flux": np.zeros_like(vector_1),
        "reverse_flux": np.zeros_like(vector_2),
    }
    flux_comp.solve_nonlinear(test_vec, test_result)
    flux_comp.apply_nonlinear(test_vec, test_result, test_residual)
    assert test_result["flux"] == pytest.approx(expected)
    assert test_result["reverse_flux"] == pytest.approx(-expected)
    assert test_residual["flux"] == pytest.approx(np.zeros_like(vector_1))
    assert test_residual["reverse_flux"] == pytest.approx(np.zeros_like(vector_2))


@pytest.mark.heatequ
@pytest.mark.heatequ_flux_comp
@pytest.mark.parametrize(
    "delta, shape, orientation",
    itertools.product([1e-1, 2e-2, 5e-1], [11, 21, 51], ["vertical", "horizontal"]),
)
def test_flux_component_partials(delta, shape, orientation):
    """Tests the partials of the flux component."""
    test_prob = om.Problem()
    flux_comp = FluxComponent(delta=delta, shape=shape, orientation=orientation)
    test_prob.model.add_subsystem("flux_comp", flux_comp)
    test_prob.setup()
    test_prob.run_model()
    test_data = test_prob.check_partials(step=1e-1)
    assert_check_partials(test_data)


@pytest.mark.heatequ
@pytest.mark.heatequ_flux_comp
@pytest.mark.parametrize(
    "delta, vector_1, vector_2, orientation",
    itertools.product(
        [1e-1, 2e-2, 5e-1],
        [np.ones(51), np.zeros(51), np.linspace(0.0, 10.0, 51)],
        [np.ones(51), np.zeros(51), np.linspace(0.0, 10.0, 51)],
        ["vertical", "horizontal"],
    ),
)
def test_flux_component_compute_and_jacvec(delta, vector_1, vector_2, orientation):
    """Tests linearity of flux component by comparing apply_nonlinear and
    apply_linear."""
    assert vector_1.size == vector_2.size
    flux_comp = FluxComponent(delta=delta, shape=vector_1.size, orientation=orientation)
    flux_comp.setup()
    test_vec = {}
    test_d_inputs = {}
    if orientation == "vertical":
        test_vec["left_side"] = vector_1
        test_d_inputs["left_side"] = np.zeros_like(vector_1)
        test_vec["right_side"] = vector_2
        test_d_inputs["right_side"] = np.zeros_like(vector_2)
    else:
        test_vec["lower_side"] = vector_1
        test_d_inputs["lower_side"] = np.zeros_like(vector_1)
        test_vec["upper_side"] = vector_2
        test_d_inputs["upper_side"] = np.zeros_like(vector_2)

    test_result = {
        "flux": np.zeros_like(vector_1),
        "reverse_flux": np.zeros_like(vector_2),
    }
    test_residual = {
        "flux": np.zeros_like(vector_1),
        "reverse_flux": np.zeros_like(vector_2),
    }
    test_d_residual = {
        "flux": np.zeros_like(vector_1),
        "reverse_flux": np.zeros_like(vector_2),
    }
    flux_comp.solve_nonlinear(test_vec, test_result)
    flux_comp.apply_nonlinear(test_vec, test_result, test_residual)
    flux_comp.apply_linear(None, None, test_vec, test_result, test_d_residual, "fwd")
    assert test_residual["flux"] == pytest.approx(test_d_residual["flux"])
    assert test_residual["reverse_flux"] == pytest.approx(
        test_d_residual["reverse_flux"]
    )
