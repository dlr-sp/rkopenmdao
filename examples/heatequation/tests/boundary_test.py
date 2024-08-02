"""Tests the functionality of the boundary for the heat equation."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


from .. import boundary


@pytest.mark.heatequ
@pytest.mark.heatequ_boundary
@pytest.mark.parametrize(
    "test_boundary, kind_dict",
    [
        (
            boundary.BoundaryCondition(),
            {
                "left": "user_updating",
                "right": "user_updating",
                "upper": "user_updating",
                "lower": "user_updating",
            },
        ),
        (
            boundary.BoundaryCondition(upper=lambda x, y, t: 1),
            {
                "left": "user_updating",
                "right": "user_updating",
                "upper": "self_updating",
                "lower": "user_updating",
            },
        ),
        (
            boundary.BoundaryCondition(
                upper=lambda x, y, t: 1,
                lower=lambda x, y, t: 1,
                left=lambda x, y, t: 1,
                right=lambda x, y, t: 1,
            ),
            {
                "left": "self_updating",
                "right": "self_updating",
                "upper": "self_updating",
                "lower": "self_updating",
            },
        ),
        (
            boundary.BoundaryCondition(upper=1),
            {
                "left": "user_updating",
                "right": "user_updating",
                "upper": "self_updating",
                "lower": "user_updating",
            },
        ),
    ],
)
def test_boundary_update_kind(test_boundary, kind_dict):
    """Tests whether update_kind_dict of BoundaryCondition gets created as expected from
    its constructor arguments."""
    assert test_boundary.boundary_update_kind() == kind_dict


# actually comparing two (lambda) functions is probably very difficult,
# so it just is tested whether its None when it should be or not


@pytest.mark.heatequ
@pytest.mark.heatequ_boundary
@pytest.mark.parametrize(
    "test_boundary, left, right, upper, lower",
    [
        (boundary.BoundaryCondition(), None, None, None, None),
        (boundary.BoundaryCondition(upper=lambda x, y, t: 1), None, None, 1, None),
        (
            boundary.BoundaryCondition(
                upper=lambda x, y, t: 1,
                lower=lambda x, y, t: 1,
                left=lambda x, y, t: 1,
                right=lambda x, y, t: 1,
            ),
            1,
            1,
            1,
            1,
        ),
        (boundary.BoundaryCondition(upper=1), None, None, 1, None),
    ],
)
def test_boundary_get_function(test_boundary, left, right, upper, lower):
    """Tests whether a function is returned only when it should be."""
    assert (test_boundary.get_function("left") is None and left is None) or (
        test_boundary.get_function("left") is not None and left is not None
    )
    assert (test_boundary.get_function("right") is None and right is None) or (
        test_boundary.get_function("right") is not None and right is not None
    )
    assert (test_boundary.get_function("upper") is None and upper is None) or (
        test_boundary.get_function("upper") is not None and upper is not None
    )
    assert (test_boundary.get_function("lower") is None and lower is None) or (
        test_boundary.get_function("lower") is not None and lower is not None
    )


@pytest.mark.heatequ
@pytest.mark.heatequ_boundary
@pytest.mark.parametrize(
    """test_boundary, time_1, coordinate_1, result_1, expect_upper, expect_lower,
    expect_left, expect_right""",
    [
        (
            boundary.BoundaryCondition(),
            0.0,
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([1.0, 1.0]),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
        ),
        (
            boundary.BoundaryCondition(upper=lambda x, y, t: 1),
            0.0,
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([1.0, 1.0]),
            does_not_raise(),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
        ),
        (
            boundary.BoundaryCondition(
                upper=lambda x, y, t: 1,
                lower=lambda x, y, t: 1,
                left=lambda x, y, t: 1,
                right=lambda x, y, t: 1,
            ),
            0.0,
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([1.0, 1.0]),
            does_not_raise(),
            does_not_raise(),
            does_not_raise(),
            does_not_raise(),
        ),
        (
            boundary.BoundaryCondition(upper=1),
            0.0,
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([1.0, 1.0]),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
            pytest.raises(TypeError),
        ),
    ],
)
def test_boundary_self_update(
    test_boundary,
    time_1,
    coordinate_1,
    result_1,
    expect_upper,
    expect_lower,
    expect_left,
    expect_right,
):
    """Tests updater of BoundaryCondition to work correctly."""
    with expect_upper:
        assert np.array_equal(
            test_boundary.self_update("upper", time_1, coordinate_1), result_1
        )
    with expect_lower:
        assert np.array_equal(
            test_boundary.self_update("lower", time_1, coordinate_1), result_1
        )
    with expect_left:
        assert np.array_equal(
            test_boundary.self_update("left", time_1, coordinate_1), result_1
        )
    with expect_right:
        assert np.array_equal(
            test_boundary.self_update("right", time_1, coordinate_1), result_1
        )
