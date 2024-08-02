"""Tests for the domain class for the heat equation."""

import numpy as np
import pytest

from .. import domain


@pytest.mark.heatequ
@pytest.mark.heatequ_domain
@pytest.mark.parametrize(
    "x_range, y_range, n_x, n_y, delta_x, delta_y",
    [
        ([0.0, 1.0], [0.0, 1.0], 5, 5, 0.25, 0.25),
        ([0.0, 1.0], [0.0, 1.0], 1, 1, 0, 0),
        ([0.0, 1.0], [0.0, 1.0], 5, 1, 0.25, 0),
    ],
)
def test_domain_init(x_range, y_range, n_x, n_y, delta_x, delta_y):
    """Tests of the initializer."""
    if n_x == 1 or n_y == 1:
        with pytest.raises(ZeroDivisionError, match=r"division by zero"):
            domain.Domain(x_range, y_range, n_x, n_y)
    else:
        test_domain = domain.Domain(x_range, y_range, n_x, n_y)
        assert test_domain.delta_x == pytest.approx(delta_x)
        assert test_domain.delta_y == pytest.approx(delta_y)


@pytest.mark.heatequ
@pytest.mark.heatequ_domain
@pytest.mark.parametrize(
    "test_domain, left_coord, right_coord, upper_coord, lower_coord",
    [
        (
            domain.Domain([0, 1], [0, 1], 2, 2),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        ),
        (
            domain.Domain([0, 5], [0, 5], 2, 2),
            np.array([[0.0, 0.0], [0.0, 5.0]]),
            np.array([[5.0, 0.0], [5.0, 5.0]]),
            np.array([[0.0, 5.0], [5.0, 5.0]]),
            np.array([[0.0, 0.0], [5.0, 0.0]]),
        ),
        (
            domain.Domain([0, 1], [0, 1], 4, 4),
            np.array([[0.0, 0.0], [0.0, 1.0 / 3.0], [0.0, 2.0 / 3.0], [0.0, 1.0]]),
            np.array([[1.0, 0.0], [1.0, 1.0 / 3.0], [1.0, 2.0 / 3.0], [1.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0 / 3.0, 1.0], [2.0 / 3.0, 1.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [1.0 / 3.0, 0.0], [2.0 / 3.0, 0.0], [1.0, 0.0]]),
        ),
    ],
)
def test_domain_boundary_coordinates(
    test_domain: domain.Domain, left_coord, right_coord, upper_coord, lower_coord
):
    """Tests correctness of coordinates for the boundary of domain."""
    generated_left = test_domain.boundary_coordinates("left")
    generated_right = test_domain.boundary_coordinates("right")
    generated_upper = test_domain.boundary_coordinates("upper")
    generated_lower = test_domain.boundary_coordinates("lower")

    assert np.array_equal(generated_left, left_coord)
    assert np.array_equal(generated_right, right_coord)
    assert np.array_equal(generated_upper, upper_coord)
    assert np.array_equal(generated_lower, lower_coord)


@pytest.mark.heatequ
@pytest.mark.heatequ_domain
@pytest.mark.parametrize(
    "test_domain, left_ind, right_ind, upper_ind, lower_ind",
    [
        (
            domain.Domain([0, 1], [0, 1], 2, 2),
            np.array([0, 2]),
            np.array([1, 3]),
            np.array([2, 3]),
            np.array([0, 1]),
        ),
        (
            domain.Domain([0, 1], [0, 1], 4, 4),
            np.array([0, 4, 8, 12]),
            np.array([3, 7, 11, 15]),
            np.array([12, 13, 14, 15]),
            np.array([0, 1, 2, 3]),
        ),
        (
            domain.Domain([0, 4], [0, 4], 10, 10),
            np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
            np.array([9, 19, 29, 39, 49, 59, 69, 79, 89, 99]),
            np.array([90, 91, 92, 93, 94, 95, 96, 97, 98, 99]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ),
    ],
)
def test_domain_boundary_indices(
    test_domain: domain.Domain, left_ind, right_ind, upper_ind, lower_ind
):
    """Tests correctness of boundary indices."""
    generated_left = test_domain.boundary_indices("left")
    generated_right = test_domain.boundary_indices("right")
    generated_upper = test_domain.boundary_indices("upper")
    generated_lower = test_domain.boundary_indices("lower")

    assert np.array_equal(generated_left, left_ind)
    assert np.array_equal(generated_right, right_ind)
    assert np.array_equal(generated_upper, upper_ind)
    assert np.array_equal(generated_lower, lower_ind)


@pytest.mark.heatequ
@pytest.mark.heatequ_domain
@pytest.mark.parametrize(
    "test_domain, i, j, coord",
    [
        (domain.Domain([0, 1], [0, 1], 2, 2), 0, 0, np.array([0.0, 0.0])),
        (domain.Domain([3, 53], [2, 22], 51, 21), 10, 10, np.array([13.0, 12.0])),
        (domain.Domain([-4, 4], [-5, 5], 21, 21), 10, 10, np.array([0.0, 0.0])),
    ],
)
def test_ij_coordinates(test_domain, i, j, coord):
    """Tests mapping between x-y-indices and coordinates."""
    generated_coord = test_domain.ij_coordinates(i, j)

    assert np.array_equal(generated_coord, coord)


@pytest.mark.heatequ
@pytest.mark.heatequ_domain
@pytest.mark.parametrize(
    "test_domain, i, j, index",
    [
        (domain.Domain([0, 1], [0, 1], 2, 2), 0, 0, 0),
        (domain.Domain([3, 53], [2, 22], 51, 21), 10, 10, 520),
        (domain.Domain([-4, 4], [-5, 5], 21, 21), 10, 10, 220),
    ],
)
def test_ij_to_index(test_domain, i, j, index):
    """Tests mapping between x-y-indices and their serialization."""
    generated_index = test_domain.ij_to_index(i, j)

    assert index == generated_index


@pytest.mark.heatequ
@pytest.mark.heatequ_domain
@pytest.mark.parametrize(
    "test_domain, index, indices",
    [
        (domain.Domain([0, 1], [0, 1], 2, 2), 0, np.array([0, 0])),
        (domain.Domain([3, 53], [2, 22], 51, 21), 520, np.array([10, 10])),
        (domain.Domain([-4, 4], [-5, 5], 21, 21), 220, np.array([10, 10])),
    ],
)
def test_index_to_ij(test_domain, index, indices):
    """Tests reverse of mapping between x-y-indices and their serialization."""
    generated_indices = test_domain.index_to_ij(index)

    assert np.array_equal(generated_indices, indices)
