import pytest
import numpy as np

from heatequation.src import domain, boundary, inhomogenity


test_domain = domain.Domain([0.0, 1.0], [0.0, 1.0], 11, 11)


@pytest.mark.heatequ
@pytest.mark.heatequ_inhom
@pytest.mark.parametrize(
    "test_inhomogenity, time, pde_inhom",
    [
        (
            inhomogenity.InhomogenityVector(
                test_domain, 1.0, lambda t, x, y: 2.0, boundary.BoundaryCondition()
            ),
            1.0,
            np.full(121, 2.0),
        ),
        (
            inhomogenity.InhomogenityVector(
                test_domain,
                1.0,
                lambda t, x, y: 2.0,
                boundary.BoundaryCondition(upper=lambda t, x, y: 3.0),
            ),
            1.0,
            np.full(121, 2.0),
        ),
    ],
)
def test_inhom_update_pde(test_inhomogenity, time, pde_inhom):
    test_inhomogenity.update_pde_inhomogenity(time)
    assert np.array_equal(test_inhomogenity.pde_inhomogenity, pde_inhom)


@pytest.mark.heatequ
@pytest.mark.heatequ_inhom
@pytest.mark.parametrize(
    "test_inhomogenity, time, upper, lower, left, right, bound_inhom",
    [
        (
            inhomogenity.InhomogenityVector(
                test_domain, 1.0, lambda t, x, y: 2.0, boundary.BoundaryCondition()
            ),
            1.0,
            None,
            None,
            None,
            None,
            np.full(121, 0.0),
        ),
        (
            inhomogenity.InhomogenityVector(
                test_domain,
                1.0,
                lambda t, x, y: 2.0,
                boundary.BoundaryCondition(upper=lambda t, x, y: 3.0),
            ),
            1.0,
            None,
            None,
            None,
            None,
            np.kron(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1]),
                np.full(11, 2 * 3.0 / test_domain.delta_y),
            ),
        ),
        (
            inhomogenity.InhomogenityVector(
                test_domain,
                1.0,
                lambda t, x, y: 2.0,
                boundary.BoundaryCondition(),
            ),
            1.0,
            None,
            None,
            np.full(11, 5.0),
            None,
            np.kron(
                np.full(11, 2 * 5.0 / test_domain.delta_x),
                np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            ),
        ),
    ],
)
def test_inhom_update_boundary(
    test_inhomogenity, time, upper, lower, left, right, bound_inhom
):
    test_inhomogenity.update_boundary_inhomogenity(time, upper, lower, left, right)
    assert np.array_equal(
        bound_inhom,
        test_inhomogenity.self_updating_boundary_inhomogenity
        + np.sum(test_inhomogenity.user_updating_boundary_inhomogenity, axis=1),
    )
