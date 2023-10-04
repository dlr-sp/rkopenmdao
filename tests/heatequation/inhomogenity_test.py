import pytest
import numpy as np

from runge_kutta_openmdao.heatequation import domain, boundary, inhomogeneity


test_domain = domain.Domain([0.0, 1.0], [0.0, 1.0], 11, 11)


@pytest.mark.heatequ
@pytest.mark.heatequ_inhom
@pytest.mark.parametrize(
    "test_inhomogeneity, time, pde_inhom",
    [
        (
            inhomogeneity.InhomogeneityVector(
                test_domain, 1.0, lambda t, x, y: 2.0, boundary.BoundaryCondition()
            ),
            1.0,
            np.full(121, 2.0),
        ),
        (
            inhomogeneity.InhomogeneityVector(
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
def test_inhom_update_pde(test_inhomogeneity, time, pde_inhom):
    test_inhomogeneity.update_pde_inhomogeneity(time)
    assert np.array_equal(test_inhomogeneity.pde_inhomogeneity, pde_inhom)


@pytest.mark.heatequ
@pytest.mark.heatequ_inhom
@pytest.mark.parametrize(
    "test_inhomogeneity, time, upper, lower, left, right, bound_inhom",
    [
        (
            inhomogeneity.InhomogeneityVector(
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
            inhomogeneity.InhomogeneityVector(
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
            inhomogeneity.InhomogeneityVector(
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
    test_inhomogeneity, time, upper, lower, left, right, bound_inhom
):
    test_inhomogeneity.update_boundary_inhomogeneity(time, upper, lower, left, right)
    assert np.array_equal(
        bound_inhom,
        test_inhomogeneity.self_updating_boundary_inhomogeneity
        + np.sum(test_inhomogeneity.user_updating_boundary_inhomogeneity, axis=1),
    )
