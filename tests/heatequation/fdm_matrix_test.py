import numpy as np
import pytest

from runge_kutta_openmdao.heatequation import fdm_matrix, domain


domain_1 = domain.Domain([0, 1], [0, 1], 10, 10)


@pytest.mark.heatequ
@pytest.mark.heatequ_matrix
@pytest.mark.parametrize(
    "test_matrix, vector, result",
    [
        (fdm_matrix.FdmMatrix(domain_1, 1.0), np.ones(10 * 10), np.zeros(10 * 10)),
        (fdm_matrix.FdmMatrix(domain_1, 1.0), np.zeros(10 * 10), np.zeros(10 * 10)),
    ],
)
def test_matrix_vector_prod(test_matrix, vector, result):
    assert np.array_equal(test_matrix.mat_vec_prod(vector), result)


@pytest.mark.heatequ
@pytest.mark.heatequ_matrix
@pytest.mark.parametrize(
    " test_matrix, other_matrix, result",
    [
        (
            fdm_matrix.FdmMatrix(domain_1, 1.0),
            np.ones((10 * 10, 10 * 10)),
            np.zeros((10 * 10, 10 * 10)),
        ),
        (
            fdm_matrix.FdmMatrix(domain_1, 1.0),
            np.zeros((10 * 10, 10 * 10)),
            np.zeros((10 * 10, 10 * 10)),
        ),
    ],
)
def test_matrix_matrix_prod(test_matrix, other_matrix, result):
    assert np.array_equal(test_matrix.mat_mat_prod(other_matrix), result)
