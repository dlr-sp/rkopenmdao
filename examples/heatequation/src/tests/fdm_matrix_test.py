import numpy as np
import pytest

from heatequation.src import fdm_matrix, domain


domain_1 = domain.Domain([0, 1], [0, 1], 10, 10)
vec = np.full(10, 2.0)
vec[0] = vec[9] = 1
vec = np.kron(vec, vec)

mat = np.empty((10 * 10, 10 * 10))
for i in range(100):
    mat[:, i] = vec


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
    "test_matrix, vector, result",
    [
        (fdm_matrix.FdmMatrix(domain_1, 1.0), vec, np.zeros(10 * 10)),
        (fdm_matrix.FdmMatrix(domain_1, 1.0), np.zeros(10 * 10), np.zeros(10 * 10)),
    ],
)
def test_transpose_matrix_vector_prod(test_matrix, vector, result):
    assert np.array_equal(test_matrix.mat_vec_prod_transpose(vector), result)


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


@pytest.mark.heatequ
@pytest.mark.heatequ_matrix
@pytest.mark.parametrize(
    " test_matrix, other_matrix, result",
    [
        (
            fdm_matrix.FdmMatrix(domain_1, 1.0),
            mat,
            np.zeros((10 * 10, 10 * 10)),
        ),
        (
            fdm_matrix.FdmMatrix(domain_1, 1.0),
            np.zeros((10 * 10, 10 * 10)),
            np.zeros((10 * 10, 10 * 10)),
        ),
    ],
)
def test_transpose_matrix_matrix_prod(test_matrix, other_matrix, result):
    assert np.array_equal(test_matrix.mat_vec_prod_transpose(other_matrix), result)
