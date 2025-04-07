import numpy as np

rng = np.random.default_rng(42)  # Seeded for reproducibility


def test_symmetric_real_eigenvalues():
    """Symmetric matrices have only real eigenvalues"""
    A = rng.random((4, 4))
    A = (A + A.T) / 2  # Make symmetric
    assert np.all(np.isreal(np.linalg.eigvals(A)))


def test_symmetric_eigenvectors_orthogonal():
    """Eigenvectors of symmetric matrix are orthogonal"""
    A = rng.random((3, 3))
    A = (A + A.T) / 2
    _, v = np.linalg.eig(A)
    dot = v[:, 0].T @ v[:, 1]
    assert np.abs(dot) < 1e-10


def test_approx_simple_spectrum():
    """Symmetric matrices can be approximated by matrices with simple spectrum"""
    A = np.full((3, 3), 2.0)
    noise = rng.normal(scale=1e-5, size=(3, 3))
    noise = (noise + noise.T) / 2  # make it symmetric
    A_t = A + noise
    eigs = np.linalg.eigvals(A_t)
    assert len(np.unique(np.round(eigs, 5))) == 3


def test_symmetric_diagonalizable_orthonormal():
    """Symmetric matrix has orthonormal eigenbasis"""
    A = rng.random((5, 5))
    A = (A + A.T) / 2
    _, v = np.linalg.eig(A)
    assert np.allclose(v.T @ v, np.eye(5))


def test_wiggle_fail_non_symmetric():
    """Wiggling fails for non-symmetric matrix"""
    A = np.array([[0, 1], [0, 0]])
    A_t = A + 1e-5 * np.outer([1, 0], [1, 0])
    _, v = np.linalg.eig(A_t)
    assert not np.allclose(v[:, 0].T @ v[:, 1], 0)


def test_normal_same_eigenbasis_AA_star():
    """A and A*A share eigenvectors if A is normal"""
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    D = np.diag(rng.normal(size=3) + 1j * rng.normal(size=3))
    A = Q @ D @ Q.conj().T  # normal matrix
    A_star_A = A.conj().T @ A
    _, V = np.linalg.eig(A)
    for v in V.T:
        assert np.allclose(A_star_A @ v, (v.conj().T @ A_star_A @ v) * v)


def test_normal_unitary_diagonalizable():
    """Normal matrix can be diagonalized by unitary matrix"""
    theta = rng.uniform(0, 2 * np.pi)
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    _, v = np.linalg.eig(A)
    assert np.allclose(v.conj().T @ v, np.eye(2))


def test_doubly_stochastic_eigenvalues():
    """2x2 doubly stochastic matrix eigenvalues and vectors"""
    p = rng.uniform(0, 1)
    A = np.array([[p, 1 - p], [1 - p, p]])
    vals, vecs = np.linalg.eig(A)
    assert np.isclose(vals[0], 1)
    assert np.allclose(
        vecs[:, 0] / np.linalg.norm(vecs[:, 0]), [1, 1] / np.linalg.norm([1, 1])
    )


def test_quaternion_matrix_eigenvalues():
    """Quaternion matrix has eigenvalues p ± i|v|"""
    p, q, r, s = rng.random(4)
    A = np.array([[p, -q, -r, -s], [q, p, s, -r], [r, -s, p, q], [s, r, -q, p]])
    eigvals = np.linalg.eigvals(A)
    expected = [p + 1j * np.linalg.norm([q, r, s]), p - 1j * np.linalg.norm([q, r, s])]
    assert all(np.any(np.isclose(eigvals, e)) for e in expected)


def test_sum_symmetric_is_symmetric():
    """Sum of symmetric matrices is symmetric"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    assert np.allclose((A + B).T, A + B)


def test_product_not_necessarily_symmetric():
    """Product of symmetric matrices may not be symmetric"""
    A = rng.random((3, 3))
    B = rng.random((3, 3))
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    AB = A @ B
    assert not np.allclose(AB, AB.T)


def test_inverse_of_symmetric_is_symmetric():
    """Inverse of symmetric matrix is symmetric"""
    A = rng.random((3, 3))
    A = A @ A.T + np.eye(3)  # Make symmetric and positive definite
    A_inv = np.linalg.inv(A)
    assert np.allclose(A_inv, A_inv.T)


def test_bt_b_is_symmetric():
    """A = BᵀB is symmetric"""
    B = rng.random((4, 3))
    A = B.T @ B
    assert np.allclose(A, A.T)


def test_similarity_preserves_symmetry():
    """Symmetry is preserved under orthogonal similarity"""
    A = rng.random((3, 3))
    A = (A + A.T) / 2  # Make A symmetric
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # Q is orthogonal
    B = Q.T @ A @ Q
    assert np.allclose(B, B.T)


def test_only_zero_is_both_sym_and_antisym():
    """Only zero matrix is symmetric and anti-symmetric"""
    A = np.zeros((3, 3))
    assert np.allclose(A, A.T)
    assert np.allclose(A, -A)


def test_normal_matrices_not_vector_space():
    """Normal matrices don't form linear space"""
    theta = rng.uniform(0, 2 * np.pi)
    A = np.array([[0, -1], [1, 0]]) * np.cos(theta)
    B = np.diag(rng.random(2))
    C = A + B
    assert not np.allclose(C @ C.T, C.T @ C)


def test_matrix_with_simple_spectrum():
    """Matrix with distinct eigenvalues has simple spectrum"""
    vals = np.sort(rng.choice(np.arange(1, 100), size=3, replace=False))
    A = np.diag(vals)
    eigvals = np.linalg.eigvals(A)
    assert len(np.unique(eigvals)) == 3


def test_spectral_theorem_eigenbasis():
    """Spectral theorem: symmetric matrix has eigenbasis"""
    A = rng.random((4, 4))
    A = (A + A.T) / 2
    _, v = np.linalg.eig(A)
    assert np.allclose(v.T @ v, np.eye(4))


def test_bunny_graph_laplacian_spectrum():
    """Bunny graph Laplacian has real spectrum"""
    A = np.array(
        [
            [2, -1, -1, 0, 0],
            [-1, 2, -1, 0, 0],
            [-1, -1, 4, -1, -1],
            [0, 0, -1, 1, 0],
            [0, 0, -1, 0, 1],
        ]
    )
    assert np.all(np.isreal(np.linalg.eigvals(A)))


def test_rotation_matrix_is_normal():
    """Rotation matrices are normal"""
    theta = rng.uniform(0, 2 * np.pi)
    A = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A_star = A.T
    assert np.allclose(A @ A_star, A_star @ A)
