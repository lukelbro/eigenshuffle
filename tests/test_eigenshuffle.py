import numpy as np
import pytest

from eigenshuffle import eigenshuffle

from .conftest import EXPECTED_EIGENVALUES, T_SEQ, make_matrix


class TestArrayInput:
    def test_eigenvalues_match_reference(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert np.allclose(values, EXPECTED_EIGENVALUES)

    def test_output_shape(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert values.shape == (len(T_SEQ), 4)

    def test_output_dtype_default(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert values.dtype == np.float64

    def test_output_dtype_explicit_float32(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float32, progress=False)
        assert values.dtype == np.float32

    def test_indx_map_is_dict(self, matrix_array):
        _, indx_map = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert isinstance(indx_map, dict)
        assert len(indx_map) == 4

    def test_first_step_sorted_ascending(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert np.all(np.diff(values[0]) > 0)


class TestFactoryInput:
    def test_eigenvalues_match_reference(self, matrix_factory):
        values, _ = eigenshuffle(
            matrix_factory, count=len(T_SEQ), dtype=np.float64, progress=False
        )
        assert np.allclose(values, EXPECTED_EIGENVALUES)

    def test_factory_matches_array(self, matrix_array, matrix_factory):
        v_arr, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        v_fac, _ = eigenshuffle(
            matrix_factory, count=len(T_SEQ), dtype=np.float64, progress=False
        )
        assert np.allclose(v_arr, v_fac)

    def test_count_required_for_callable(self, matrix_factory):
        with pytest.raises(ValueError):
            eigenshuffle(matrix_factory, dtype=np.float64, progress=False)


class TestParallelism:
    @pytest.mark.parametrize("workers,parallel", [(1, 1), (2, 2), (4, 4), (2, 4)])
    def test_parallelism_variants(self, matrix_array, workers, parallel):
        values, _ = eigenshuffle(
            matrix_array,
            dtype=np.float64,
            max_workers=workers,
            max_parallel_diag=parallel,
            progress=False,
        )
        assert np.allclose(values, EXPECTED_EIGENVALUES)


class TestEdgeCases:
    def test_single_matrix(self):
        mat = make_matrix(0.0)
        arr = mat[np.newaxis, ...]
        values, indx_map = eigenshuffle(arr, dtype=np.float64, progress=False)
        assert values.shape == (1, 4)
        assert isinstance(indx_map, dict)

    def test_two_matrices(self):
        arr = np.array([make_matrix(t) for t in [-1.0, 0.0]])
        values, _ = eigenshuffle(arr, dtype=np.float64, progress=False)
        assert values.shape == (2, 4)

    def test_force_gc_produces_same_result(self, matrix_array):
        v_no_gc, _ = eigenshuffle(
            matrix_array, dtype=np.float64, force_gc=False, progress=False
        )
        v_gc, _ = eigenshuffle(
            matrix_array, dtype=np.float64, force_gc=True, progress=False
        )
        assert np.allclose(v_no_gc, v_gc)

    def test_identity_of_eigenvalue_tracking(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert np.allclose(values, EXPECTED_EIGENVALUES)


class TestNumericalProperties:
    def test_eigenvalues_are_real(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        assert np.isrealobj(values) or np.allclose(values.imag, 0)

    def test_each_step_is_permutation_of_eigh(self, matrix_array):
        values, _ = eigenshuffle(matrix_array, dtype=np.float64, progress=False)
        for i, mat in enumerate(matrix_array):
            raw_vals = np.linalg.eigvalsh(mat)
            assert np.allclose(np.sort(values[i]), np.sort(raw_vals))


class TestBackend:
    def test_unknown_backend_raises(self, matrix_array):
        with pytest.raises(ValueError):
            eigenshuffle(matrix_array, backend="nope")

    def test_mlx_matches_numpy(self, matrix_array):
        pytest.importorskip("mlx.core")
        values, _ = eigenshuffle(matrix_array, backend="mlx")
        assert values.dtype == np.float64
        assert np.allclose(values, EXPECTED_EIGENVALUES)

    def test_mlx_factory(self, matrix_factory):
        pytest.importorskip("mlx.core")
        values, _ = eigenshuffle(matrix_factory, count=len(T_SEQ), backend="mlx")
        assert np.allclose(values, EXPECTED_EIGENVALUES)


class TestReturnVectors:
    def test_default_returns_two(self, matrix_array):
        assert len(eigenshuffle(matrix_array)) == 2

    def test_vectors_shape_and_eigenpairs(self, matrix_array):
        values, _, vectors = eigenshuffle(matrix_array, return_vectors=True)
        assert vectors.shape == matrix_array.shape
        for i, mat in enumerate(matrix_array):
            assert np.allclose(mat @ vectors[i], vectors[i] * values[i])
            assert np.allclose(vectors[i].T @ vectors[i], np.eye(4))

    def test_values_unchanged(self, matrix_array):
        v_plain, _ = eigenshuffle(matrix_array)
        v_vec, _, _ = eigenshuffle(matrix_array, return_vectors=True)
        assert np.allclose(v_plain, v_vec)

    def test_signs_continuous(self, matrix_array):
        _, _, vectors = eigenshuffle(matrix_array, return_vectors=True)
        overlaps = np.einsum("mij,mij->mj", vectors[:-1], vectors[1:])
        assert np.all(overlaps > 0)

    def test_single_matrix(self):
        arr = make_matrix(0.0)[np.newaxis, ...]
        _, _, vectors = eigenshuffle(arr, return_vectors=True)
        assert vectors.shape == (1, 4, 4)


class TestComplexHermitian:
    @pytest.fixture
    def complex_array(self, matrix_array):
        # unitary rotation of the real sequence: same spectrum, complex vectors
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
        return q @ matrix_array @ q.conj().T

    def test_eigenvalues_match_reference(self, complex_array):
        values, _ = eigenshuffle(complex_array)
        assert np.allclose(values, EXPECTED_EIGENVALUES)

    def test_phases_continuous(self, complex_array):
        values, _, vectors = eigenshuffle(complex_array, return_vectors=True)
        assert np.iscomplexobj(vectors)
        for i, mat in enumerate(complex_array):
            assert np.allclose(mat @ vectors[i], vectors[i] * values[i])
            assert np.allclose(vectors[i].conj().T @ vectors[i], np.eye(4))
        overlaps = np.einsum("mij,mij->mj", vectors[:-1].conj(), vectors[1:])
        assert np.allclose(overlaps.imag, 0)
        assert np.all(overlaps.real > 0)
