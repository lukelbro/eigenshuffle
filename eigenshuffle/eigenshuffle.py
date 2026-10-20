from __future__ import annotations

import gc
import concurrent.futures
from typing import Callable, Literal, Sequence, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ._core import apply_sign_correction, compute_indx_map, get_eigh


@overload
def eigenshuffle(
    matrices: (
        npt.NDArray[np.inexact]
        | Sequence[npt.NDArray[np.inexact]]
        | Callable[[int], npt.NDArray[np.inexact]]
    ),
    count: int | None = ...,
    dtype: npt.DTypeLike = ...,
    progress: bool = ...,
    max_workers: int = ...,
    max_parallel_diag: int = ...,
    force_gc: bool = ...,
    backend: Literal["numpy", "mlx"] = ...,
    return_vectors: Literal[False] = ...,
) -> tuple[npt.NDArray[np.floating], dict[int, int]]: ...


@overload
def eigenshuffle(
    matrices: (
        npt.NDArray[np.inexact]
        | Sequence[npt.NDArray[np.inexact]]
        | Callable[[int], npt.NDArray[np.inexact]]
    ),
    count: int | None = ...,
    dtype: npt.DTypeLike = ...,
    progress: bool = ...,
    max_workers: int = ...,
    max_parallel_diag: int = ...,
    force_gc: bool = ...,
    backend: Literal["numpy", "mlx"] = ...,
    *,
    return_vectors: Literal[True],
) -> tuple[npt.NDArray[np.floating], dict[int, int], npt.NDArray[np.inexact]]: ...


def eigenshuffle(
    matrices: (
        npt.NDArray[np.inexact]
        | Sequence[npt.NDArray[np.inexact]]
        | Callable[[int], npt.NDArray[np.inexact]]
    ),
    count: int | None = None,
    dtype: npt.DTypeLike = np.float64,
    progress: bool = False,
    max_workers: int = 4,
    max_parallel_diag: int = 4,
    force_gc: bool = False,
    backend: Literal["numpy", "mlx"] = "numpy",
    return_vectors: bool = False,
) -> (
    tuple[npt.NDArray[np.floating], dict[int, int]]
    | tuple[npt.NDArray[np.floating], dict[int, int], npt.NDArray[np.inexact]]
):
    """
    Track eigenvalues across a sequence of Hermitian matrices.

    Diagonalises each matrix with ``numpy.linalg.eigh`` (or
    ``mlx.core.linalg.eigh`` when ``backend="mlx"``) and matches eigenvectors
    between consecutive steps with ``scipy.optimize.linear_sum_assignment``
    (Hungarian algorithm).  Upcoming diagonalisations run in a
    ``ThreadPoolExecutor`` while the current step is being matched.

    Parameters
    ----------
    matrices:
        Either an ``(m, n, n)`` array of Hermitian matrices, a list of
        ``(n, n)`` arrays, or a callable ``factory(i) -> (n, n)`` array.
        When a callable is passed ``count`` is required.
    count:
        Number of matrices.  Required when ``matrices`` is callable;
        ignored otherwise.
    dtype:
        Output dtype for the returned eigenvalue array.  Defaults to
        ``np.float64``.
    progress:
        Show a ``tqdm`` progress bar over the assignment loop.
    max_workers:
        Thread-pool size.  Controls how many diagonalisations can run
        concurrently in the background.
    max_parallel_diag:
        How many future diagonalisations are pre-submitted ahead of the
        current assignment step.
    force_gc:
        Call ``gc.collect()`` after every iteration.  Auto-enabled for
        matrices of size 1500×1500 or larger.
    backend:
        Eigensolver to use: ``"numpy"`` (default) or ``"mlx"``.  The MLX
        backend requires the optional ``mlx`` package (Apple silicon) and
        runs on the CPU stream, since MLX has no GPU ``eigh``.
    return_vectors:
        Also return the tracked eigenvectors.  This stores an ``(m, n, n)``
        array, as large as the input, so it is off by default.

    Returns
    -------
    values : ndarray of shape (m, n)
        Tracked eigenvalues.  Row ``i`` corresponds to input matrix ``i``.
        Columns track the same physical state across all rows.
    indx_map : dict[int, int]
        Mapping from column index of ``values`` to the index of the basis
        state with the largest weight in that column's eigenvector at the
        first step.  If one eigenvector dominates several basis states, only
        the last of them is kept.
    vectors : ndarray of shape (m, n, n)
        Only returned when ``return_vectors=True``.  ``vectors[i][:, j]`` is
        the eigenvector for ``values[i, j]``.  Each column's sign (or complex
        phase) is chosen so its overlap with ``vectors[i - 1][:, j]`` is real
        and non-negative.
    """
    eigh = get_eigh(backend)

    is_callable = callable(matrices)
    if is_callable and count is None:
        raise ValueError("`count` must be provided when passing a matrix factory")

    if is_callable:
        m: int = count  # type: ignore[assignment]

        def get_mat(i: int) -> npt.NDArray[np.inexact]:
            return matrices(i)  # type: ignore[operator]
    else:
        arr = np.asarray(matrices)
        if arr.ndim != 3:
            raise ValueError("matrices must be shape (m, n, n)")
        m = arr.shape[0]

        def get_mat(i: int) -> npt.NDArray[np.inexact]:
            return arr[i]

    sample = get_mat(0)
    n = sample.shape[-1]
    out_dtype = np.dtype(dtype)

    if not force_gc and n >= 1500:
        force_gc = True

    values = np.empty((m, n), dtype=out_dtype)

    vals, vecs = eigh(sample)
    values[0] = vals.astype(out_dtype)
    vectors = np.empty((m, n, n), dtype=vecs.dtype) if return_vectors else None
    if vectors is not None:
        vectors[0] = vecs
    prev_vecs = vecs
    del sample

    indx_map = compute_indx_map(vecs)
    del vals

    if m == 1:
        if vectors is not None:
            return values, indx_map, vectors
        return values, indx_map

    iterator = (
        tqdm(range(1, m), desc="eigenshuffle", unit="step")
        if progress
        else range(1, m)
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        diag_futures: dict[int, concurrent.futures.Future] = {}
        for i in range(1, min(m, 1 + max_parallel_diag)):
            diag_futures[i] = executor.submit(eigh, get_mat(i))

        for i in iterator:
            if i in diag_futures:
                vals, vecs = diag_futures.pop(i).result()
            else:
                vals, vecs = eigh(get_mat(i))

            next_idx = i + max_parallel_diag
            if next_idx < m and next_idx not in diag_futures:
                diag_futures[next_idx] = executor.submit(
                    eigh, get_mat(next_idx)
                )

            distance = 1.0 - np.abs(prev_vecs.conj().T @ vecs)
            _, col_ind = linear_sum_assignment(distance)
            del distance

            vals, vecs = vals[col_ind], vecs[:, col_ind]
            vecs = apply_sign_correction(prev_vecs, vecs)

            values[i] = vals.astype(out_dtype)
            if vectors is not None:
                vectors[i] = vecs
            prev_vecs = vecs
            del vals, vecs

            if force_gc:
                gc.collect()

    if vectors is not None:
        return values, indx_map, vectors
    return values, indx_map
