from __future__ import annotations

import gc
import concurrent.futures
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ._core import apply_sign_correction, compute_indx_map


def eigenshuffle(
    matrices: (
        npt.NDArray[np.floating]
        | Sequence[npt.NDArray[np.floating]]
        | Callable[[int], npt.NDArray[np.floating]]
    ),
    count: int | None = None,
    dtype: npt.DTypeLike = np.float64,
    progress: bool = False,
    max_workers: int = 4,
    max_parallel_diag: int = 4,
    force_gc: bool = False,
) -> tuple[npt.NDArray[np.floating], dict[int, int]]:
    """
    Track eigenvalues across a sequence of Hermitian matrices.

    Uses ``numpy.linalg.eigh`` for diagonalisation and
    ``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm) for optimal
    eigenvector matching between consecutive steps.  Diagonalisation and
    assignment are overlapped via a ``ThreadPoolExecutor``.

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
        matrices larger than 1500×1500.

    Returns
    -------
    values : ndarray of shape (m, n)
        Tracked eigenvalues.  Row ``i`` corresponds to input matrix ``i``.
        Columns track the same physical state across all rows.
    indx_map : dict[int, int]
        Mapping from physical-state label to column index, derived from the
        dominant-weight decomposition of the first eigenvector matrix.
    """
    is_callable = callable(matrices)
    if is_callable and count is None:
        raise ValueError("`count` must be provided when passing a matrix factory")

    if is_callable:
        m: int = count  # type: ignore[assignment]

        def get_mat(i: int) -> npt.NDArray[np.floating]:
            return matrices(i)  # type: ignore[operator]
    else:
        arr = np.asarray(matrices)
        if arr.ndim != 3:
            raise ValueError("matrices must be shape (m, n, n)")
        m = arr.shape[0]

        def get_mat(i: int) -> npt.NDArray[np.floating]:
            return arr[i]

    sample = get_mat(0)
    n = sample.shape[-1]
    out_dtype = np.dtype(dtype)

    if not force_gc and n >= 1500:
        force_gc = True

    values = np.empty((m, n), dtype=out_dtype)

    vals, vecs = np.linalg.eigh(sample)
    values[0] = vals.astype(out_dtype)
    prev_vecs = vecs
    del sample

    indx_map = compute_indx_map(vecs)
    del vals

    if m == 1:
        return values, indx_map

    iterator = (
        tqdm(range(1, m), desc="eigenshuffle", unit="step")
        if progress
        else range(1, m)
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        diag_futures: dict[int, concurrent.futures.Future] = {}
        for i in range(1, min(m, 1 + max_parallel_diag)):
            diag_futures[i] = executor.submit(np.linalg.eigh, get_mat(i))

        for i in iterator:
            if i in diag_futures:
                vals, vecs = diag_futures.pop(i).result()
            else:
                vals, vecs = np.linalg.eigh(get_mat(i))

            next_idx = i + max_parallel_diag
            if next_idx < m and next_idx not in diag_futures:
                diag_futures[next_idx] = executor.submit(
                    np.linalg.eigh, get_mat(next_idx)
                )

            distance = 1.0 - np.abs(prev_vecs.T @ vecs)
            _, col_ind = linear_sum_assignment(distance)
            del distance

            vals, vecs = vals[col_ind], vecs[:, col_ind]
            vecs = apply_sign_correction(prev_vecs, vecs)

            values[i] = vals.astype(out_dtype)
            prev_vecs = vecs
            del vals, vecs

            if force_gc:
                gc.collect()

    return values, indx_map
