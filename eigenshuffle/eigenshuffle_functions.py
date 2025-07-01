from typing import Literal, Sequence, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from tqdm.notebook import tqdm



__all__ = ["eigenshuffle_eig", "eigenshuffle_eigh", "eigenshuffle_eigvals"]

eigenvals_complex_or_float = TypeVar(
    "eigenvals_complex_or_float",
    npt.NDArray[np.floating],
    npt.NDArray[np.complexfloating],
)

eigenvecs_complex_or_float = TypeVar(
    "eigenvecs_complex_or_float",
    npt.NDArray[np.floating],
    npt.NDArray[np.complexfloating],
)


def distance_matrix(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Compute the interpoint distance matrix

    Args:
        vec1 (npt.NDArray[np.floating]): vector1
        vec2 (npt.NDArray[np.floating]): vector2

    Returns:
        npt.NDArray[np.floating]: interpoint distance matrix between vector1 and vector2
    """
    return np.abs(vec1[:, np.newaxis] - vec2[np.newaxis, :]).T


def _shuffle(
    eigenvalues: eigenvals_complex_or_float,
    eigenvectors: eigenvecs_complex_or_float,
    use_eigenvalues: bool = True,
    progress: bool = True,
) -> tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]:
    """
    Consistently reorder eigenvalues/vectors based on the initial ordering.
    Uses SciPy's linear_sum_assignment to solve the assignment problem that finds
    the best mapping between two successive systems, and then adjusts sign consistency.
    Optionally displays a progress bar for the shuffling phase.

    Args:
        eigenvalues (eigenvals_complex_or_float): mxn eigenvalues.
        eigenvectors (eigenvecs_complex_or_float): mxnxn eigenvectors.
        use_eigenvalues (bool, optional): whether to use eigenvalues in the distance calculation.
        progress (bool, optional): show progress bar if True.

    Returns:
        tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]: consistently ordered eigenvalues and eigenvectors.
    """
    iterator = tqdm(range(1, len(eigenvalues)),
                    desc="Time to complete shuffle",
                    unit="pair") if progress else range(1, len(eigenvalues))
    
    for i in iterator:
        # Compute distance between systems
        D1, D2 = eigenvalues[i - 1 : i + 1]
        V1, V2 = eigenvectors[i - 1 : i + 1]

        distance = 1 - np.abs(V1.T @ V2)
        if use_eigenvalues:
            dist_vals = np.sqrt(
                distance_matrix(D1.real, D2.real) ** 2 +
                distance_matrix(D1.imag, D2.imag) ** 2
            )
            distance *= dist_vals

        # Use SciPy's linear_sum_assignment to get optimal assignment.
        _, col_ind = linear_sum_assignment(distance)
        reorder = col_ind  # permutation of indices

        eigenvectors[i] = eigenvectors[i][:, reorder]
        eigenvalues[i] = eigenvalues[i, reorder]

        # Adjust the sign consistency as in the original implementation.
        dot = np.sum(eigenvectors[i - 1] * eigenvectors[i], axis=0).real
        flip_factors = -(((dot < 0).astype(int) * 2) + 1)
        eigenvectors[i] = eigenvectors[i] * flip_factors

    return eigenvalues, eigenvectors


def _reorder(
    eigenvalues: eigenvals_complex_or_float, eigenvectors: eigenvecs_complex_or_float
) -> tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]:
    """
    Reorder eigenvalues (mxn) and eigenvectors (mxnxn) for each i entry (m) from low
    to high.

    Args:
        eigenvalues (eigenvals_complex_or_float): mxn eigenvalue array
        eigenvectors (eigenvecs_complex_or_float): mxnxn eigenvector array

    Returns:
        tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]: reordered eigenvalues and eigenvectors
    """
    indices_sort_all = np.argsort(eigenvalues.real)
    for i in range(len(eigenvalues)):
        # initial ordering is purely in decreasing order.
        # If any are complex, the sort is in terms of the
        # real part.
        indices_sort = indices_sort_all[i]

        eigenvalues[i] = eigenvalues[i][indices_sort]
        eigenvectors[i] = eigenvectors[i][:, indices_sort]
    return eigenvalues, eigenvectors


@overload
def _eigenshuffle(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    hermitian: Literal[True],
    use_eigenvalues: bool,
) -> tuple[
    npt.NDArray[np.floating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating]
]: ...


@overload
def _eigenshuffle(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    hermitian: Literal[False],
    use_eigenvalues: bool,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]: ...


def _eigenshuffle(
    matrices,
    hermitian,
    use_eigenvalues,
    progress: bool = True,
    dtype: np.dtype | None = None,
    count: int | None = None,
    use_gpu: bool = False,
):
    """
    Consistently reorder eigenvalues and eigenvectors based on the initial ordering,
    which sorts the eigenvalues from low to high. Diagonalizes each matrix with a progress
    bar indicating "time to complete diagonalization", then performs the shuffling with its own progress display.

    Args:
        matrices (Sequence[NDArray] or NDArray): eigenvalue/vector problems (shape m×n×n)
        hermitian (bool): bool specifying hermitian
        use_eigenvalues (bool): bool specifying use of eigenvalues for re-ordering in _shuffle
        progress (bool, optional): show progress bar if True.

    Returns:
        tuple[NDArray, NDArray]: consistently ordered eigenvalues/vectors
    """
    # allow lazy generator or sequence; require count if matrices is callable
    is_callable = callable(matrices)
    if is_callable and count is None:
        raise ValueError("`count` must be provided when passing a matrix factory")
    # determine iteration indices and matrix getter
    if is_callable:
        m = count
        def get_mat(i): return matrices(i)
    else:
        arr = np.asarray(matrices)
        assert arr.ndim == 3, "matrices must be shape m×n×n"
        m = arr.shape[0]
        def get_mat(i): return arr[i]

    # diagonalize with optional progress over indices
    idxs = range(m)
    iterator = tqdm(idxs, desc="Time to complete diagonalization", unit="matrix") if progress else idxs
    # build first matrix to infer size and dtype
    sample = get_mat(0)
    n = sample.shape[-1]
    in_dtype = sample.dtype

    # choose output dtype (real or complex) defaulting to match input
    if dtype is None:
        if hermitian:
            out_dtype = in_dtype
        else:
            out_dtype = np.promote_types(in_dtype, np.complex64)
    else:
        out_dtype = dtype
    # preallocate outputs
    eigenvalues = np.empty((m, n), dtype=out_dtype)
    eigenvectors = np.empty((m, n, n), dtype=out_dtype)
    # compute each matrix on demand (GPU if requested)
    if use_gpu:
        try:
            import importlib
            cp = importlib.import_module("cupy")
        except ImportError:
            raise ImportError("CuPy is required for GPU diagonalization. Install cupy-cudaXX.")
    for i in iterator:
        mat = get_mat(i)
        # ensure GPU matrix uses desired output dtype
        if use_gpu:
            mat_gpu = cp.asarray(mat, dtype=out_dtype)
            if hermitian:
                vals_gpu, vecs_gpu = cp.linalg.eigh(mat_gpu)
            else:
                vals_gpu, vecs_gpu = cp.linalg.eig(mat_gpu)
            vals = cp.asnumpy(vals_gpu)
            vecs = cp.asnumpy(vecs_gpu)
        else:
            if hermitian:
                vals, vecs = np.linalg.eigh(mat)
            else:
                vals, vecs = np.linalg.eig(mat)
        eigenvalues[i] = vals
        eigenvectors[i] = vecs

    # reorder and shuffle full sequence
    eigenvalues, eigenvectors = _reorder(eigenvalues, eigenvectors)
    eigenvalues, eigenvectors = _shuffle(
        eigenvalues, eigenvectors, use_eigenvalues, progress
    )
    return eigenvalues, eigenvectors


def eigenshuffle_eigh(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = True,
    progress: bool = True,
    dtype: np.dtype | None = None,
    use_gpu: bool = False,
    count: int | None = None,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]:
    """
    Compute eigenvalues and eigenvectors with eigh (hermitian) of a series of matrices
    (mxnxn) and keep eigenvalues and eigenvectors consistently sorted; starting with the
    lowest eigenvalue.

    Args:
        matrices (Sequence[npt.NDArray[np.floating]] | npt.NDArray[np.floating] | Sequence[npt.NDArray[np.complexfloating]] | npt.NDArray[np.complexfloating]): mxnxn array of eigenvalue problems
        use_eigenvalues (bool, optional): Use the distance between successive eigenvalues as part of the shuffling. Defaults to False.
        progress (bool, optional): show progress bar if True.

    Returns:
        tuple[ npt.NDArray[np.floating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], ]: sorted eigenvalues and eigenvectors
    """
    return _eigenshuffle(
        matrices,
        hermitian=True,
        use_eigenvalues=use_eigenvalues,
        progress=progress,
        dtype=dtype,
        count=count,
        use_gpu=use_gpu,
    )


def eigenshuffle_eig(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = False,
    progress: bool = True,
    dtype: np.dtype | None = None,
    use_gpu: bool = False,
    count: int | None = None,
) -> tuple[
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating] | npt.NDArray[np.complexfloating],
]:
    """
    Compute eigenvalues and eigenvectors with eig of a series of matrices (mxnxn) and
    keep eigenvalues and eigenvectors consistently sorted; starting with the lowest
    eigenvalue.

    Args:
        matrices (Sequence[npt.NDArray[np.floating]] | npt.NDArray[np.floating] | Sequence[npt.NDArray[np.complexfloating]] | npt.NDArray[np.complexfloating]): mxnxn array of eigenvalue problems
        use_eigenvalues (bool, optional): Use the distance between successive eigenvalues as part of the shuffling. Defaults to False.
        progress (bool, optional): show progress bar if True.

    Returns:
        tuple[ npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], ]: sorted eigenvalues and eigenvectors
    """
    return _eigenshuffle(
        matrices,
        hermitian=False,
        use_eigenvalues=use_eigenvalues,
        progress=progress,
        dtype=dtype,
        count=count,
        use_gpu=use_gpu,
    )


def eigenshuffle_eigvals(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = False,
    progress: bool = True,
    dtype: np.dtype | None = None,
    use_gpu: bool = False,
    count: int | None = None,
) -> npt.NDArray[np.floating] | npt.NDArray[np.complexfloating]:
    """
    Compute eigenvalues only with eig of a series of matrices (mxnxn) and keep eigenvalues consistently sorted; starting with the lowest eigenvalue.

    Args:
        matrices: mxnxn array of eigenvalue problems
        use_eigenvalues: Use the distance between successive eigenvalues as part of the shuffling. Defaults to False.
        progress: show progress bar if True.
        dtype: Desired dtype for output.
        use_gpu: Use GPU for diagonalization if True.
        count: Number of matrices when using a generator.

    Returns:
        npt.NDArray[np.floating] | npt.NDArray[np.complexfloating]: sorted eigenvalues
    """
    # Memory-efficient iterative eigenvalue shuffle: only two eigenvector sets in memory
    # prepare matrix sequence or factory
    is_callable = callable(matrices)
    if is_callable and count is None:
        raise ValueError("`count` must be provided when passing a matrix factory")
    if is_callable:
        m = count
        get_mat = lambda i: matrices(i)
    else:
        arr = np.asarray(matrices)
        assert arr.ndim == 3, "matrices must be shape m×n×n"
        m = arr.shape[0]
        get_mat = lambda i: arr[i]

    # infer dtype, size, GPU setup
    sample = get_mat(0)
    n = sample.shape[-1]
    in_dtype = sample.dtype
    out_dtype = dtype if dtype is not None else np.promote_types(in_dtype, np.complex64)
    if use_gpu:
        import importlib
        cp = importlib.import_module("cupy")

    # allocate eigenvalues array
    values = np.empty((m, n), dtype=out_dtype)

    # diagonalize, sort initial frame
    if use_gpu:
        mat0_gpu = cp.asarray(sample, dtype=out_dtype)
        v0_gpu, e0_gpu = cp.linalg.eigh(mat0_gpu)
        vals, vecs = cp.asnumpy(v0_gpu), cp.asnumpy(e0_gpu)
    else:
        vals, vecs = np.linalg.eig(sample)
    idx_sort = np.argsort(vals.real)
    vals, vecs = vals[idx_sort], vecs[:, idx_sort]
    values[0] = vals
    prev_vecs = vecs

    # Use first set of eigenvectors for state mapping
    inv_vs = np.linalg.inv(vecs)
    # Compute squared absolute values and then the argmax along rows for each column.
    indx_map = np.argmax(np.abs(inv_vs)**2, axis=0)

    
    # iterate through remaining matrices
    idxs = range(1, m)
    iterator = tqdm(idxs, desc="Time to complete eigval diag+shuffle", unit="matrix") if progress else idxs
    for i in iterator:
        mat = get_mat(i)
        if use_gpu:
            mg = cp.asarray(mat, dtype=out_dtype)
            v_gpu, e_gpu = cp.linalg.eigh(mg)
            vals, vecs = cp.asnumpy(v_gpu), cp.asnumpy(e_gpu)
        else:
            vals, vecs = np.linalg.eig(mat)
        # per-frame sort
        idx_sort = np.argsort(vals.real)
        vals, vecs = vals[idx_sort], vecs[:, idx_sort]
        # compute shuffle assignment
        distance = 1 - np.abs(prev_vecs.T @ vecs)
        if use_eigenvalues:
            dist_vals = np.sqrt(
                distance_matrix(values[i-1].real, vals.real)**2 +
                distance_matrix(values[i-1].imag, vals.imag)**2
            )
            distance *= dist_vals
        _, col_ind = linear_sum_assignment(distance)
        vals, vecs = vals[col_ind], vecs[:, col_ind]
        # enforce sign consistency
        dot = np.sum(prev_vecs * vecs, axis=0).real
        flip = -(((dot < 0).astype(int) * 2) + 1)
        vecs *= flip
        values[i] = vals
        prev_vecs = vecs
    return values, indx_map 
