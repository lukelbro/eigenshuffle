from typing import Literal, Sequence, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from tqdm import tqdm


class LineLoggingTQDM(tqdm):
    def __iter__(self):
        for item in super().__iter__():
            msg = self.format_meter(**self.format_dict)
            tqdm.write(f"{self.desc or ''} | {msg}")
            yield item



__all__ = ["eigenshuffle_eig", "eigenshuffle_eigh", "eigenshuffle_eighvals", "eigenshuffle_eigvals"]

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


def eigenshuffle_eighvals(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = False,
    progress: bool = True,
    dtype: np.dtype | None = None,
    use_gpu: bool = False, 
    count: int | None = None,
    use_jax: bool = False,  # New argument
) -> tuple[npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], npt.NDArray[np.int_]]:
    """
    Compute eigenvalues only with eigh (Hermitian) of a series of matrices (mxnxn) and keep eigenvalues consistently sorted; starting with the lowest eigenvalue.

    Args:
        matrices: mxnxn array of eigenvalue problems
        use_eigenvalues: Use the distance between successive eigenvalues as part of the shuffling. Defaults to False.
        progress: show progress bar if True.
        dtype: Desired dtype for output.
        use_gpu: Use GPU for diagonalization if True.
        count: Number of matrices when using a generator.
        use_jax: Use JAX-based Hungarian matcher for assignment if True.

    Returns:
        tuple: (sorted eigenvalues, indx_map from first set of eigenvectors)
    """
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

    if progress and is_callable:
        from tqdm import tqdm as _tqdm
        _tqdm.write("Generating matrix elements..")
    sample = get_mat(0)
    n = sample.shape[-1]
    in_dtype = sample.dtype
    out_dtype = dtype if dtype is not None else np.promote_types(in_dtype, np.complex64)

    values = np.empty((m, n), dtype=out_dtype)

    # Set up diagonalization function for all four cases
    if use_jax:
        import jax
        import jax.numpy as jnp
        from jax import device_put, jit
        from eigenshuffle.hungarian_cover import hungarian_single
        hungarian_single_jit = jit(hungarian_single)
        jax_gpu_available = any([d.platform == 'gpu' for d in jax.devices()])
        def eigh_func(mat):
            # JAX + GPU
            if use_gpu and jax_gpu_available:
                mat_jax = device_put(mat.astype(np.complex64), device=jax.devices('gpu')[0])
            # JAX + CPU
            else:
                mat_jax = jnp.asarray(mat)
            vals, vecs = jnp.linalg.eigh(mat_jax)
            return np.array(vals), np.array(vecs)
    elif use_gpu:
        # CuPy (GPU, not JAX)
        try:
            import importlib
            cp = importlib.import_module("cupy")
        except ImportError:
            raise ImportError("CuPy is required for GPU diagonalization. Install cupy-cudaXX.")
        def eigh_func(mat):
            mat_gpu = cp.asarray(mat, dtype=out_dtype)
            vals_gpu, vecs_gpu = cp.linalg.eigh(mat_gpu)
            vals = cp.asnumpy(vals_gpu)
            vecs = cp.asnumpy(vecs_gpu)
            return vals, vecs
    else:
        # NumPy (CPU, not JAX)
        def eigh_func(mat):
            return np.linalg.eigh(mat)

    # diagonalize, sort initial frame
    if progress:
        _tqdm.write("Diagonalizing first matrix...")
    vals, vecs = eigh_func(sample)
    values[0] = vals
    prev_vecs = vecs

    if progress:
        _tqdm.write("Calculating state mapping")
        _tqdm.write("(a) inverting eigenvector matrix")
    # Use first set of eigenvectors for state mapping
    if use_gpu:
        try:
            import importlib
            cp = importlib.import_module("cupy")
        except ImportError:
            raise ImportError("CuPy is required for GPU diagonalization. Install cupy-cudaXX.")
        inv_vs_gpu = cp.linalg.inv(cp.asarray(vecs))
        if progress:
            _tqdm.write("(b) calculating abs(v)^2 and argmax on GPU")
        indxs_gpu = cp.argmax(cp.abs(inv_vs_gpu) ** 2, axis=0)
        indxs = cp.asnumpy(indxs_gpu)
    else:
        inv_vs = np.linalg.inv(vecs)
        if progress:
            _tqdm.write("(b) calculating abs(v)^2 and argmax on CPU")
        indxs = np.argmax(np.abs(inv_vs) ** 2, axis=0)
    if progress:
        _tqdm.write("(c) mapping state indices")
    indx_map = {val: idx for idx, val in enumerate(indxs)}

    idxs = range(1, m)
    iterator =  LineLoggingTQDM(idxs, desc="Time to complete eigval diag+shuffle", unit="matrix") if progress else idxs
    for i in iterator:
        mat = get_mat(i)
        vals, vecs = eigh_func(mat)
        distance = 1 - np.abs(prev_vecs.T @ vecs)
        if use_eigenvalues:
            dist_vals = np.sqrt(
                distance_matrix(values[i-1].real, vals.real)**2 +
                distance_matrix(values[i-1].imag, vals.imag)**2
            )
            distance *= dist_vals
        if use_jax:
            if use_gpu:
                distance_jax = device_put(distance.astype(np.float32), device=jax.devices('gpu')[0])
            else:
                distance_jax = jnp.asarray(distance)
            assignment = hungarian_single_jit(distance_jax)
            col_ind = np.array(assignment[1])
        else:
            _, col_ind = linear_sum_assignment(distance)
        vals, vecs = vals[col_ind], vecs[:, col_ind]
        dot = np.sum(prev_vecs * vecs, axis=0).real
        flip = -(((dot < 0).astype(int) * 2) + 1)
        vecs *= flip
        values[i] = vals
        prev_vecs = vecs
    return values, indx_map


def eigenshuffle_eigvals(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = False,
    progress: bool = True,
    dtype: np.dtype | None = None,
    count: int | None = None,
    use_mlx: bool = False,  # New argument
) -> tuple[npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], npt.NDArray[np.int_]]:
    """
    Compute eigenvalues only with eig (general, not Hermitian) of a series of matrices (mxnxn) and keep eigenvalues consistently sorted; starting with the lowest eigenvalue.

    Args:
        matrices: mxnxn array of eigenvalue problems
        use_eigenvalues: Use the distance between successive eigenvalues as part of the shuffling. Defaults to False.
        progress: show progress bar if True.
        dtype: Desired dtype for output.
        count: Number of matrices when using a generator.
        use_mlx: Use mlx for eigenvalue computation if True.

    Returns:
        tuple: (sorted eigenvalues, indx_map from first set of eigenvectors)
    """
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

    sample = get_mat(0)
    n = sample.shape[-1]
    in_dtype = sample.dtype
    out_dtype = dtype if dtype is not None else np.promote_types(in_dtype, np.complex64)

    values = np.empty((m, n), dtype=out_dtype)

    if use_mlx:
        import mlx.core as mx
        def eig_func(mat):
            A_mx = mx.array(mat, dtype=mx.float32)
            eigvals_mx, eigvecs_mx = mx.linalg.eig(A_mx, stream=mx.cpu)
            return np.array(eigvals_mx, copy=False), np.array(eigvecs_mx, copy=False)
    else:
        def eig_func(mat):
            return np.linalg.eig(mat)

    # diagonalize, sort initial frame
    vals, vecs = eig_func(sample)
    idx_sort = np.argsort(vals.real)
    vals, vecs = vals[idx_sort], vecs[:, idx_sort]
    values[0] = vals
    prev_vecs = vecs

   
    inv_vs = np.linalg.inv(vecs)
    indxs = np.argmax(np.abs(inv_vs) ** 2, axis=0)
    indx_map = {val: idx for idx, val in enumerate(indxs)}

    idxs = range(1, m)
    iterator = tqdm(idxs, desc="Time to complete eigval diag+shuffle", unit="matrix") if progress else idxs
    for i in iterator:
        mat = get_mat(i)
        vals, vecs = eig_func(mat)
        idx_sort = np.argsort(vals.real)
        vals, vecs = vals[idx_sort], vecs[:, idx_sort]
        distance = 1 - np.abs(prev_vecs.T @ vecs)
        if use_eigenvalues:
            dist_vals = np.sqrt(
                distance_matrix(values[i-1].real, vals.real)**2 +
                distance_matrix(values[i-1].imag, vals.imag)**2
            )
            distance *= dist_vals
        _, col_ind = linear_sum_assignment(distance)
        vals, vecs = vals[col_ind], vecs[:, col_ind]
        dot = np.sum(prev_vecs * vecs, axis=0).real
        flip = -(((dot < 0).astype(int) * 2) + 1)
        vecs *= flip
        values[i] = vals
        prev_vecs = vecs
    return values, indx_map


