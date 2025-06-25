from typing import Literal, Sequence, TypeVar, overload

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment


__all__ = ["eigenshuffle_eig", "eigenshuffle_eigh"]

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
) -> tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]:
    """
    Consistently reorder eigenvalues/vectors based on the initial ordering.
    Uses SciPy's linear_sum_assignment to solve the assignment problem that finds
    the best mapping between two successive systems, and then adjusts sign consistency.

    Args:
        eigenvalues (eigenvals_complex_or_float): mxn eigenvalues.
        eigenvectors (eigenvecs_complex_or_float): mxnxn eigenvectors.
        use_eigenvalues (bool, optional): whether to use eigenvalues in the distance calculation.

    Returns:
        tuple[eigenvals_complex_or_float, eigenvecs_complex_or_float]: consistently ordered eigenvalues and eigenvectors.
    """
    for i in range(1, len(eigenvalues)):
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
        # Replicate original factor: if dot < 0, multiply by -3; else by -1.
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
):
    """
    Consistently reorder eigenvalues and eigenvectors based on the initial ordering,
    which sorts the eigenvalues from low to

    Args:
        matrices (Sequence[NDArray], NDArray): eigenvalue/vector problems
        hermitian (bool): bool specifying hermitian
        use_eigenvalues (bool): bool specifying use of eigenvalues for re-ordering in _shuffle

    Returns:
        tuple[NDArray, NDArray]: consistently ordered eigenvalues/vectors
    """
    assert len(np.shape(matrices)) > 2, "matrices must be of shape mxnxn"

    if hermitian:
        eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(matrices)

    eigenvalues, eigenvectors = _reorder(
        eigenvalues=eigenvalues, eigenvectors=eigenvectors
    )
    eigenvalues, eigenvectors = _shuffle(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        use_eigenvalues=use_eigenvalues,
    )
    return eigenvalues, eigenvectors


def eigenshuffle_eigh(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = True,
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

    Returns:
        tuple[ npt.NDArray[np.floating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], ]: sorted eigenvalues and eigenvectors
    """
    return _eigenshuffle(matrices, hermitian=True, use_eigenvalues=use_eigenvalues)


def eigenshuffle_eig(
    matrices: Sequence[npt.NDArray[np.floating]]
    | npt.NDArray[np.floating]
    | Sequence[npt.NDArray[np.complexfloating]]
    | npt.NDArray[np.complexfloating],
    use_eigenvalues: bool = False,
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

    Returns:
        tuple[ npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], npt.NDArray[np.floating] | npt.NDArray[np.complexfloating], ]: sorted eigenvalues and eigenvectors
    """
    return _eigenshuffle(matrices, hermitian=False, use_eigenvalues=use_eigenvalues)
