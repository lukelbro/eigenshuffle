from typing import Callable

import numpy as np
import numpy.typing as npt

EighFn = Callable[
    [npt.NDArray[np.floating]],
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]],
]


def _mlx_eigh(
    mat: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    import mlx.core as mx

    mat = np.asarray(mat)
    # mx.array downcasts float64 to float32 unless the dtype is explicit
    a = mx.array(mat, dtype=mx.float64) if mat.dtype == np.float64 else mx.array(mat)
    vals, vecs = mx.linalg.eigh(a, stream=mx.cpu)
    return np.array(vals), np.array(vecs)


def get_eigh(backend: str) -> EighFn:
    if backend == "numpy":
        return np.linalg.eigh
    if backend == "mlx":
        try:
            import mlx.core  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='mlx' requires the mlx package: pip install eigenshuffle[mlx]"
            ) from e
        return _mlx_eigh
    raise ValueError(f"unknown backend {backend!r}; expected 'numpy' or 'mlx'")


def distance_matrix(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    return np.abs(vec1[:, np.newaxis] - vec2[np.newaxis, :]).T


def compute_indx_map(vecs: npt.NDArray[np.floating]) -> dict[int, int]:
    inv_vs = np.linalg.inv(vecs)
    indxs = np.argmax(np.abs(inv_vs) ** 2, axis=0)
    return {int(val): idx for idx, val in enumerate(indxs)}


def apply_sign_correction(
    prev_vecs: npt.NDArray[np.inexact],
    vecs: npt.NDArray[np.inexact],
    tol: float = 1e-12,
) -> npt.NDArray[np.inexact]:
    overlaps = np.sum(prev_vecs.conj() * vecs, axis=0)
    if np.isrealobj(prev_vecs) and np.isrealobj(vecs):
        return vecs * np.where(overlaps < 0, -1.0, 1.0)
    # align complex phases; the tol floor avoids blowing up when |overlap|≈0
    return vecs * (overlaps.conj() / np.maximum(np.abs(overlaps), tol))
