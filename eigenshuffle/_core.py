import numpy as np
import numpy.typing as npt


def distance_matrix(
    vec1: npt.NDArray[np.floating], vec2: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    return np.abs(vec1[:, np.newaxis] - vec2[np.newaxis, :]).T


def compute_indx_map(vecs: npt.NDArray[np.floating]) -> dict[int, int]:
    inv_vs = np.linalg.inv(vecs)
    indxs = np.argmax(np.abs(inv_vs) ** 2, axis=0)
    return {int(val): idx for idx, val in enumerate(indxs)}


def apply_sign_correction(
    prev_vecs: npt.NDArray[np.floating], vecs: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    dot = np.sum(prev_vecs * vecs, axis=0).real
    flip = -(((dot < 0).astype(int) * 2) + 1)
    return vecs * flip
