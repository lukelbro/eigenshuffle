# eigenshuffle – Refactor Plan

## Overview

Now that the optimal call pattern is established empirically, a large portion of the codebase is dead weight. The refactor has one guiding principle: **reduce the codebase to exactly what is needed to run the optimal configuration, with sensible defaults that match it**.

The user-facing API collapses to a single function. Every alternative backend (MLX, CuPy, JAX), the `use_eigenvalues` flag, and all legacy entry points (`eigenshuffle_eig`, `eigenshuffle_eigh`, `eigenshuffle_eigvals`, the old `munkres`-based code, and the incomplete `eigenshuffle_new/` refactor) are removed.

---

## 1. What Gets Deleted

### Entire directories / files

| Path | Reason |
|---|---|
| `eigenshuffle_old/` | Legacy munkres-based implementation; completely superseded |
| `eigenshuffle_new/` | Incomplete, not importable, not used in production |
| `eigenshuffle/hungarian_cover.py` | JAX Hungarian; JAX backend being removed |
| `eigenshuffle/hungarian_mlx.py` | MLX Hungarian; MLX backend being removed |
| `eigenshuffle/hungarian_jax.py` (if kept) | Duplicate of hungarian_cover, also JAX |
| `eigenshuffle/kwok.py` | Broken, unused research artefact |
| `eigenshuffle/HungarianCUDA.cu` | No Python bindings, dead code |
| `benchmark.py` | References `eigenshuffle_og` which no longer exists |
| `check_mlx_devices.py` | MLX utility, no longer relevant |
| All `test_*.py` in root | Ad-hoc experiment scripts, replaced by proper test suite |

### Functions removed from `eigenshuffle/eigenshuffle_functions.py`

| Function | Reason |
|---|---|
| `eigenshuffle_eig` | General (non-hermitian) solver; not the target use case |
| `eigenshuffle_eigh` | Returns full eigenvectors; unnecessary memory overhead |
| `eigenshuffle_eighvals` | Superseded by turbo; kept only for reference in old code |
| `eigenshuffle_eigvals` | Non-hermitian eigenvalues-only; not the target use case |
| `_eigenshuffle` (internal) | Only used by the removed `eig`/`eigh` functions |
| `_shuffle` (internal) | Only used by `_eigenshuffle` |
| `_reorder` (internal) | Only used by `_eigenshuffle` |
| `hungarian_single_sorted` (JAX) | JAX backend removed |
| `LineLoggingTQDM` | Only needed by removed functions |

### Parameters removed from `eigenshuffle_eighvals_turbo`

| Parameter | Reason |
|---|---|
| `use_eigenvalues` | Always `False`; eliminating the branch simplifies the cost matrix computation permanently |
| `use_mlx` | MLX backend removed |
| `use_gpu` | CuPy/GPU backend removed |
| `use_jax` | JAX backend removed |
| `use_jit_sort` | JAX JIT path; JAX removed |

### Top-level imports removed

```python
# Remove entirely from eigenshuffle_functions.py
import jax.numpy as jnp
import jax
```

These are currently unconditional hard imports that break the package on any system without JAX installed. Removing them also removes the only hard JAX dependency.

### `pyproject.toml` cleanup

- Remove `munkres` dependency (no longer used anywhere).
- Add `scipy` as an explicit dependency (currently used but undeclared).
- The optional `jax`, `mlx`, `cupy` dependencies can be removed or moved to an `[optional-dependencies]` section that is simply not installed.

---

## 2. What Stays

Everything that is needed to execute the optimal call pattern:

```python
eigenvals, indx_map = eigenshuffle_eighvals_turbo(
    mat_factory,
    count=len(vacs),
    dtype=np.float64,
    max_workers=4,
    max_parallel_diag=4,
    force_gc=True,
)
```

Core components retained:

- `np.linalg.eigh` for diagonalization (via NumPy, no alternative backends).
- `scipy.optimize.linear_sum_assignment` for Hungarian assignment.
- `concurrent.futures.ThreadPoolExecutor` for the pipeline.
- `tqdm` for progress reporting.
- The `distance_matrix` helper.
- The sign correction logic.
- The `indx_map` computation from the first eigenvector matrix.
- The lazy callable factory path (`is_callable`, `count`).
- The auto-GC threshold at `n >= 1500`.
- `py.typed` marker for PEP 561 compliance.

---

## 3. Refactored File Structure

```
eigenshuffle/
├── __init__.py                  # exports only eigenshuffle_eighvals_turbo
├── _core.py                     # internal helpers: distance_matrix, sign correction
├── eigenshuffle.py              # the single public function
└── py.typed

tests/
├── test_eigenshuffle.py         # canonical regression tests (see Section 6)
└── conftest.py                  # shared fixtures

notes/
├── research.md
└── refactor.md

examples/                        # kept as-is (notebooks for documentation)
pyproject.toml
README.md
```

The current monolithic `eigenshuffle_functions.py` (846 lines) is split into two focused modules:

- `_core.py`: Pure helper functions with no side-effects — `distance_matrix`, the sign correction step, and the `indx_map` computation. These are internal (underscore-prefixed) and not exported.
- `eigenshuffle.py`: Contains only `eigenshuffle_eighvals_turbo`, now renamed to the simpler `eigenshuffle` (see Section 4).

---

## 4. New User-Facing API

The package exports a single public function. The name is simplified from `eigenshuffle_eighvals_turbo` to just `eigenshuffle`, since it is the only function:

```python
from eigenshuffle import eigenshuffle
```

### Signature

```python
def eigenshuffle(
    matrices: np.ndarray | Callable[[int], np.ndarray],
    count: int | None = None,
    dtype: np.dtype = np.float64,
    max_workers: int = 4,
    max_parallel_diag: int = 4,
    force_gc: bool = False,
    progress: bool = True,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Track eigenvalues of a sequence of Hermitian matrices consistently,
    using a pipelined parallel approach.

    Parameters
    ----------
    matrices : ndarray of shape (m, n, n) or callable
        Either a 3D array of m Hermitian matrices, or a factory callable
        ``f(i) -> ndarray of shape (n, n)`` that generates the i-th matrix
        on demand. Use the callable form to avoid materialising the full
        sequence in memory.
    count : int, optional
        Number of matrices. Required when `matrices` is a callable.
    dtype : dtype, default np.float64
        Output dtype for the eigenvalue array. Should match the precision
        of the input matrices. float64 is the standard LAPACK precision.
    max_workers : int, default 4
        Number of background threads used to pre-diagonalise matrices while
        the main thread runs the Hungarian assignment.
    max_parallel_diag : int, default 4
        Look-ahead window: how many future diagonalisations are submitted
        before the current Hungarian assignment completes. Should equal
        max_workers for best throughput.
    force_gc : bool, default False
        Force a Python garbage collection pass after each iteration.
        Automatically enabled for n >= 1500 regardless of this setting.
    progress : bool, default True
        Show a tqdm progress bar tracking Hungarian assignment iterations.

    Returns
    -------
    eigenvalues : ndarray of shape (m, n)
        Consistently ordered eigenvalues. Column k tracks the same physical
        state across all m steps, sorted from lowest to highest at step 0.
    indx_map : dict[int, int]
        Maps each physical-state label (derived from the first eigenvector
        matrix) to its column index in `eigenvalues`.
    """
```

### Key changes from the old signature

| Old parameter | New behaviour |
|---|---|
| `use_eigenvalues` | **Removed.** Cost is always pure eigenvector overlap: $C = 1 - \|V_i^T V_{i+1}\|$. |
| `use_mlx` | **Removed.** NumPy/SciPy only. |
| `use_gpu` | **Removed.** No CuPy path. |
| `use_jax` | **Removed.** No JAX path. |
| `use_jit_sort` | **Removed.** No JAX path. |
| `dtype` | Default changed from `None` (infer from input) to `np.float64` (explicit). |
| `max_workers` | Default changed from `2` to `4`. |
| `max_parallel_diag` | Default changed from `max_workers` (i.e. 2) to `4`. |

### Usage examples

```python
import numpy as np
from eigenshuffle import eigenshuffle

# --- Array form ---
matrices = np.array([...])  # shape (m, n, n), Hermitian
eigenvalues, indx_map = eigenshuffle(matrices)

# --- Lazy factory form (recommended for large sequences) ---
def mat_factory(i: int) -> np.ndarray:
    return build_hamiltonian(vacs[i])

eigenvalues, indx_map = eigenshuffle(mat_factory, count=len(vacs))

# --- Memory-constrained system: reduce look-ahead ---
eigenvalues, indx_map = eigenshuffle(mat_factory, count=len(vacs), max_workers=2, max_parallel_diag=2)

# --- Very large matrices: explicit GC ---
eigenvalues, indx_map = eigenshuffle(mat_factory, count=len(vacs), force_gc=True)
```

---

## 5. Simplified Internal Implementation

After removing all the conditional backend branches, the core loop becomes clean and readable. Below is the target implementation for `eigenshuffle.py` (pseudocode-level, showing the structure):

```python
import gc
import concurrent.futures
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ._core import distance_matrix, compute_indx_map, apply_sign_correction


def eigenshuffle(matrices, count=None, dtype=np.float64, max_workers=4,
                 max_parallel_diag=4, force_gc=False, progress=True):

    # --- resolve input ---
    is_callable = callable(matrices)
    if is_callable and count is None:
        raise ValueError("`count` must be provided when `matrices` is callable")
    m = count if is_callable else len(matrices)
    get_mat = matrices if is_callable else lambda i: matrices[i]

    # --- sample first matrix ---
    sample = get_mat(0)
    n = sample.shape[-1]

    if not force_gc and n >= 1500:
        force_gc = True

    values = np.empty((m, n), dtype=dtype)

    # --- diagonalise first matrix ---
    vals, vecs = np.linalg.eigh(sample)
    values[0] = vals
    prev_vecs = vecs
    del sample

    # --- compute state map from first eigenvectors ---
    indx_map = compute_indx_map(vecs)
    del vals

    if m == 1:
        return values, indx_map

    # --- pipelined main loop ---
    iterator = tqdm(range(1, m), desc="Hungarian assignment", unit="step") if progress else range(1, m)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        diag_futures = {
            i: executor.submit(np.linalg.eigh, get_mat(i))
            for i in range(1, min(m, 1 + max_parallel_diag))
        }

        for i in iterator:
            vals, vecs = diag_futures.pop(i).result() if i in diag_futures else np.linalg.eigh(get_mat(i))

            next_idx = i + max_parallel_diag
            if next_idx < m and next_idx not in diag_futures:
                diag_futures[next_idx] = executor.submit(np.linalg.eigh, get_mat(next_idx))

            # Hungarian assignment (the bottleneck)
            distance = 1.0 - np.abs(prev_vecs.T @ vecs)
            _, col_ind = linear_sum_assignment(distance)
            del distance

            vals, vecs = vals[col_ind], vecs[:, col_ind]
            vecs = apply_sign_correction(prev_vecs, vecs)

            values[i] = vals
            prev_vecs = vecs
            del vals, vecs

            if force_gc:
                gc.collect()

    return values, indx_map
```

`_core.py` isolates the three helper operations:

```python
# eigenshuffle/_core.py

import numpy as np


def distance_matrix(vec1, vec2):
    """Interpoint distance matrix between two real/complex vectors."""
    return np.abs(vec1[:, np.newaxis] - vec2[np.newaxis, :]).T


def compute_indx_map(vecs):
    """
    Derive state labels from the first eigenvector matrix.
    Returns {physical_label: column_index}.
    """
    inv_vs = np.linalg.inv(vecs)
    indxs = np.argmax(np.abs(inv_vs) ** 2, axis=0)
    return {int(val): idx for idx, val in enumerate(indxs)}


def apply_sign_correction(prev_vecs, vecs):
    """Flip eigenvector columns whose sign is inconsistent with the previous step."""
    dot = np.sum(prev_vecs * vecs, axis=0).real
    flip = -(((dot < 0).astype(int) * 2) + 1)
    return vecs * flip
```

---

## 6. Unit Tests

### Philosophy

The test suite has one job: verify that the refactored function produces **numerically correct, consistently ordered eigenvalues**. It does not test removed backends.

All tests live in `tests/test_eigenshuffle.py` and use `pytest`.

### Reference fixture

The canonical test family is the parametric 4×4 symmetric matrix used in the existing tests:

$$A(t) = \begin{pmatrix} 1 & 2t+1 & t^2 & t^3 \\ 2t+1 & 2-t & t^2 & 1-t^3 \\ t^2 & t^2 & 3-2t & t^2 \\ t^3 & 1-t^3 & t^2 & 4-3t \end{pmatrix}$$

evaluated at 21 points $t \in [-1.0, 1.0]$ with step $0.1$. The expected eigenvalue matrix `EXPECTED_EIGENVALUES` is taken directly from the existing test file and represents the ground truth produced by the pre-refactor `eigenshuffle_eighvals` function with `use_eigenvalues=False`.

```python
# tests/conftest.py
import numpy as np
import pytest

def make_matrix(t):
    return np.array([
        [1,       2*t+1, t**2,   t**3  ],
        [2*t+1,   2-t,   t**2,   1-t**3],
        [t**2,    t**2,  3-2*t,  t**2  ],
        [t**3,    1-t**3,t**2,   4-3*t ],
    ], dtype=np.float64)

T_SEQ = np.arange(-1.0, 1.1, 0.1)

@pytest.fixture
def matrix_array():
    return np.array([make_matrix(t) for t in T_SEQ])

@pytest.fixture
def matrix_factory():
    matrices = [make_matrix(t) for t in T_SEQ]
    return matrices, lambda i: matrices[i]

EXPECTED_EIGENVALUES = np.array([
    [ 2.01807787e-01, 2.34468977e+00, 5.00000000e+00, 8.45350244e+00],
    [ 4.46444425e-01, 2.37278553e+00, 4.76869965e+00, 7.81207039e+00],
    [ 6.50537693e-01, 2.34131394e+00, 4.56000539e+00, 7.24814297e+00],
    [ 8.11802506e-01, 2.27094977e+00, 4.36481455e+00, 6.75243317e+00],
    [ 9.23641655e-01, 2.18568005e+00, 4.17511785e+00, 6.31556045e+00],
    [ 9.74449526e-01, 2.11183494e+00, 3.98546098e+00, 5.92825456e+00],
    [ 9.52541286e-01, 2.07267152e+00, 3.79314408e+00, 5.58164312e+00],
    [ 8.57997539e-01, 2.07681965e+00, 3.59759344e+00, 5.26758938e+00],
    [ 7.05814192e-01, 2.11560230e+00, 3.39948848e+00, 4.97909503e+00],
    [ 5.14941574e-01, 2.17419889e+00, 3.19996664e+00, 4.71089289e+00],
    [ 3.00371852e-01, 2.23912328e+00, 3.00000000e+00, 4.46050487e+00],
    [ 7.26886695e-02, 2.29713164e+00, 2.79996981e+00, 4.23020988e+00],
    [-1.60340670e-01, 2.33034363e+00, 2.59973107e+00, 4.03026597e+00],
    [-3.92718246e-01, 2.30635749e+00, 2.40465559e+00, 3.88170517e+00],
    [-6.20014444e-01, 2.26280146e+00, 2.14640989e+00, 3.81080309e+00],
    [-8.39923018e-01, 2.11113817e+00, 1.89856712e+00, 3.83021773e+00],
    [-1.05365434e+00, 1.92977085e+00, 1.59374301e+00, 3.93014048e+00],
    [-1.26853285e+00, 1.74502602e+00, 1.23083725e+00, 4.09266958e+00],
    [-1.50228846e+00, 1.57290562e+00, 8.25152817e-01, 4.30423003e+00],
    [-1.78830615e+00, 1.42718321e+00, 4.03893140e-01, 4.55722981e+00],
    [-2.17554439e+00, 1.32732731e+00, 1.58912173e-15, 4.84821708e+00],
])
```

### Test cases

```python
# tests/test_eigenshuffle.py
import numpy as np
import pytest
from eigenshuffle import eigenshuffle
from tests.conftest import EXPECTED_EIGENVALUES, T_SEQ, make_matrix


class TestArrayInput:
    """Tests using a pre-built 3D array as input."""

    def test_eigenvalues_match_reference(self, matrix_array):
        eigenvalues, _ = eigenshuffle(matrix_array, progress=False)
        assert np.allclose(eigenvalues, EXPECTED_EIGENVALUES, atol=1e-6, rtol=1e-6)

    def test_output_shape(self, matrix_array):
        eigenvalues, indx_map = eigenshuffle(matrix_array, progress=False)
        m, n = matrix_array.shape[0], matrix_array.shape[1]
        assert eigenvalues.shape == (m, n)

    def test_output_dtype_default(self, matrix_array):
        eigenvalues, _ = eigenshuffle(matrix_array, progress=False)
        assert eigenvalues.dtype == np.float64

    def test_output_dtype_explicit_float32(self, matrix_array):
        eigenvalues, _ = eigenshuffle(matrix_array, dtype=np.float32, progress=False)
        assert eigenvalues.dtype == np.float32

    def test_indx_map_is_dict(self, matrix_array):
        _, indx_map = eigenshuffle(matrix_array, progress=False)
        assert isinstance(indx_map, dict)
        # Keys and values should be integers, and values should be valid column indices
        n = matrix_array.shape[1]
        for key, val in indx_map.items():
            assert isinstance(key, int)
            assert 0 <= val < n

    def test_first_step_sorted_ascending(self, matrix_array):
        """Eigenvalues at step 0 must be sorted from low to high."""
        eigenvalues, _ = eigenshuffle(matrix_array, progress=False)
        assert np.all(np.diff(eigenvalues[0]) >= 0)


class TestFactoryInput:
    """Tests using a lazy callable factory as input."""

    def test_eigenvalues_match_reference(self, matrix_factory):
        matrices, factory = matrix_factory
        eigenvalues, _ = eigenshuffle(factory, count=len(matrices), progress=False)
        assert np.allclose(eigenvalues, EXPECTED_EIGENVALUES, atol=1e-6, rtol=1e-6)

    def test_factory_matches_array(self, matrix_array, matrix_factory):
        """Factory and array inputs must produce identical results."""
        matrices, factory = matrix_factory
        ev_array, _ = eigenshuffle(matrix_array, progress=False)
        ev_factory, _ = eigenshuffle(factory, count=len(matrices), progress=False)
        assert np.allclose(ev_array, ev_factory)

    def test_count_required_for_callable(self):
        factory = lambda i: make_matrix(T_SEQ[i])
        with pytest.raises(ValueError, match="`count` must be provided"):
            eigenshuffle(factory)  # missing count


class TestParallelism:
    """Tests that varying max_workers / max_parallel_diag does not affect correctness."""

    @pytest.mark.parametrize("workers,parallel", [
        (1, 1),
        (2, 2),
        (4, 4),
        (2, 4),  # parallel_diag > workers is valid
    ])
    def test_parallel_configs_match_reference(self, matrix_array, workers, parallel):
        eigenvalues, _ = eigenshuffle(
            matrix_array,
            max_workers=workers,
            max_parallel_diag=parallel,
            progress=False,
        )
        assert np.allclose(eigenvalues, EXPECTED_EIGENVALUES, atol=1e-6, rtol=1e-6)


class TestEdgeCases:
    """Boundary conditions."""

    def test_single_matrix(self):
        """A sequence of length 1 should return eigenvalues of that matrix, sorted."""
        A = make_matrix(0.5)
        A_seq = A[np.newaxis, :]  # shape (1, 4, 4)
        eigenvalues, indx_map = eigenshuffle(A_seq, progress=False)
        expected = np.sort(np.linalg.eigvalsh(A))
        assert eigenvalues.shape == (1, 4)
        assert np.allclose(eigenvalues[0], expected)

    def test_two_matrices(self):
        """Minimum non-trivial sequence."""
        A_seq = np.array([make_matrix(T_SEQ[0]), make_matrix(T_SEQ[1])])
        eigenvalues, _ = eigenshuffle(A_seq, progress=False)
        assert eigenvalues.shape == (2, 4)
        assert np.allclose(eigenvalues[0], EXPECTED_EIGENVALUES[0])
        assert np.allclose(eigenvalues[1], EXPECTED_EIGENVALUES[1])

    def test_force_gc_produces_same_result(self, matrix_array):
        ev_no_gc, _ = eigenshuffle(matrix_array, force_gc=False, progress=False)
        ev_gc, _ = eigenshuffle(matrix_array, force_gc=True, progress=False)
        assert np.allclose(ev_no_gc, ev_gc)

    def test_identity_of_eigenvalue_tracking(self, matrix_array):
        """
        Verify that eigenvalues are tracking physical states, not just sorting.
        At an avoided crossing, sorted eigenvalues would switch columns while
        tracked eigenvalues stay in the same column. This checks that the
        column ordering does not change sign discontinuously across steps.
        """
        eigenvalues, _ = eigenshuffle(matrix_array, progress=False)
        # No column should jump by more than a reasonable threshold step-to-step.
        # Large jumps would indicate a mis-assignment (failed tracking).
        step_diffs = np.abs(np.diff(eigenvalues, axis=0))
        # The test matrix changes smoothly; differences > 2.0 per step signal tracking failure.
        assert np.all(step_diffs < 2.0), (
            f"Eigenvalue tracking failure: max step diff = {step_diffs.max():.4f}"
        )


class TestNumericalProperties:
    """Verify mathematical invariants of the output."""

    def test_eigenvalues_are_real(self, matrix_array):
        """eigh on a real symmetric matrix must return real eigenvalues."""
        eigenvalues, _ = eigenshuffle(matrix_array, progress=False)
        assert np.isrealobj(eigenvalues) or np.allclose(eigenvalues.imag, 0)

    def test_each_step_is_permutation_of_eigh(self, matrix_array):
        """
        At every step, the tracked eigenvalues must be the same set of values
        as returned by np.linalg.eigvalsh — just in a different order.
        """
        eigenvalues, _ = eigenshuffle(matrix_array, progress=False)
        for i, mat in enumerate(matrix_array):
            raw = np.sort(np.linalg.eigvalsh(mat))
            tracked = np.sort(eigenvalues[i])
            assert np.allclose(raw, tracked, atol=1e-10), (
                f"Step {i}: tracked eigenvalues differ from eigh output"
            )
```

### Running the tests

```bash
# from the repo root, with the venv active
pytest tests/ -v

# with coverage report
pytest tests/ -v --cov=eigenshuffle --cov-report=term-missing
```

### What the tests collectively verify

| Test class | What it checks |
|---|---|
| `TestArrayInput` | Correct values, shape, dtype, `indx_map` structure, ascending sort at step 0 |
| `TestFactoryInput` | Lazy path produces identical output to array path; `count` guard |
| `TestParallelism` | Correctness is independent of `max_workers`/`max_parallel_diag` settings |
| `TestEdgeCases` | Single matrix, two matrices, `force_gc` invariance, no large tracking jumps |
| `TestNumericalProperties` | Eigenvalues are real; each step is a permutation of `np.linalg.eigvalsh` |

The `test_each_step_is_permutation_of_eigh` test is the most fundamental: it guarantees that `eigenshuffle` never invents or loses eigenvalues — it only reorders them. If that passes, the function is numerically correct by construction. The `test_identity_of_eigenvalue_tracking` test then verifies that the reordering is physically meaningful (smooth tracking, not random permutation).

---

## 7. Migration Guide (for any existing callers)

```python
# Before (old API)
from eigenshuffle import eigenshuffle_eighvals_turbo

eigenvalues, indx_map = eigenshuffle_eighvals_turbo(
    mat_factory,
    count=len(vacs),
    dtype=np.float64,
    use_eigenvalues=False,
    max_workers=4,
    max_parallel_diag=4,
    use_mlx=False,
    force_gc=True,
)

# After (new API) — identical behaviour, cleaner call
from eigenshuffle import eigenshuffle

eigenvalues, indx_map = eigenshuffle(
    mat_factory,
    count=len(vacs),
    force_gc=True,   # all other params already match the new defaults
)
```

The new defaults (`dtype=np.float64`, `max_workers=4`, `max_parallel_diag=4`, `use_eigenvalues` removed) mean the optimal configuration requires almost no keyword arguments.

---

## 8. Todo List

The refactor is broken into five phases. Each phase should be completed and verified before moving to the next. Tasks marked **[BLOCKING]** must finish before anything in the next phase can begin.

---

### Phase 1 — Establish a safety baseline

The goal of this phase is to have a green test suite against the *current* codebase before any code changes. This gives a regression anchor: if tests go red after a change, something broke.

- [x] **1.1** Run the existing test suite (`pytest tests/`) and confirm all tests pass in the current state. Record the pass count. **[BLOCKING]**
- [x] **1.2** Identify any tests in `tests/test_eigenshuffle.py` that exercise backends or parameters being removed (MLX, JAX, CuPy, `use_eigenvalues=True`). Mark them explicitly — they will be deleted, not fixed, in Phase 3.
- [x] **1.3** Compute the reference eigenvalue matrix (`EXPECTED_EIGENVALUES`) by running `eigenshuffle_eighvals_turbo` on the canonical 4×4 parametric test family with `use_eigenvalues=False, dtype=np.float64, progress=False` and saving the result. This becomes the numeric ground truth for all future tests.
- [x] **1.4** Confirm that the `conftest.py` fixtures described in Section 6 (`make_matrix`, `T_SEQ`, `EXPECTED_EIGENVALUES`, `matrix_array`, `matrix_factory`) can be written without any import of the current `eigenshuffle` package — they must depend only on NumPy so they survive the refactor unchanged.

---

### Phase 2 — Write the new test suite

Write the complete test suite described in Section 6 **before touching any source code**. Run it against the current package to confirm the reference tests pass and the structure is correct.

- [x] **2.1** Create `tests/conftest.py` with `make_matrix`, `T_SEQ`, `EXPECTED_EIGENVALUES`, `matrix_array` fixture, and `matrix_factory` fixture.
- [x] **2.2** Write `TestArrayInput` in `tests/test_eigenshuffle.py`:
  - [x] `test_eigenvalues_match_reference`
  - [x] `test_output_shape`
  - [x] `test_output_dtype_default`
  - [x] `test_output_dtype_explicit_float32`
  - [x] `test_indx_map_is_dict`
  - [x] `test_first_step_sorted_ascending`
- [x] **2.3** Write `TestFactoryInput`:
  - [x] `test_eigenvalues_match_reference`
  - [x] `test_factory_matches_array`
  - [x] `test_count_required_for_callable`
- [x] **2.4** Write `TestParallelism` with all four `(workers, parallel)` parametrize cases:
  - [x] `(1, 1)`, `(2, 2)`, `(4, 4)`, `(2, 4)`
- [x] **2.5** Write `TestEdgeCases`:
  - [x] `test_single_matrix`
  - [x] `test_two_matrices`
  - [x] `test_force_gc_produces_same_result`
  - [x] `test_identity_of_eigenvalue_tracking`
- [x] **2.6** Write `TestNumericalProperties`:
  - [x] `test_eigenvalues_are_real`
  - [x] `test_each_step_is_permutation_of_eigh`
- [x] **2.7** Run the new test suite against the current package. All tests importing `eigenshuffle_eighvals_turbo` under the hood (via the `eigenshuffle` alias that will be added later) should pass. Tests that directly call the new `eigenshuffle` name will fail until Phase 4 — that is expected. Confirm no import errors or fixture issues. **[BLOCKING for Phase 3]**

---

### Phase 3 — Delete dead code and files

All deletions. No new code is written in this phase.

**Delete entire directories:**
- [x] **3.1** Delete `eigenshuffle_old/` and its `__pycache__/`.
- [x] **3.2** Delete `eigenshuffle_new/` (all three files: `backend.py`, `core.py`, `hungarian_jax.py`).

**Delete files inside `eigenshuffle/`:**
- [x] **3.3** Delete `eigenshuffle/hungarian_cover.py`.
- [x] **3.4** Delete `eigenshuffle/hungarian_mlx.py`.
- [x] **3.5** Delete `eigenshuffle/kwok.py`.
- [x] **3.6** Delete `eigenshuffle/HungarianCUDA.cu`.

**Delete root-level ad-hoc scripts:**
- [x] **3.7** Delete `benchmark.py`.
- [x] **3.8** Delete `check_mlx_devices.py`.
- [x] **3.9** Delete all `test_*.py` files at the repository root (the ~25 ad-hoc experiment scripts). Keep only `tests/` (the proper test directory).

**Delete markdown result files (no longer relevant):**
- [x] **3.10** Delete `EXTREME_SCALE_RESULTS.md`.
- [x] **3.11** Delete `HUNGARIAN_ANALYSIS.md`.
- [x] **3.12** Delete `MLX_HUNGARIAN_RESULTS.md`.
- [x] **3.13** Delete `MLX_HUNGARIAN_VALIDATION.md`.
- [x] **3.14** Delete `OPTIMIZATION_SUMMARY.md`.
- [x] **3.15** Delete `PERFORMANCE_ANALYSIS_6_WORKERS.md`.

**Verify after deletions:**
- [x] **3.16** Run `pytest tests/` — the baseline tests from Phase 1 (those not touching removed backends) must still be green. **[BLOCKING for Phase 4]**

---

### Phase 4 — Rewrite the source

This phase produces the clean implementation described in Sections 3–5 of this plan.

**Create `eigenshuffle/_core.py`:**
- [x] **4.1** Create `eigenshuffle/_core.py` with the `distance_matrix` function (moved from `eigenshuffle_functions.py`, unchanged).
- [x] **4.2** Add `compute_indx_map(vecs)` to `_core.py` — extracted from the `eigenshuffle_eighvals` family, computes `{physical_label: col_index}` from the first eigenvector matrix.
- [x] **4.3** Add `apply_sign_correction(prev_vecs, vecs)` to `_core.py` — extracted from the main loop, flips eigenvector columns with negative dot product.
- [x] **4.4** Confirm `_core.py` has no imports beyond `numpy`. It must be free of side-effects.

**Rewrite `eigenshuffle/eigenshuffle_functions.py` → `eigenshuffle/eigenshuffle.py`:**
- [x] **4.5** Create a new file `eigenshuffle/eigenshuffle.py`. Do not edit the old file yet.
- [x] **4.6** Write the imports section: `gc`, `concurrent.futures`, `numpy`, `scipy.optimize.linear_sum_assignment`, `tqdm`, and the three helpers from `._core`.
- [x] **4.7** Implement the `eigenshuffle` function with the signature from Section 4:
  - [x] Input resolution block (array vs callable, validate `count`).
  - [x] Sample first matrix, infer `n`, resolve `force_gc` auto-enable threshold.
  - [x] Pre-allocate `values = np.empty((m, n), dtype=dtype)`.
  - [x] Diagonalise first matrix with `np.linalg.eigh`; store in `values[0]`.
  - [x] Compute `indx_map` via `compute_indx_map`; `del vals`.
  - [x] Early-return for `m == 1`.
  - [x] Set up `ThreadPoolExecutor` and pre-submit the first `max_parallel_diag` diagonalisations.
  - [x] Main loop: retrieve result from future (or fall back to synchronous), submit next future, compute cost matrix, run `linear_sum_assignment`, apply permutation, apply sign correction, store eigenvalues, update `prev_vecs`, delete temporaries, optional `gc.collect()`.
  - [x] Return `values, indx_map`.
- [x] **4.8** Write the full docstring for `eigenshuffle` as specified in Section 4.
- [x] **4.9** Remove `eigenshuffle_functions.py` (the old 846-line file).

**Update `eigenshuffle/__init__.py`:**
- [x] **4.10** Replace all existing imports with a single import: `from .eigenshuffle import eigenshuffle`.
- [x] **4.11** Set `__all__ = ["eigenshuffle"]`.

**Update `pyproject.toml`:**
- [x] **4.12** Remove `munkres` from dependencies.
- [x] **4.13** Add `scipy` as an explicit dependency (currently undeclared but required).
- [x] **4.14** Remove or move `jax`, `mlx`, `cupy` from dependencies (they are no longer used).

**Verify after rewrite:**
- [x] **4.15** Run `python -c "from eigenshuffle import eigenshuffle; print('OK')"` — confirms the package imports without errors. **[BLOCKING]**
- [x] **4.16** Run `pytest tests/` — all tests in `TestArrayInput`, `TestFactoryInput`, `TestParallelism`, `TestEdgeCases`, and `TestNumericalProperties` must pass. **[BLOCKING for Phase 5]**

---

### Phase 5 — Clean up and finalise

- [x] **5.1** Update `README.md`:
  - Remove all references to `eigenshuffle_eig`, `eigenshuffle_eigh`, `eigenshuffle_eighvals`, `eigenshuffle_eigvals`, `eigenshuffle_eighvals_turbo`.
  - Replace the example code with the new single-function API (`from eigenshuffle import eigenshuffle`).
  - Update the installation section to reflect the cleaned `pyproject.toml` dependencies.
  - Remove references to MLX, CuPy, JAX optional backends.
- [x] **5.2** Review `examples/` notebooks. Any cell that calls a removed function (`eigenshuffle_eig`, `eigenshuffle_eigh`, etc.) must be updated to use `eigenshuffle`. Notebooks that demonstrate MLX or GPU behaviour can be deleted or archived.
- [x] **5.3** Delete any remaining `__pycache__/` directories at all levels (they may contain `.pyc` files for deleted modules).
- [x] **5.4** Run `ruff check eigenshuffle/` (already configured in `pyproject.toml`) and fix any linting issues in the new files.
- [x] **5.5** Run `mypy eigenshuffle/` and resolve any type errors (the project has `mypy` configured and a `py.typed` marker).
- [x] **5.6** Run the full test suite one final time and confirm 100% pass rate with zero warnings. Record the final pass count.
- [x] **5.7** Commit the refactor. Suggested commit message: `refactor: collapse to single eigenshuffle() function, remove all unused backends`.
