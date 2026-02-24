# eigenshuffle – Deep Research Report

## Table of Contents
1. [What the Project Does](#1-what-the-project-does)
2. [Core Algorithm](#2-core-algorithm)
3. [Public API](#3-public-api)
4. [Implementation Architecture](#4-implementation-architecture)
5. [Hungarian Algorithm Implementations](#5-hungarian-algorithm-implementations)
6. [Backend Abstraction Layer (`eigenshuffle_new/`)](#6-backend-abstraction-layer-eigenshuffle_new)
7. [The Turbo Pipeline (`eigenshuffle_eighvals_turbo`)](#7-the-turbo-pipeline-eigenshuffle_eighvals_turbo)
8. [Hardware Acceleration](#8-hardware-acceleration)
9. [Performance Benchmarks & Results](#9-performance-benchmarks--results)
10. [Memory Management Strategy](#10-memory-management-strategy)
11. [Evolution: Old → Current → New](#11-evolution-old--current--new)
12. [Test Suite & Validation](#12-test-suite--validation)
13. [Dependencies](#13-dependencies)
14. [Known Limitations & Open Issues](#14-known-limitations--open-issues)
15. [Key Design Decisions & Quirks](#15-key-design-decisions--quirks)
16. [Optimal Production Call Pattern](#16-optimal-production-call-pattern)

---

## 1. What the Project Does

`eigenshuffle` solves the **eigenvalue tracking problem**: given a sequence of $m$ matrices $A_1, A_2, \ldots, A_m$ (each $n \times n$), compute eigenvalues and eigenvectors for every matrix such that the $k$-th eigenvalue at step $i+1$ corresponds to the *same physical state* as the $k$-th eigenvalue at step $i$.

Without this, `numpy.linalg.eigh` or `numpy.linalg.eig` return eigenvalues in an arbitrary order at each step. Eigenvalue curves can then appear to cross each other spuriously (an "avoided crossing" artifact) even when the underlying states are continuously evolving. `eigenshuffle` eliminates those crossings by consistently re-labelling eigenvalues as they evolve.

**Primary use case** (evident from examples and comments): quantum physics / molecular physics simulations where Hamiltonians vary as a function of some external parameter (e.g. field strength, lens position), and one needs to track which energy level is which across the parameter sweep.

---

## 2. Core Algorithm

The algorithm operates on a sequence of already-computed (eigenvalue, eigenvector) pairs and runs in two phases.

### Phase 1 – Initial sort

Each matrix's eigenpairs are sorted from lowest to highest eigenvalue (using real part when complex). This establishes a canonical starting order.

### Phase 2 – Sequential shuffle (the key step)

For consecutive pairs $(D_i, V_i)$ and $(D_{i+1}, V_{i+1})$, a **cost matrix** is built:

$$C = 1 - \left| V_i^T V_{i+1} \right|$$

This is an $n \times n$ matrix where entry $C_{jk}$ measures how dissimilar eigenvector $j$ from step $i$ is from eigenvector $k$ at step $i+1$. A value near 0 means the vectors are nearly identical; a value near 1 means they are orthogonal.

When `use_eigenvalues=True` (the default for `eigh`, optional for `eig`), the cost is extended:

$$C = C \cdot \sqrt{ \left(\Delta \text{Re}(\lambda)\right)^2 + \left(\Delta \text{Im}(\lambda)\right)^2 }$$

where $\Delta$ is the interpoint eigenvalue distance matrix. This pulls eigenvectors with similar eigenvalues closer together in the assignment.

The **optimal permutation** $\sigma$ is found by solving the **linear assignment problem** (LAP) on $C$:

$$\sigma^* = \arg\min_\sigma \sum_j C_{j,\sigma(j)}$$

This is solved with `scipy.optimize.linear_sum_assignment` (the default). The result reorders $(D_{i+1}, V_{i+1})$ so the states are consistently matched to $(D_i, V_i)$.

### Phase 3 – Sign correction

After reordering, the dot product $\sum_j (V_i)_j \cdot (V_{i+1})_j$ is checked. If any component is negative, the corresponding eigenvector column is flipped (multiplied by $-1$). This ensures eigenvector continuity across steps even though eigenvectors are only defined up to a global phase.

### State mapping (eigenvalues-only functions)

In the eigenvalues-only variants (`eigenshuffle_eighvals`, `eigenshuffle_eigvals`, `eigenshuffle_eighvals_turbo`), the eigenvectors of the *first* matrix are used to compute an **index map** `indx_map`. This is done by:

1. Computing $V_0^{-1}$ (matrix inverse of the first eigenvector matrix).
2. For each column $k$: finding the row $r$ with maximum $|V_0^{-1}|^2_{r,k}$.
3. Building `indx_map = {physical_state_label: column_index}`.

This map relates the initial ordering (sorted by eigenvalue magnitude) to the natural labelling of the physical states, which is useful for downstream analysis.

---

## 3. Public API

All public functions are exported from `eigenshuffle/__init__.py`.

| Function | Solver | Returns | Notes |
|---|---|---|---|
| `eigenshuffle_eigh(matrices, ...)` | `np.linalg.eigh` | `(eigenvalues, eigenvectors)` | Hermitian matrices |
| `eigenshuffle_eig(matrices, ...)` | `np.linalg.eig` | `(eigenvalues, eigenvectors)` | General matrices |
| `eigenshuffle_eighvals(matrices, ...)` | `np.linalg.eigh` | `(eigenvalues, indx_map)` | Hermitian, skip eigenvector storage |
| `eigenshuffle_eigvals(matrices, ...)` | `np.linalg.eig` | `(eigenvalues, indx_map)` | General, skip eigenvector storage |
| `eigenshuffle_eighvals_turbo(matrices, ...)` | `np.linalg.eigh` or MLX | `(eigenvalues, indx_map)` | Pipelined parallel version |

### Common parameters

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `matrices` | `ndarray` (m×n×n) or `Callable(i) → n×n` | required | The matrix sequence or lazy factory |
| `use_eigenvalues` | `bool` | `True` (`eigh`), `False` (`eig`) | Include eigenvalue distance in cost |
| `progress` | `bool` | `True` | Show `tqdm` progress bars |
| `dtype` | `np.dtype` or `None` | auto | Force output dtype |
| `use_gpu` | `bool` | `False` | Use CuPy (NVIDIA GPU) for diagonalization |
| `use_jax` | `bool` | `False` | Use JAX for diag + Hungarian |
| `count` | `int` or `None` | `None` | Number of matrices when `matrices` is callable |

### Turbo-specific parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_workers` | `2` | Thread pool size (conservative for memory) |
| `max_parallel_diag` | `max_workers` | Max simultaneous diagonalizations in flight |
| `use_mlx` | `False` | Use Apple MLX for diagonalization |
| `force_gc` | `False` | Force Python GC after each iteration; auto-enabled for n ≥ 1500 |

### Lazy factory interface

All functions accept a callable `matrices` argument instead of a full 3D array, together with a `count` argument. This avoids holding all $m \times n^2$ elements in memory at once (important for large sequences):

```python
def mat_factory(i: int) -> np.ndarray:
    return hamiltonian(param_sequence[i])

eigenvals, indx_map = eigenshuffle_eighvals(mat_factory, count=len(param_sequence))
```

---

## 4. Implementation Architecture

```
eigenshuffle/
├── __init__.py                    # Public exports
├── eigenshuffle_functions.py      # All current implementations (846 lines)
├── hungarian_cover.py             # JAX Hungarian (from Google Scenic, Apache 2.0)
├── hungarian_mlx.py               # MLX Hungarian (Apple Silicon hybrid)
├── kwok.py                        # Alternative BFS-based maximum matching
├── HungarianCUDA.cu               # CUDA Hungarian (unreferenced in Python)
└── py.typed                       # PEP 561 marker

eigenshuffle_new/                  # Refactored backend-abstracted version (WIP)
├── backend.py                     # ABC + concrete backends (NumPy/CuPy/JAX/MLX)
├── core.py                        # Core algorithm using backend abstraction
└── hungarian_jax.py               # JAX Hungarian (duplicate of hungarian_cover.py)

eigenshuffle_old/                  # Original version (uses munkres library)
├── __init__.py
└── eigenshuffle_functions.py      # Original munkres-based implementation
```

The **active** implementation is in `eigenshuffle/eigenshuffle_functions.py`. The `eigenshuffle_new/` directory is a partially-complete refactor with a cleaner backend abstraction (not yet wired into the package's `__init__.py`).

---

## 5. Hungarian Algorithm Implementations

The LAP (linear assignment problem) is the computational bottleneck for large matrices. Several solvers are available:

### 5.1 SciPy (default)
`scipy.optimize.linear_sum_assignment` — a highly optimized C++ implementation. Complexity $O(n^3)$. This is the ground truth reference.

### 5.2 JAX (`hungarian_cover.py` / `hungarian_jax.py`)
A pure JAX re-implementation of the standard **Successive Shortest Paths / potential-based** Hungarian algorithm (adapted from Google's Scenic project, Apache 2.0 licensed). Key characteristics:

- Uses `jax.lax.scan` for the outer row loop (unrolled as a scan to allow JIT compilation).
- Uses `jax.lax.while_loop` for the inner DFS augmenting path loop.
- `jax.jit`-compiled for speed after first call (JIT compilation overhead on first use).
- Supports `vmap` for batched Hungarian: `hungarian_tpu_matcher = jax.jit(jax.vmap(hungarian_single))`.
- Two versions exist: one in `hungarian_cover.py` (returns `[2, n]` stack with sorted row indices), one in `hungarian_jax.py` (duplicate code, returns `[2, n]` with row/col indices).
- There is a subtle difference: `hungarian_cover.py`'s `hungarian_single` returns indices in a `[parent, indices]` format where `parent` holds column assignments, while the version in `eigenshuffle_functions.py` (`hungarian_single_sorted`) explicitly ensures `row_indices = [0, 1, ..., n-1]` and derives `col_indices` from `parent`.

### 5.3 MLX (`hungarian_mlx.py`)
Four variants of increasing complexity:

| Function | Strategy | Speed |
|---|---|---|
| `hungarian_single_sorted_mlx` | Row/col reduction in MLX + greedy fallback + SciPy | Slow |
| `hungarian_single_sorted_mlx_optimized` | Row/col reduction in MLX + SciPy for assignment | ~1.22× SciPy at 2000×2000 |
| `hungarian_single_sorted_mlx_full` | Full DFS in NumPy (not MLX), preceded by MLX reduction | Medium |
| `hungarian_single_sorted_mlx_gpu` | Same as optimized but explicitly requests GPU device | Same as optimized |

The **recommended** MLX variant (`hungarian_single_sorted_mlx_optimized`) is a **hybrid**: uses MLX for the expensive row and column minimum operations (which benefit from Apple Silicon vectorization), then hands control to SciPy for the actual assignment logic. This correctly identifies that MLX's current `while_loop` limitations make a pure-MLX Hungarian slower than SciPy at small sizes.

The default export (`hungarian_single_sorted`) is aliased to `hungarian_single_sorted_mlx_optimized`.

### 5.4 Kwok (`kwok.py`)
An experimental Python implementation of the Kwok & Zamel (2025, arXiv:2502.20889) maximum weight matching algorithm. Uses BFS-style augmenting paths. It is **not currently used** by any function in the active codebase — it is included as a research artefact. The file has a bug: `@dataclass` and `from dataclasses import dataclass` are missing from the import block, so it would fail on import.

### 5.5 CUDA (`HungarianCUDA.cu`)
A raw CUDA C++ file. Not compiled or called from Python. Appears to be a placeholder or reference for a future GPU Hungarian implementation.

---

## 6. Backend Abstraction Layer (`eigenshuffle_new/`)

`eigenshuffle_new/backend.py` defines a clean abstract interface not present in the main package:

```
LinalgBackend (ABC)
├── NumPyLinalgBackend
├── CuPyLinalgBackend
├── JAXLinalgBackend
└── MLXLinalgBackend

AssignmentBackend (ABC)
├── SciPyAssignmentBackend
└── JAXAssignmentBackend
```

Factory functions `get_linalg_backend(name)` and `get_assignment_backend(name)` return the appropriate concrete class. The `core.py` file begins re-implementing the shuffle logic using this abstraction but is **incomplete** — the `_shuffle` function in `core.py` references `tqdm` and `linear_sum_assignment` that are not imported, and the file ends mid-class. This directory is not importable as-is.

---

## 7. The Turbo Pipeline (`eigenshuffle_eighvals_turbo`)

This is the most performance-engineered function. It overlaps matrix diagonalization with Hungarian assignment using `concurrent.futures.ThreadPoolExecutor`.

### Pipeline design

```
Iteration i:
  Thread pool:   [diagonalize(i+max_parallel_diag)]  ← pre-submitted
  Main thread:   distance_matrix + linear_sum_assignment(i)  ← bottleneck
  After assign:  apply permutation + sign correction
```

Key implementation details:
- `max_parallel_diag` controls the "look-ahead" window: how many future diagonalizations are pre-submitted before the current Hungarian assignment completes.
- Default `max_workers=2` is intentionally conservative to avoid memory explosion with large matrices.
- When `n >= 1500`, `force_gc=True` is auto-enabled, calling `gc.collect()` after every iteration.
- Memory is explicitly reclaimed after each step (`del distance`, `del vals`, `del vecs`, etc.).
- The progress bar (`tqdm`) tracks only the Hungarian assignment loop — not the diagonalization — because that was the user's explicit request (to see how long the bottleneck operation takes).

### Why threads, not processes?

NumPy's `eigh` releases the GIL during LAPACK calls (because LAPACK is C). Therefore, multiple threads can genuinely run NumPy diagonalizations in parallel without being serialized by the GIL.

---

## 8. Hardware Acceleration

### NVIDIA GPU (CuPy)
Triggered by `use_gpu=True` in `eigenshuffle_eigh`, `eigenshuffle_eig`, `eigenshuffle_eighvals`. Uses `cupy.linalg.eigh` / `cupy.linalg.eig` for diagonalization and `cupy.linalg.inv` for the state map. Result is copied back to NumPy for the Hungarian assignment (no GPU Hungarian is implemented in the active path).

### Apple Silicon (MLX)
Triggered by `use_mlx=True` in `eigenshuffle_eighvals_turbo` and `eigenshuffle_eigvals`. Uses `mlx.linalg.eigh` / `mlx.linalg.eig` on the CPU stream (`stream=mx.cpu`) — not GPU — due to MLX's current GPU linalg limitations. The MLX Hungarian uses Apple Silicon vectorized reductions.

### JAX (CPU or GPU)
Triggered by `use_jax=True` in `eigenshuffle_eighvals`. JAX can target GPU if `jax.devices('gpu')` is available, otherwise falls back to CPU. JIT-compiled after first invocation.

---

## 9. Performance Benchmarks & Results

All benchmarks were run on an Apple Silicon Mac (M-series chip), targeting the primary use case of 2000×2000 Hermitian matrices.

### Hungarian assignment scaling (HUNGARIAN_ANALYSIS.md)

| Matrix size | SciPy | MLX Optimized | Winner |
|---|---|---|---|
| 50×50 | 0.019 ms | 0.336 ms | SciPy (17.7×) |
| 200×200 | 0.814 ms | 0.858 ms | ~tie |
| **500×500** | 5.040 ms | **4.601 ms** | **MLX** |
| **2000×2000** | 124.4 ms | **101.8 ms** | **MLX (1.22×)** |
| **10000×10000** | 4.758 s | **3.704 s** | **MLX (1.28×)** |

MLX becomes competitive at ~500×500 and increasingly dominant at scale.

### Full turbo pipeline for 2000×2000 (PERFORMANCE_ANALYSIS_6_WORKERS.md)

| Config | Time | Speedup |
|---|---|---|
| 2 workers, NumPy | 75.6 s | 1.00× |
| 6 workers, NumPy | 65.4 s | 1.16× |
| 2 workers, MLX | 12.6 s | **6.00×** |
| 6 workers, MLX | 12.0 s | **6.30×** |

### Memory at 2000×2000

| Config | Memory increase |
|---|---|
| 2 workers, NumPy | −176 MB (likely due to GC) |
| 6 workers, NumPy | +962 MB |
| 2 workers, MLX | +66 MB |
| 6 workers, MLX | +218 MB |

NumPy with 6 workers is memory-hungry because multiple 2000×2000 complex128 matrices (~64 MB each) are held simultaneously. MLX appears to benefit from Apple Silicon's unified memory architecture.

### Bottleneck analysis

The Hungarian assignment is the **primary bottleneck**. Scaling from 2→6 workers with NumPy barely helps (1.16×) because the solver is a sequential operation that can't be parallelized across matrices — only overlapped with diagonalization. MLX's 6× speedup comes almost entirely from faster matrix diagonalization (MLX `eigh` vs NumPy `eigh` on Apple Silicon), not from faster Hungarian.

---

## 10. Memory Management Strategy

The main concern is large matrices (2000×2000 complex128 = ~64 MB each). The codebase has explicit memory management:

1. **Eigenvalues-only functions** (`eigenshuffle_eighvals`, `eigenshuffle_eighvals_turbo`) avoid ever materializing the full $m \times n \times n$ eigenvector array. Only the *previous* eigenvector matrix `prev_vecs` is kept in memory at any time.

2. **Explicit `del`** after each use: `del distance`, `del vals`, `del vecs`, `del dot`, `del flip`, `del dist_vals`.

3. **Auto-GC threshold**: `n >= 1500` triggers `force_gc=True` automatically, calling `gc.collect()` after every iteration.

4. **Lazy matrix factory**: Pass a callable instead of a 3D array so matrices are generated on demand and never all stored simultaneously.

5. **Conservative defaults**: `max_workers=2` to limit how many large arrays exist concurrently.

---

## 11. Evolution: Old → Current → New

### `eigenshuffle_old/` — original munkres-based version

- Uses the `munkres` Python library (a pure-Python Hungarian implementation).
- Simple sequential loop: diagonalize all → sort all → shuffle all.
- No progress bars, no GPU support, no lazy factory.
- Kept for reference and regression testing.

### `eigenshuffle/eigenshuffle_functions.py` — current active version

Key additions over old:
- Replaced `munkres` with `scipy.optimize.linear_sum_assignment` (significantly faster).
- Added JAX-based Hungarian alternative (`use_jax=True`).
- Added CuPy GPU diagonalization (`use_gpu=True`).
- Added lazy callable interface (`count=` parameter).
- Added eigenvalues-only functions (`eigenshuffle_eighvals`, `eigenshuffle_eigvals`) that skip eigenvector storage.
- Added `eigenshuffle_eighvals_turbo` with pipelined parallelism and MLX support.
- Added `tqdm` progress bars with separate phases (diagonalization vs Hungarian).
- Added `LineLoggingTQDM` subclass that writes each progress step as a log line (useful for non-interactive terminals).
- Added comprehensive memory management.

### `eigenshuffle_new/` — in-progress refactor

Introduces a proper OOP backend abstraction to replace the ad-hoc `if use_jax / elif use_gpu / else` branching. Not yet complete or integrated.

---

## 12. Test Suite & Validation

### `tests/test_eigenshuffle.py` (899 lines)

Canonical regression tests with hardcoded expected values for a parametric 4×4 test matrix family:

$$A(t) = \begin{pmatrix} 1 & 2t+1 & t^2 & t^3 \\ 2t+1 & 2-t & t^2 & 1-t^3 \\ t^2 & t^2 & 3-2t & t^2 \\ t^3 & 1-t^3 & t^2 & 4-3t \end{pmatrix}$$

tested at 21 points $t \in [-1, 1.1)$ with step $0.1$.

Tests:
- `test_eigenshuffle_hermitian` — `eigenshuffle_eigh` eigenvalues and eigenvectors.
- `test_eigenshuffle` — `eigenshuffle_eig` eigenvalues and eigenvectors.
- `test_eigenshuffle_eigenvalues_only` — `eigenshuffle_eighvals` eigenvalues only.
- (More tests follow in lines 500–899, likely covering turbo and additional edge cases.)

### Ad-hoc test scripts (root directory)

The repository contains ~25 test/benchmark scripts at the root, reflecting an iterative development and experimentation process:

| Script | Purpose |
|---|---|
| `test_turbo_features.py` | Tests `eigenshuffle_eighvals_turbo` with varying `max_parallel_diag` and MLX |
| `test_turbo_2000_features.py` | Same but for 2000×2000 matrices |
| `test_2000x2000.py` | Raw benchmark at target scale |
| `test_memory_optimization.py` | Memory profiling |
| `test_hungarian_comprehensive.py` | Cross-validates all Hungarian implementations |
| `test_hungarian_output_identity.py` | Checks MLX vs SciPy assignment equivalence |
| `test_mlx_gpu_hungarian.py` | MLX GPU path testing |
| `test_extreme_scale_hungarian.py` | 10,000×10,000 scale test |
| `test_pipelined.py` | Pipeline overlap verification |
| `test_force_gc_2000.py` | GC effectiveness measurement |
| `check_mlx_devices.py` | Utility to introspect available MLX devices |

### Numerical validation notes

- MLX and JAX Hungarian occasionally produce **different but equally optimal** assignments (different tie-breaking). This is expected behaviour for the LAP when the cost matrix has multiple optimal solutions. Validated across 125 test cases with 100% optimal cost match.
- Eigenvector signs are ambiguous (defined up to ±1); the sign correction step handles this, but tests compare `Vseq` to hardcoded expected values, meaning the sign correction must be deterministic.

---

## 13. Dependencies

### Required
| Package | Version | Role |
|---|---|---|
| `numpy` | `^2.2.5` | All linear algebra, array storage |
| `scipy` | (implied) | `linear_sum_assignment`, not in pyproject but imported |
| `tqdm` | `^4.67.1` | Progress bars |
| `munkres` | `^1.1.4` | Used only in `eigenshuffle_old/` |

### Optional / hardware-specific
| Package | Role |
|---|---|
| `jax` / `jaxlib` | JAX Hungarian + diagonalization |
| `mlx` | Apple Silicon acceleration |
| `cupy-cudaXX` | NVIDIA GPU diagonalization |

**Note**: `scipy` is not listed in `pyproject.toml` but is unconditionally imported at the top of `eigenshuffle_functions.py`. This is a packaging oversight — the package would fail to import without SciPy.

Similarly, `jax` is unconditionally imported at module level:
```python
import jax.numpy as jnp
import jax
```
This means `import eigenshuffle` will raise `ImportError` on systems without JAX installed, even if the user never uses the JAX backend. This is a significant correctness bug.

---

## 14. Known Limitations & Open Issues

### Hard import of JAX at module level
`eigenshuffle_functions.py` imports `jax` and `jax.numpy` unconditionally at the top of the file, making JAX a **hard dependency** despite being optional. Systems without JAX cannot use the package.

### `eigenshuffle_new/` is incomplete
`core.py` has unresolved `NameError`s (`tqdm` and `linear_sum_assignment` not imported) and is not exported from any `__init__.py`. It cannot be used as-is.

### `kwok.py` is broken
Missing `from dataclasses import dataclass` and `@dataclass` decorator. Raises `NameError` on import.

### `scipy` missing from `pyproject.toml`
`scipy.optimize.linear_sum_assignment` is imported unconditionally but `scipy` is not declared as a dependency.

### `HungarianCUDA.cu` is dead code
No Python bindings, no Makefile, no compilation step. Purely a reference file.

### Sign correction for complex eigenvectors
The sign correction `vecs *= flip` works for real eigenvectors where the sign is globally ambiguous. For complex Hermitian matrices, the global phase is $e^{i\theta}$, not just $\pm 1$. The current implementation only fixes real-part sign, which may not fully resolve phase ambiguity in highly complex eigenvectors.

### `use_eigenvalues=False` default for `eigenshuffle_eig`
The non-Hermitian `eig` function defaults to `use_eigenvalues=False`, while `eigh` defaults to `True`. This is intentional (eigenvalues of general matrices may be complex, making the eigenvalue distance less meaningful) but is a subtle gotcha for users switching between the two.

### `eigenshuffle_old/` not tested independently
The old implementation uses `munkres` which has a different API (`m.compute(distance)` returns list-of-tuples vs SciPy's array-pair). It is kept for reference but `tests/test_eigenshuffle.py` imports from the new `eigenshuffle` package, not the old one.

---

## 15. Key Design Decisions & Quirks

### Assignment via eigenvector overlap, not eigenvalue proximity alone
The primary signal for matching is **eigenvector overlap** ($1 - |V_i^T V_{i+1}|$), not eigenvalue proximity. This is the right choice for avoided crossings: two states may have very close eigenvalues and yet be entirely different states (the whole point of avoided crossings). The eigenvector overlap correctly tracks the identity of the state even as its eigenvalue crosses another.

### `eigh` vs `eig` distinction
The hermitian path (`eigh`) guarantees real eigenvalues and orthonormal eigenvectors, which simplifies the sign correction and distance computation. The general path (`eig`) may return complex eigenvalues and non-orthogonal right eigenvectors; both distance matrices (real and imaginary parts) are combined in quadrature.

### Why `1 - |overlap|` and not just `|overlap|`?
Minimizing `1 - |overlap|` is equivalent to maximizing `|overlap|`, which finds the permutation that most strongly aligns consecutive eigenvectors. The `1 -` transforms a maximization into a minimization so SciPy's `linear_sum_assignment` (a minimizer) applies directly.

### `indx_map` semantics
The returned `indx_map = {physical_label: column_index}` maps from the labelling inferred from the *first* eigenvector matrix (via inverse and argmax) to the column ordering used in the output array. This is domain-specific — in quantum physics, it helps identify which output column corresponds to which physical eigenstate of the zeroth Hamiltonian.

### `LineLoggingTQDM`
A custom `tqdm` subclass that writes each progress tick as a plain log line (via `tqdm.write`). Used in non-interactive contexts where a live progress bar would produce garbled output.

### Conservative `max_workers=2` default
The `turbo` function defaults to 2 workers despite tests showing 6 workers can be faster. The rationale is **memory safety**: at 2000×2000 complex128, each matrix is ~64 MB and the eigenvector matrix is ~32 GB — actually ~64 MB for the vector array. Having 6 workers pre-fetch 6 matrices simultaneously uses ~384 MB extra RAM just for that prefetch window. The default protects users on memory-constrained machines.

### Auto-GC for large matrices
The threshold `n >= 1500` for auto-enabling `force_gc=True` is hard-coded heuristic based on benchmarking. At this size, Python's reference-counted GC does not always reclaim memory promptly between iterations, leading to gradual memory growth that can cause OOM kills on longer sequences.

---

## 16. Optimal Production Call Pattern

The following invocation has been identified through empirical testing as the most performant configuration for this codebase:

```python
es_p_0_mjmax, indx_map_p_0_mjmax = eigenshuffle_eighvals_turbo(
    mat_factory,
    count=len(vacs),
    dtype=np.float64,
    use_eigenvalues=False,
    max_workers=4,
    max_parallel_diag=4,
    use_mlx=False,
    force_gc=True,
)
```

### Step-by-step execution trace

**1. Lazy factory path activated**

Because `mat_factory` is a callable, `is_callable = True`. The function never allocates a 3D `(m, n, n)` array for all matrices at once. Instead it calls `mat_factory(i)` on demand inside the thread pool. Combined with `force_gc=True`, peak memory is bounded to roughly `(max_parallel_diag + 1) × sizeof(one matrix)` at any moment, rather than the full sequence.

**2. `dtype=np.float64` — output array allocation**

The output eigenvalue array is pre-allocated as:
```python
values = np.empty((len(vacs), n), dtype=np.float64)
```
`float64` matches standard NumPy `eigh` precision exactly, so there is no implicit upcast or downcast anywhere in the pipeline. Had `dtype` been left as `None`, the default `out_dtype = in_dtype` path would be taken — which is the same result only if `mat_factory` already returns `float64` matrices. Specifying it explicitly is therefore safer and avoids a potential silent precision change.

**3. `use_eigenvalues=False` — cost matrix is eigenvector-overlap only**

The shuffle cost matrix becomes simply:
$$C = 1 - \left| V_i^T V_{i+1} \right|$$

The eigenvalue-distance term is **not** multiplied in. This is a deliberate choice with two consequences:

- **Speed**: The `distance_matrix` call and the element-wise multiplication are skipped each iteration, saving $O(n^2)$ work per step. For large $n$ (e.g. 2000), that is ~4 million floating-point operations avoided per step.
- **Correctness trade-off**: For tightly clustered eigenvalues (e.g. near-degenerate states), omitting the eigenvalue distance can occasionally allow a mis-assignment if two eigenvectors happen to have similar overlaps with both candidates. In practice, for the `vacs` Hamiltonians, the eigenvector overlap signal is already strong enough that the eigenvalue penalty is redundant — the testing confirmed this is the empirically optimal choice.

**4. `max_workers=4, max_parallel_diag=4` — pipeline look-ahead**

At the start of the main loop, the thread pool pre-submits the next 4 diagonalizations before the first Hungarian assignment begins:
```
Pre-submit: mat_factory(1), mat_factory(2), mat_factory(3), mat_factory(4)

Iteration 1:  [MAIN] Hungarian(0→1)   [POOL] diag(5)
Iteration 2:  [MAIN] Hungarian(1→2)   [POOL] diag(6)
...
```

The effect is that diagonalization of matrix $i + 4$ runs in a background thread while the main thread performs the Hungarian assignment for step $i$. Since NumPy's LAPACK routines release the GIL, the 4 worker threads can truly execute in parallel (bounded by available CPU cores).

Why 4 is optimal here:
- Fewer than 4 leaves CPU cores idle during the Hungarian solve.
- More than 4 increases peak memory (each in-flight matrix occupies RAM) without adding throughput, because the Hungarian assignment is the sequential bottleneck and cannot be made faster by more diagonalization parallelism.
- 4 strikes the right balance between look-ahead depth and memory pressure for the matrix sizes in `vacs`.

**5. `use_mlx=False` — pure NumPy/SciPy path**

With `use_mlx=False`, the function uses:
- `np.linalg.eigh` for diagonalization (releases GIL → genuinely parallel across workers).
- `scipy.optimize.linear_sum_assignment` for Hungarian assignment (single-threaded, C++).

Despite MLX showing a 6× speedup in the benchmarks above, the *empirically* best result here is `use_mlx=False`. The likely explanation is that `mat_factory` generates matrices that are already in a NumPy-native dtype/layout, so the MLX path incurs a conversion overhead (`np → mx.array → np`) that outweighs the speedup from vectorized MLX reductions for the specific matrix sizes in `vacs`. The NumPy/SciPy path has zero conversion cost.

**6. `force_gc=True` — explicit garbage collection**

After each iteration:
```python
del distance, dist_vals (if applicable), vals, vecs, dot, flip
gc.collect()
```
This is important when `mat_factory` creates matrices with any Python-side overhead (e.g. constructing sparse Hamiltonians or allocating intermediate arrays). Without it, CPython's cyclic GC may not reclaim intermediate arrays promptly, causing slow memory growth over a long `vacs` sequence.

Note: `force_gc=True` would also be auto-enabled anyway if `n >= 1500`, so for large Hamiltonians this flag is redundant but harmless. For smaller matrices it would not auto-enable, so setting it explicitly is the safe choice.

### What the outputs contain

| Variable | Type | Shape | Content |
|---|---|---|---|
| `es_p_0_mjmax` | `np.ndarray` | `(len(vacs), n)` | Eigenvalues, consistently tracked across the `vacs` sequence, sorted from lowest to highest at the first step and then shuffled to maintain identity |
| `indx_map_p_0_mjmax` | `dict` | `{int: int}` | Maps each physical-state label (derived from $\arg\max |V_0^{-1}|^2$) to its column index in `es_p_0_mjmax`. Used to look up which output column corresponds to a given initial eigenstate |

### Why this is the most performant configuration

| Parameter | Choice | Reason |
|---|---|---|
| `mat_factory` + `count` | Lazy generation | Never materialises the full `(m, n, n)` array; O(1) memory per matrix |
| `dtype=np.float64` | Explicit | Avoids silent type promotion; matches LAPACK native precision |
| `use_eigenvalues=False` | Skip eigenvalue term | Saves O(n²) work per step; eigenvector overlap alone is sufficient for this dataset |
| `max_workers=4` | 4 threads | Fills available CPU cores during diagonalization without oversubscribing |
| `max_parallel_diag=4` | Matches workers | 4-step look-ahead fully hides diagonalization latency behind Hungarian assignment |
| `use_mlx=False` | NumPy/SciPy | Avoids NumPy↔MLX conversion overhead that offsets MLX gains for this matrix size/dtype |
| `force_gc=True` | Explicit GC | Prevents memory creep across a long sequence; safe for all matrix sizes |
