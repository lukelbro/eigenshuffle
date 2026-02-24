[![Python versions on PyPI](https://img.shields.io/pypi/pyversions/eigenshuffle.svg)](https://pypi.python.org/pypi/eigenshuffle/)
[![eigenshuffle version on PyPI](https://img.shields.io/pypi/v/eigenshuffle.svg "eigenshuffle on PyPI")](https://pypi.python.org/pypi/eigenshuffle/)

# eigenshuffle

Consistently track eigenvalues and eigenvectors across a sequence of Hermitian matrices, so each column represents the same physical state at every step.

Adapted from code by [bmachiel](https://github.com/bmachiel/python-nport/blob/master/nport/eigenshuffle.py), originally based on the MATLAB [eigenshuffle](http://www.mathworks.com/matlabcentral/fileexchange/22885).

## Installation

```
pip install eigenshuffle
```

or directly from GitHub:

```
pip install git+https://github.com/ograsdijk/eigenshuffle
```

## Usage

```python
import numpy as np
from eigenshuffle import eigenshuffle

def make_matrix(t):
    return np.array([
        [1,     2*t+1, t**2,   t**3  ],
        [2*t+1, 2-t,   t**2,   1-t**3],
        [t**2,  t**2,  3-2*t,  t**2  ],
        [t**3,  1-t**3, t**2,  4-3*t ],
    ], dtype=np.float64)

tseq = np.arange(-1, 1.1, 0.1)
Aseq = np.array([make_matrix(t) for t in tseq])

values, indx_map = eigenshuffle(Aseq)
# values: (m, n) array – row i holds eigenvalues for matrix i,
#         columns track the same state across all steps.
# indx_map: dict mapping physical-state label to column index.
```

### Lazy factory input

Pass a callable and a `count` to avoid materialising the full array in memory:

```python
def mat_factory(i):
    return build_hamiltonian(params[i]).astype(np.float64)

values, indx_map = eigenshuffle(
    mat_factory,
    count=len(params),
    dtype=np.float64,
    max_workers=4,
    max_parallel_diag=4,
)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `matrices` | — | `(m, n, n)` array, list of `(n, n)` arrays, or `callable(i) -> (n, n)` |
| `count` | `None` | Required when `matrices` is callable |
| `dtype` | `np.float64` | Output dtype |
| `progress` | `False` | Show tqdm progress bar |
| `max_workers` | `4` | Thread-pool size |
| `max_parallel_diag` | `4` | Pre-submitted diagonalisation futures |
| `force_gc` | `False` | Call `gc.collect()` each iteration (auto-enabled for n ≥ 1500) |

![consistently sorted eigenvalues](https://raw.githubusercontent.com/ograsdijk/eigenshuffle/main/images/sorted_vs_unsorted.png)
