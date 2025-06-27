[![Python versions on PyPI](https://img.shields.io/pypi/pyversions/eigenshuffle.svg)](https://pypi.python.org/pypi/eigenshuffle/)
[![CeNTREX-TlF version on PyPI](https://img.shields.io/pypi/v/eigenshuffle.svg "eigenshuffle on PyPI")](https://pypi.python.org/pypi/eigenshuffle/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# eigenshuffle
Adapted from code by [bmachiel](https://github.com/bmachiel/python-nport/blob/master/nport/eigenshuffle.py), which in turn was based on matlab [eigenshuffle](http://www.mathworks.com/matlabcentral/fileexchange/22885).  

Consistently sort eigenvalues and eigenvectors of a series of matrices based on initial ordering from low to high.

Includes `eigenshuffle_eig` and `eigenshuffle_eigh` for non-hermitian and hermitian matrices, respectively.

# Installation
Install from pypi with:
```
pip install eigenshuffle
```

or clone repo and install with `pip` or directly install from GitHub with:  
```
pip install git+https://github.com/ograsdijk/eigenshuffle
pip install git+https://github.com/ograsdijk/eigenshuffle
```

# Example
```Python
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from eigenshuffle import eigenshuffle_eig

def eigenvalue_function(
    t: float,
) -> npt.NDArray[np.floating]:
    return np.array(
        [
            [1, 2 * t + 1, t**2, t**3],
            [2 * t + 1, 2 - t, t**2, 1 - t**3],
            [t**2, t**2, 3 - 2 * t, t**2],
            [t**3, 1 - t**3, t**2, 4 - 3 * t],
        ]
    )

tseq = np.arange(-1, 1.1, 0.1)
Aseq = np.array([eigenvalue_function(ti) for ti in tseq])

e, v = np.linalg.eig(Aseq)

es, vs = eigenshuffle_eig(Aseq)

# sorting original eig result from low to high
v[np.argsort(e)]
e = np.sort(e)

fig, ax = plt.subplots()
lines = ax.plot(tseq, e)

for i in range(ei.shape[-1]):
    ax.plot(tseq, ei.real[:, i], "--", color=lines[i].get_color())

# for generating the legend
line1 = plt.Line2D([0, 1], [0, 1], linestyle="-", color="black")
line2 = plt.Line2D([0, 1], [0, 1], linestyle="--", color="black")

ax.set_xlabel("t")
ax.set_ylabel("eigenvalue")
ax.legend([line1, line2], ["sorted", "eigenshuffle"])
ax.grid()

```
![consistenly sorted eigenvalues](https://raw.githubusercontent.com/ograsdijk/eigenshuffle/main/images/sorted_vs_unsorted.png)  
Here the eigenvalues are consistently ordered, and are not switching positions after a level crossing (around t=0.3) when using `eigenshuffle`.

## Lazy matrix generation

Both `eigenshuffle_eig` and `eigenshuffle_eigh` now accept a callable factory instead of a full 3D array, together with a `count` of matrices. This allows you to generate each Hamiltonian (or any matrix) on demand and keep peak memory at O(n²).

Example:
```python
# define your matrix builder taking an index -> 2D matrix
def mat_factory(i):
    v = vacs[i]
    return hamiltonian(v).astype(np.float32)

# call with factory and total count
es, vs = eigenshuffle_eig(
    mat_factory,
    count=len(vacs),       # number of matrices
    dtype=np.float32       # output data type
)
```
