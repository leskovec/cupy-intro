import cupy as cp
import numpy as np


# Generate Hermitian matrix on GPU
n = 2000  # matrix size

# random complex matrix on CPU and copy to GPU
T = np.random.randn(n, n) + 1j * np.random.randn(n, n)
A = cp.asarray(T)  # host -> device

# make it Hermitian: H = (A + A^dagger)/2
H = 0.5 * (A + A.conj().T)

# Compute eigenvalues on GPU
# cupy.linalg.eigh -> cuSolver Hermitian eigensolver
evals, evecs = cp.linalg.eigh(H)

# Transfer to CPU ONLY to print small summary
print("First 10 eigenvalues:")
print(cp.asnumpy(evals[:10]))

