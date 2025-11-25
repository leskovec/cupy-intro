#1_gpu_basics.py
# ----------------------------------------
# CuPy Fundamentals – How GPUs Work (Basics)
# ----------------------------------------
import numpy as np
import cupy as cp

# Terminology:
#   Host   = CPU + system RAM
#   Device = GPU + GPU memory

print("=== CPU vs GPU Basics ===")

# ---------------------------------------------------------------------------
# Create arrays on CPU (NumPy) and GPU (CuPy)
# ---------------------------------------------------------------------------

# NumPy array lives in host (CPU) memory
x_cpu = np.array([2.0, 4.0, 6.0, 8.0])
print("Array x on the CPU (NumPy):", x_cpu)
print("NumPy array type:", type(x_cpu))

# CuPy array lives in device (GPU) memory
y_gpu = cp.array([3.0, 5.0, 7.0, 9.0])
print("Array y on the GPU (CuPy):", y_gpu)
print("CuPy array type:", type(y_gpu))


# ---------------------------------------------------------------------------
# Compute norms on CPU and GPU
# ---------------------------------------------------------------------------

# Standard L2 norm computed on CPU
norm_cpu = np.linalg.norm(x_cpu)
print("Norm of x computed on the CPU:", norm_cpu)

# GPU norm: same API as NumPy, but runs on GPU
norm_gpu = cp.linalg.norm(y_gpu)
print("Norm of y computed on the GPU:", norm_gpu)


# ---------------------------------------------------------------------------
# Copy NumPy array → GPU (host → device)
# ---------------------------------------------------------------------------

# cp.asarray() moves a NumPy array into GPU memory
y_gpu_copied = cp.asarray(x_cpu)
print("Copied x_cpu to GPU as y':", y_gpu_copied)

# Compute its norm *on the GPU*
norm_gpu_copied = cp.linalg.norm(y_gpu_copied)
print("Norm of y' computed on the GPU:", norm_gpu_copied)


# ---------------------------------------------------------------------------
# Copy CuPy array → CPU (device → host)
# ---------------------------------------------------------------------------

# cp.asnumpy() retrieves GPU data back to CPU memory
x_cpu_copied = cp.asnumpy(y_gpu)
print("Copied y_gpu to CPU as x':", x_cpu_copied)

# Compute the norm *on the CPU*
norm_cpu_copied = np.linalg.norm(x_cpu_copied)
print("Norm of x' computed on the CPU:", norm_cpu_copied)

