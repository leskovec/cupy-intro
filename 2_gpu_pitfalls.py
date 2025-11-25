# 2_gpu_pitfalls.py
# ----------------------------------------
# CuPy Fundamentals – Pitfalls & Safe Usage
# ----------------------------------------
import numpy as np
import cupy as cp
import time

print("=== COPYS LEFT and RIGHT ===")

# ---------------------------------------------------------------------------
# 1. CuPy vs NumPy array identity
# ---------------------------------------------------------------------------
print("Array identities: NumPy vs CuPy")
a_cpu = np.ones((3, 3))
a_gpu = 2.0*cp.ones((3, 3))

print("a_cpu is a NumPy array:",a_cpu)
print("a_gpu is a CuPy array:", a_gpu)

# ---------------------------------------------------------------------------
# Explicit transfers (safe practice)
# ---------------------------------------------------------------------------
print("Safe, explicit transfers")
b_gpu = cp.asarray(a_cpu)       # Host → Device
b_cpu = cp.asnumpy(a_gpu)       # Device → Host

print("After roundtrip transfer:")
print("a_cpu:\n", a_cpu)
print("b_cpu:\n", b_cpu)

print("a_gpu:\n", a_gpu)
print("b_gpu:\n", b_gpu)


# ---------------------------------------------------------------------------
# Mixing NumPy and CuPy safely
# ---------------------------------------------------------------------------
print("Mixing NumPy & CuPy safely")

x_gpu = cp.arange(5)
x_cpu = x_gpu.get()   # equivalent to cp.asnumpy()

# CPU-side computation is now safe:
print("Safe CPU dot:", np.dot(x_cpu, x_cpu))
print("Always convert manually before mixing libraries.\n")

# unsafe example:
# print("UNSAFE CPU dot:", np.dot(x_gpu, x_gpu))  # raises an error

# ---------------------------------------------------------------------------
# Implicit transfers — POTENTIAL PITFALL
# ---------------------------------------------------------------------------
print("Implicit data transfers: avoid these!\n")

# x is an array on the GPU
x = cp.array([1, 2, 3])

# x is on the GPU
print("Printing x forces a GPU->CPU transfer:")
print(x)  # BAD: sync + transfer

# x is still on the GPU!
print("Accessing a scalar forces a transfer + sync:")
print("float(x[0]) =", float(x[0]))

# GPU's are baad at boolean/branching - so they offload to CPU!
print("Boolean checks also transfer data:")
print("Does x have nonzero elements? ->", bool(x.any()))

# AVOID these patterns — they synchronize the GPU, kill performance, 
# and silently move data back to CPU.


# ---------------------------------------------------------------------------
# Fallback to NumPy — hidden CPU execution
# ---------------------------------------------------------------------------
print("Fallback to NumPy: operations that DO NOT exist in CuPy\n")
# Checking whether CuPy's einsum executes on GPU...

try:
    out = cp.einsum("i,i->", x, x)  # CuPy API call
    # Force actual GPU execution
    out.sum()  # triggers kernel
    cp.cuda.runtime.deviceSynchronize()
    print("cp.einsum executed on GPU.")
except Exception:
    print("cp.einsum fell back to NumPy or failed (CPU execution).")

# General rule: If a CuPy function is missing, a call may:
# - fail loudly (good)
# - OR fall back to NumPy internally (bad: silent CPU execution + transfer!)


# ---------------------------------------------------------------------------
# Dtype pitfalls
# ---------------------------------------------------------------------------
print("no strings in CuPy\n")

# Unsupported or problematic dtypes:
# - object, string → not supported
# - complex128 sometimes slower than float32
# - random generators differ from NumPy

print("example of a dtype mismatch error:")
try:
    bad = cp.array(["a", "b", "c"])   # Strings → unsupported
except Exception as e:
    print("Expected error:", e, "\n")

