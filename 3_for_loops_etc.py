
import numpy as np
import cupy as cp
import time

print("=== FOR LOOPS ETC. ===")

# ---------------------------------------------------------------------------
# For-loops on GPU — guaranteed slow
# ---------------------------------------------------------------------------
print("Python for-loops on GPU are slow\n")

N = 20000
a_cpu = np.zeros(N, dtype=np.float64)
a_gpu = cp.zeros(N, dtype=cp.float64)

# CPU loop
t0 = time.time()
for i in range(N):
    a_cpu[i] += 1
t_cpu = time.time() - t0

# GPU loop: 
# each iteration in the for loop is a GPU kernel launch with one element
cp.cuda.runtime.deviceSynchronize()
t1 = time.time()
for i in range(N):
    a_gpu[i] += 1
cp.cuda.runtime.deviceSynchronize()
t_gpu = time.time() - t1

a_gpu_copied = cp.asnumpy(a_gpu)  # copy a_gpu to CPU for comparison
print("CPU and GPU results match:", np.allclose(a_cpu, a_gpu_copied))

print(f"CPU loop time: {t_cpu:.4f} s")
print(f"GPU loop time: {t_gpu:.4f} s")

# Explanation:
# GPU must launch 20k kernels, each doing 1 addition → catastrophic overhead
# LESSON: Vectorize instead


# Vectorized GPU operation
print("Vectorized GPU operation")
t2 = time.time()
a_gpu2 = cp.zeros(N, dtype=cp.float64)
# single kernel launch does all additions in parallel
a_gpu2 += 1
cp.cuda.runtime.deviceSynchronize()
print(f"GPU loop time: : { time.time() - t2:.4f} s")

a_gpu2_copied = cp.asnumpy(a_gpu2)  # copy a_gpu2 to CPU for comparison
print("CPU and vectorized GPU results match:", np.allclose(a_cpu, a_gpu2_copied))

#Lesson: GPU = massively parallel; scalar loops destroy performance


# ---------------------------------------------------------------------------
# Indexing & slicing: cheap on CPU, expensive on GPU
# ---------------------------------------------------------------------------
print("Indexing & slicing costs\n")

z = cp.zeros(10000000, dtype=cp.float64)  # large array on GPU

print("Accessing z[0] forces sync:", float(z[0]))
print("Slicing, e.g. z[1000:2000], creates a view → OK")
s = z[1000:1005]  # no transfer from GPU to CPU
print("Slice created on GPU without transfer")
s_cpu = cp.asnumpy(s)  # now copy to CPU
print("Slice copied to CPU:", s_cpu)

# ---------------------------------------------------------------------------
# In-place operations are super fast but must match dtype + shape!
# ---------------------------------------------------------------------------
print("In-place operations\n")

a = cp.ones(5, dtype=cp.float32)
b = cp.ones(5, dtype=cp.float64)

try:
    a += b   # fails: dtype mismatch (float64 vs float64)
except Exception as e:
    print("Expected error using in-place += with mismatched dtypes:")
    print(e, "\n")

a = cp.ones(5, dtype=cp.float32)
b = cp.ones(5, dtype=cp.float32)
try:
    a += b   # succeeds: same dtype
    print("In-place += succeeded with matching dtypes:", a)
except Exception as e:
    print("Unexpected error:", e)


# • Accessing scalars forces sync + transfer
# • Unsupported dtypes (string, object)
# • Python for-loops on GPU → extremely slow
# • In-place ops require same dtype
# • Hidden device-host transfers inside conditions or logging
# • Use cp.asarray / cp.asnumpy explicitly
# • Keep arrays on GPU as long as possible
