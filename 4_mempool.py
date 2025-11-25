import cupy
import numpy

print("=== CuPy Memory Pools: Device, Pinned, Limits, Custom Allocators ===")

# ----------------------------------------
# CuPy Memory Pools – Allocation, Limits, Async, Unified Memory
# ----------------------------------------



# ---------------------------------------------------------------------------
# Access the default memory pools
# ---------------------------------------------------------------------------
print("Accessing CuPy memory pools")

mempool = cupy.get_default_memory_pool()               # GPU device memory pool
pinned_mempool = cupy.get_default_pinned_memory_pool() # Pinned (page-locked CPU memory)

print("Initial device pool used bytes:", mempool.used_bytes())
print("Initial device pool total bytes:", mempool.total_bytes())
print("Initial pinned pool free blocks:", pinned_mempool.n_free_blocks())


# ---------------------------------------------------------------------------
# Allocate NumPy CPU array (not seen in CuPy pool)
# ---------------------------------------------------------------------------
print("Allocate NumPy array on CPU")

a_cpu = numpy.ndarray(10000, dtype=numpy.float64)
print("NumPy array size bytes:", a_cpu.nbytes)
print("Device pool:", mempool.used_bytes())

# device pool remains unchanged

# ---------------------------------------------------------------------------
# Transfer CPU array → GPU (CuPy allocates from device + pinned pools)
# ---------------------------------------------------------------------------
print("Transfer array host → device (trigger GPU + pinned pool usage)")

a = cupy.array(a_cpu)  # triggers device + pinned allocation

print("CuPy array size bytes:", a.nbytes)
print("Device pool used bytes:", mempool.used_bytes())
print("Device pool total bytes:", mempool.total_bytes())
print("Pinned pool free blocks:", pinned_mempool.n_free_blocks())


# ---------------------------------------------------------------------------
# When object goes out of scope, memory stays in pool for reuse
# ---------------------------------------------------------------------------
print("Deleting GPU array (memory returned to pool, not freed to system)")

a = None   # a is now deleted; it no longer holds the GPU memory

print("Device pool used bytes :", mempool.used_bytes())
print("Device pool total bytes:", mempool.total_bytes())
print("Pinned pool free blocks:", pinned_mempool.n_free_blocks())


# ---------------------------------------------------------------------------
# Clearing the memory pools explicitly
# ---------------------------------------------------------------------------
print("Clearing all memory pool blocks")

mempool.free_all_blocks()
pinned_mempool.free_all_blocks()

print("Device pool used bytes:", mempool.used_bytes())
print("Device pool total bytes:", mempool.total_bytes())
print("Pinned pool free blocks:", pinned_mempool.n_free_blocks())


# ---------------------------------------------------------------------------
# Hard-limiting GPU memory 
# ---------------------------------------------------------------------------
print("GPU memory limits via environment or API")

# Environment variable example (cannot set inside script):
#   export CUPY_GPU_MEMORY_LIMIT="1073741824"   # 1 GB
#   export CUPY_GPU_MEMORY_LIMIT="50%"          # alternative percentage form

print("Current memory limit:", cupy.get_default_memory_pool().get_limit(), "")

# Setting via API:
print("Setting explicit memory limits per device")
mempool = cupy.get_default_memory_pool()

with cupy.cuda.Device(0):
    mempool.set_limit(size=1024**3)      # 1 GB

print("Device new limit:", mempool.get_limit(), "")


