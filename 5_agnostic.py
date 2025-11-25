import numpy as np
import cupy as cp
import time

# cpu/gpu agnostic implementation of log(1 + exp(x))
def softplus(x):
    xp = cp.get_array_module(x)   
    # determines whether it is numpy (cpu) or cupy (gpu) and assigns it to xp
    print("Using:", xp.__name__)
    # then uses either np/cp to call the functions (the same functions exist in both!)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

# array of a lot of random numbers
x_cpu = np.random.rand(10000000)
x_gpu = cp.asarray(x_cpu)  

print("---- use the cpu ----")
t0 = time.time()
y_cpu = softplus(x_cpu)
t1 = time.time()
print("cpu:", y_cpu)
print("CPU time:", t1 - t0, "seconds\n")

print("---- use the gpu ----")
t0 = time.time()
y_gpu = softplus(x_gpu)
# IMPORTANT: synchronize for correct timing
cp.cuda.runtime.deviceSynchronize()
t1 = time.time()
print("gpu:", y_gpu)
print("GPU time:", t1 - t0, "seconds\n")

