import torch
import time

size = 10000

# CPU
A_cpu = torch.randn(size, size)
B_cpu = torch.randn(size, size)

start = time.time()
C_cpu = A_cpu @ B_cpu
cpu_time = time.time() - start

print(f"CPU: {cpu_time:.4f}s")


# GPU
A_gpu = A_cpu.to("cuda")
B_gpu = B_cpu.to("cuda")

# Pequeno warm-up
_ = A_gpu @ B_gpu
torch.cuda.synchronize()

start = time.time()

C_gpu = A_gpu @ B_gpu

torch.cuda.synchronize()
gpu_time = time.time() - start

print(f"GPU: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")

print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("Reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")