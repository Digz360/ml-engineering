import time
import torch
import torch.nn.functional as F

from src.minigpt.model import MiniGPT


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 32_000
BLOCK_SIZE = 1024
N_EMBD = 768
NUM_HEADS = 12
NUM_LAYERS = 12
DROPOUT = 0.1

DEVICE = "cuda"

# Vamos até ao limite da GPU.
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64]

# Mais do que uma medição para evitar conclusões
# baseadas num único step ruidoso.
WARMUP_STEPS = 3
BENCHMARK_STEPS = 10


# ============================================================
# GPU INFO
# ============================================================

assert torch.cuda.is_available(), "CUDA não está disponível."

gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

print("=" * 70)
print("GPU BENCHMARK")
print("=" * 70)

print(f"GPU: {gpu_name}")
print(f"VRAM total: {gpu_memory:.2f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")


# ============================================================
# MODEL
# ============================================================

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    block_size=BLOCK_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

model.train()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
)

num_params = sum(p.numel() for p in model.parameters())

print(f"Parâmetros: {num_params:,} ({num_params / 1e6:.2f}M)")
print(f"Context length: {BLOCK_SIZE}")
print(f"Warmup steps: {WARMUP_STEPS}")
print(f"Benchmark steps: {BENCHMARK_STEPS}")


# ============================================================
# BENCHMARK
# ============================================================

results = []

for batch_size in BATCH_SIZES:

    print()
    print("=" * 70)
    print(f"BATCH SIZE {batch_size}")
    print("=" * 70)

    try:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        x = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, BLOCK_SIZE),
            dtype=torch.long,
            device=DEVICE,
        )

        y = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, BLOCK_SIZE),
            dtype=torch.long,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # WARM-UP
        # ----------------------------------------------------

        for _ in range(WARMUP_STEPS):

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        # Não queremos que memória usada durante warm-up
        # contamine o peak medido.
        torch.cuda.reset_peak_memory_stats()

        # ----------------------------------------------------
        # REAL BENCHMARK
        # ----------------------------------------------------

        start = time.perf_counter()

        for _ in range(BENCHMARK_STEPS):

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        total_tokens = (
            batch_size
            * BLOCK_SIZE
            * BENCHMARK_STEPS
        )

        tokens_per_second = total_tokens / elapsed

        seconds_per_step = elapsed / BENCHMARK_STEPS

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(f"Tempo total:       {elapsed:.3f} s")
        print(f"Tempo / step:      {seconds_per_step:.4f} s")
        print(f"Tokens / segundo:  {tokens_per_second:,.0f}")
        print(f"Peak VRAM:         {peak_memory:.2f} GB")

        results.append(
            {
                "batch": batch_size,
                "tokens_per_second": tokens_per_second,
                "seconds_per_step": seconds_per_step,
                "peak_vram": peak_memory,
            }
        )

        del x, y, logits, loss

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    except torch.OutOfMemoryError:

        print("OOM ❌")

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        break


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("RESUMO")
print("=" * 70)

print(
    f"{'Batch':>8} "
    f"{'Tokens/s':>15} "
    f"{'s/step':>12} "
    f"{'VRAM GB':>12}"
)

for result in results:

    print(
        f"{result['batch']:>8} "
        f"{result['tokens_per_second']:>15,.0f} "
        f"{result['seconds_per_step']:>12.4f} "
        f"{result['peak_vram']:>12.2f}"
    )


if results:

    best = max(
        results,
        key=lambda x: x["tokens_per_second"],
    )

    print()
    print(
        f"Melhor throughput: batch {best['batch']} "
        f"→ {best['tokens_per_second']:,.0f} tokens/s"
    )