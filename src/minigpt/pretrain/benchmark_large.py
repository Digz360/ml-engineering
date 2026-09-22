import time
import torch
import torch.nn.functional as F

from model import MiniGPT


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 32_000
BLOCK_SIZE = 1024
N_EMBD = 768
NUM_HEADS = 12
NUM_LAYERS = 12
DROPOUT = 0.1

device = "cuda"

batch_sizes = [1, 2, 4, 8, 16, 32]


# ============================================================
# MODEL
# ============================================================

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    block_size=BLOCK_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

print(
    f"Parâmetros: "
    f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
)


# ============================================================
# BENCHMARK
# ============================================================

for batch_size in batch_sizes:

    print(f"\n--- Batch size {batch_size} ---")

    try:
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)

        x = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, BLOCK_SIZE),
            device=device
        )

        y = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, BLOCK_SIZE),
            device=device
        )

        torch.cuda.reset_peak_memory_stats()

        # warm-up
        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        # benchmark
        optimizer.zero_grad(set_to_none=True)

        start = time.perf_counter()

        logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        tokens = batch_size * BLOCK_SIZE
        tokens_per_second = tokens / elapsed

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(f"Tempo: {elapsed:.3f}s")
        print(f"Tokens/s: {tokens_per_second:,.0f}")
        print(f"Peak VRAM: {peak_memory:.2f} GB")

        del x, y, logits, loss

    except torch.OutOfMemoryError:

        print("OOM ❌")

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        break