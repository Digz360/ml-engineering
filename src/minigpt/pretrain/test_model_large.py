from model import MiniGPT

VOCAB_SIZE = 32_000
BLOCK_SIZE = 1024
N_EMBD = 768
NUM_HEADS = 12
NUM_LAYERS = 12
DROPOUT = 0.1

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    num_heads=NUM_HEADS,
    block_size=BLOCK_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

num_params = sum(p.numel() for p in model.parameters())

print(f"Parâmetros: {num_params:,}")
print(f"Parâmetros: {num_params / 1e6:.2f}M")