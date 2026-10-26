from src.minigpt.model_loader import load_minigpt


CHECKPOINT = r"C:\Users\diogo\Documents\ml-engineering\models\minigpt_large_v2_sft_best.pt"
TOKENIZER = r"C:\Users\diogo\Documents\ml-engineering\models\tokenizer_large.json"


model, tokenizer, checkpoint, device = load_minigpt(
    checkpoint_path=CHECKPOINT,
    tokenizer_path=TOKENIZER,
)

print("Model loaded successfully!")
print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\nCheckpoint metadata:")
print(f"Step: {checkpoint['step']:,}")
print(f"Train loss: {checkpoint['train_loss']:.4f}")
print(f"Val loss: {checkpoint['val_loss']:.4f}")
print(f"Vocab size: {checkpoint['vocab_size']:,}")
print(f"Embedding: {checkpoint['n_embd']}")
print(f"Layers: {checkpoint['num_layers']}")
print(f"Heads: {checkpoint['num_heads']}")
print(f"Block size: {checkpoint['block_size']}")