from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "src/data/llm/pretrain_v4_train.txt"
TOKENIZER_FILE = "src/data/llm/tokenizer_v4.json"

VOCAB_SIZE = 8000


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = ByteLevel(
    add_prefix_space=False
)

tokenizer.decoder = ByteLevelDecoder()


# ============================================================
# TRAINER
# ============================================================

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "[UNK]",
        "[PAD]",
        "[EOS]"
    ],
    initial_alphabet=ByteLevel.alphabet()
)


# ============================================================
# TREINO
# ============================================================

print("A treinar tokenizer...")

tokenizer.train(
    [TRAIN_FILE],
    trainer
)

tokenizer.save(TOKENIZER_FILE)

print(f"Tokenizer guardado em: {TOKENIZER_FILE}")


# ============================================================
# INFORMAÇÃO
# ============================================================

print(f"\nVocab size: {tokenizer.get_vocab_size()}")
print(f"UNK: {tokenizer.token_to_id('[UNK]')}")
print(f"PAD: {tokenizer.token_to_id('[PAD]')}")
print(f"EOS: {tokenizer.token_to_id('[EOS]')}")


# ============================================================
# TESTE
# ============================================================

test_text = (
    "Hoje estou a treinar o meu próprio modelo de linguagem "
    "com dados reais em português."
)

encoded = tokenizer.encode(test_text)

print("\n--- TESTE ---")
print(f"Original: {test_text}")
print(f"IDs: {encoded.ids}")
print(f"Tokens: {encoded.tokens}")
print(f"Decode: {tokenizer.decode(encoded.ids)}")