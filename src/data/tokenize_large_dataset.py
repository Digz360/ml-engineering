from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


TOKENIZER_FILE = "/workspace/llm/tokenizer_large.json"

TRAIN_TEXT = "/workspace/llm/pretrain_large_train.txt"
VAL_TEXT = "/workspace/llm/pretrain_large_val.txt"

TRAIN_BIN = "/workspace/llm/pretrain_large_train.bin"
VAL_BIN = "/workspace/llm/pretrain_large_val.bin"

CHUNK_SIZE = 10_000_000


tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

vocab_size = tokenizer.get_vocab_size()

print(f"Vocab size: {vocab_size:,}")

if vocab_size > 65_536:
    raise ValueError(
        "Vocab demasiado grande para uint16."
    )


def tokenize_file(input_file, output_file):

    total_tokens = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as source, open(
        output_file,
        "wb"
    ) as destination:

        while True:

            text = source.read(CHUNK_SIZE)

            if not text:
                break

            ids = tokenizer.encode(text).ids

            tokens = np.asarray(
                ids,
                dtype=np.uint16
            )

            tokens.tofile(destination)

            total_tokens += len(tokens)

            print(
                f"\rTokens escritos: {total_tokens:,}",
                end="",
                flush=True
            )

    print()

    size_gb = (
        Path(output_file).stat().st_size
        / 1024**3
    )

    print(f"Total tokens: {total_tokens:,}")
    print(f"Tamanho: {size_gb:.2f} GB")
    print(f"Guardado em: {output_file}")

    return total_tokens


print("\nA tokenizar TRAIN...")

train_tokens = tokenize_file(
    TRAIN_TEXT,
    TRAIN_BIN
)


print("\nA tokenizar VALIDATION...")

val_tokens = tokenize_file(
    VAL_TEXT,
    VAL_BIN
)


print()
print("=" * 60)
print("DATASET TOKENIZADO")
print("=" * 60)

print(f"Train tokens: {train_tokens:,}")
print(f"Val tokens:   {val_tokens:,}")
print(
    f"Total:        "
    f"{train_tokens + val_tokens:,}"
)