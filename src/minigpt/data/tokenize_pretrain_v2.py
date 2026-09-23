from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer


TOKENIZER_FILE = Path("/workspace/llm/tokenizer_large.json")

TRAIN_TXT = Path("/workspace/llm/pretrain_v2_train.txt")
VAL_TXT = Path("/workspace/llm/pretrain_v2_val.txt")

TRAIN_BIN = Path("/workspace/llm/pretrain_v2_train.bin")
VAL_BIN = Path("/workspace/llm/pretrain_v2_val.bin")

# Quantas linhas enviamos de cada vez ao tokenizer.
BATCH_LINES = 1000


tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

vocab_size = tokenizer.get_vocab_size()

print(f"Vocab size: {vocab_size:,}")

if vocab_size > np.iinfo(np.uint16).max + 1:
    raise ValueError("Vocabulary does not fit in uint16.")


def tokenize_file(input_file: Path, output_file: Path):
    print(f"\nTokenizando: {input_file}")
    print(f"Destino:     {output_file}")

    start_time = time.time()

    total_tokens = 0
    total_chars = 0
    batch = []

    # wb = binary write
    with (
        input_file.open("r", encoding="utf-8") as source,
        output_file.open("wb") as destination,
    ):

        for line in source:
            batch.append(line)
            total_chars += len(line)

            if len(batch) >= BATCH_LINES:
                encodings = tokenizer.encode_batch(batch)

                for encoding in encodings:
                    ids = np.asarray(encoding.ids, dtype=np.uint16)

                    destination.write(ids.tobytes())

                    total_tokens += len(ids)

                batch.clear()

            if total_chars // 100_000_000 > (
                total_chars - len(line)
            ) // 100_000_000:

                elapsed = time.time() - start_time

                print(
                    f"Chars: {total_chars / 1e9:.2f}B | "
                    f"Tokens: {total_tokens / 1e6:.1f}M | "
                    f"Tempo: {elapsed / 60:.1f} min"
                )

        # Último batch, caso tenha menos de 1000 linhas.
        if batch:
            encodings = tokenizer.encode_batch(batch)

            for encoding in encodings:
                ids = np.asarray(encoding.ids, dtype=np.uint16)

                destination.write(ids.tobytes())

                total_tokens += len(ids)

    elapsed = time.time() - start_time

    print("\nConcluído.")
    print(f"Caracteres: {total_chars:,}")
    print(f"Tokens:     {total_tokens:,}")
    print(f"BIN:        {output_file.stat().st_size / 1024**3:.2f} GiB")
    print(f"Tempo:      {elapsed / 60:.2f} min")

    return total_tokens


train_tokens = tokenize_file(TRAIN_TXT, TRAIN_BIN)
val_tokens = tokenize_file(VAL_TXT, VAL_BIN)


print("\n--- TOKENIZAÇÃO V2 TERMINADA ---")

print(f"Train tokens: {train_tokens:,}")
print(f"Val tokens:   {val_tokens:,}")

print(f"\nTrain BIN: {TRAIN_BIN}")
print(f"Val BIN:   {VAL_BIN}")