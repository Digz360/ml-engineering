from datasets import load_dataset
from pathlib import Path
import hashlib


OUTPUT_DIR = Path("/workspace/llm")

TRAIN_FILE = OUTPUT_DIR / "pretrain_v2_train.txt"
VAL_FILE = OUTPUT_DIR / "pretrain_v2_val.txt"


def is_good_article(article):
    title = article["title"].strip()
    text = article["text"].strip()

    if not title:
        return False

    if len(text) < 2500:
        return False

    if "(desambiguação)" in title.lower():
        return False

    return True


def is_validation(article_id):
    """
    Aproximadamente 1% dos artigos vão para validation.

    Usamos hash para o split ser determinístico:
    o mesmo artigo irá sempre para o mesmo split.
    """
    digest = hashlib.md5(str(article_id).encode()).hexdigest()

    number = int(digest[:8], 16)

    return number % 100 == 0


dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.pt",
    split="train",
    streaming=True,
)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0
accepted = 0
rejected = 0

train_articles = 0
val_articles = 0

train_chars = 0
val_chars = 0


with (
    TRAIN_FILE.open("w", encoding="utf-8") as train_file,
    VAL_FILE.open("w", encoding="utf-8") as val_file,
):

    for article in dataset:

        total += 1

        if not is_good_article(article):
            rejected += 1
            continue

        accepted += 1

        title = article["title"].strip()
        text = article["text"].strip()

        formatted = f"{title}\n\n{text}\n\n\n"

        if is_validation(article["id"]):
            val_file.write(formatted)

            val_articles += 1
            val_chars += len(formatted)

        else:
            train_file.write(formatted)

            train_articles += 1
            train_chars += len(formatted)

        if total % 100_000 == 0:
            print(
                f"Processados: {total:,} | "
                f"Aceites: {accepted:,} | "
                f"Train: {train_articles:,} | "
                f"Val: {val_articles:,}"
            )


print("\n--- DATASET V2 CONSTRUÍDO ---")

print(f"Total analisado: {total:,}")
print(f"Aceites: {accepted:,}")
print(f"Rejeitados: {rejected:,}")

print()

print(f"Train artigos: {train_articles:,}")
print(f"Train chars: {train_chars:,}")
print(f"Train: {train_chars / 1_000_000_000:.2f} GB")

print()

print(f"Val artigos: {val_articles:,}")
print(f"Val chars: {val_chars:,}")
print(f"Val: {val_chars / 1_000_000:.2f} MB")

print()

print("Train:", TRAIN_FILE)
print("Val:", VAL_FILE)