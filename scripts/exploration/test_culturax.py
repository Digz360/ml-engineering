from datasets import load_dataset
from pathlib import Path
import hashlib
import random
import re


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("src/data/llm")

TRAIN_FILE = OUTPUT_DIR / "pretrain_v4_train.txt"
VAL_FILE = OUTPUT_DIR / "pretrain_v4_val.txt"

# Vamos começar por caracteres, porque ainda NÃO treinámos
# o tokenizer V4.
#
# ~4 caracteres/token é apenas uma aproximação.
TARGET_TOKENS = 20_000_000
APPROX_CHARS_PER_TOKEN = 4

TARGET_CHARS = TARGET_TOKENS * APPROX_CHARS_PER_TOKEN

VAL_RATIO = 0.01

MIN_CHARS = 300
MAX_CHARS = 20_000

SEED = 42


# ============================================================
# CLEANING
# ============================================================

def clean_text(text):
    """
    Limpeza leve.

    Não queremos 'reescrever' a Internet.
    Queremos apenas remover lixo óbvio.
    """

    if not text:
        return None

    text = text.replace("\x00", " ")

    # Normalizar whitespace horizontal
    text = re.sub(r"[ \t]+", " ", text)

    # Não permitir quantidades absurdas de linhas vazias
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text


# ============================================================
# QUALITY FILTER
# ============================================================

def is_good_document(text):
    if text is None:
        return False

    if len(text) < MIN_CHARS:
        return False

    # Documentos gigantes podem ser páginas cheias de lixo,
    # menus, dumps, etc.
    if len(text) > MAX_CHARS:
        return False

    words = text.split()

    if len(words) < 50:
        return False

    # Quantos caracteres são letras?
    letters = sum(c.isalpha() for c in text)

    if letters / len(text) < 0.60:
        return False

    # Evitar páginas com URLs por todo o lado
    url_count = text.lower().count("http")

    if url_count > 5:
        return False

    # Evitar textos extremamente repetitivos.
    # Olhamos para as primeiras palavras para manter isto barato.
    sample_words = words[:500]

    if len(sample_words) >= 100:
        unique_ratio = len(set(sample_words)) / len(sample_words)

        if unique_ratio < 0.20:
            return False

    return True


# ============================================================
# DEDUPLICATION
# ============================================================

def document_hash(text):
    """
    Hash do texto normalizado.

    Remove duplicados EXATOS.
    """

    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return hashlib.sha1(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("A carregar CulturaX PT em streaming...")

    dataset = load_dataset(
        "uonlp/CulturaX",
        "pt",
        split="train",
        streaming=True
    )

    seen_hashes = set()

    accepted_documents = 0
    rejected_documents = 0
    duplicate_documents = 0

    train_documents = 0
    val_documents = 0

    total_chars = 0

    with open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
    ) as train_file, open(
        VAL_FILE,
        "w",
        encoding="utf-8"
    ) as val_file:

        for example in dataset:

            text = clean_text(example.get("text"))

            # --------------------------
            # Quality
            # --------------------------

            if not is_good_document(text):
                rejected_documents += 1
                continue

            # --------------------------
            # Exact deduplication
            # --------------------------

            h = document_hash(text)

            if h in seen_hashes:
                duplicate_documents += 1
                continue

            seen_hashes.add(h)

            # --------------------------
            # Train / Validation split
            # --------------------------

            if random.random() < VAL_RATIO:

                val_file.write(text)
                val_file.write("\n\n")

                val_documents += 1

            else:

                train_file.write(text)
                train_file.write("\n\n")

                train_documents += 1

            # --------------------------
            # Statistics
            # --------------------------

            accepted_documents += 1
            total_chars += len(text)

            if accepted_documents % 1000 == 0:

                estimated_tokens = (
                    total_chars / APPROX_CHARS_PER_TOKEN
                )

                progress = (
                    total_chars / TARGET_CHARS
                ) * 100

                print(
                    f"aceites={accepted_documents:,} | "
                    f"rejeitados={rejected_documents:,} | "
                    f"duplicados={duplicate_documents:,} | "
                    f"chars={total_chars:,} | "
                    f"tokens≈{estimated_tokens:,.0f} | "
                    f"{progress:.1f}%"
                )

            # --------------------------
            # Finished
            # --------------------------

            if total_chars >= TARGET_CHARS:
                break

    estimated_tokens = total_chars / APPROX_CHARS_PER_TOKEN

    print()
    print("=" * 60)
    print("DATASET V4 TERMINADO")
    print("=" * 60)

    print(f"Documentos aceites:     {accepted_documents:,}")
    print(f"Documentos rejeitados:  {rejected_documents:,}")
    print(f"Duplicados removidos:    {duplicate_documents:,}")

    print()
    print(f"Train documents:         {train_documents:,}")
    print(f"Validation documents:    {val_documents:,}")

    print()
    print(f"Caracteres:              {total_chars:,}")
    print(f"Tokens aproximados:      {estimated_tokens:,.0f}")

    print()
    print(f"Train: {TRAIN_FILE}")
    print(f"Val:   {VAL_FILE}")


if __name__ == "__main__":
    main()