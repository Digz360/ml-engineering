import json
import hashlib
import random
from pathlib import Path
from collections import Counter

from datasets import load_dataset


OUTPUT_DIR = Path("/workspace/llm")

TRAIN_FILE = OUTPUT_DIR / "chat_large_train.jsonl"
VAL_FILE = OUTPUT_DIR / "chat_large_val.jsonl"

VAL_RATIO = 0.01
MIN_SCORE = 0.90
MIN_CONTENT_CHARS = 20
MAX_CONVERSATION_CHARS = 12_000

SEED = 42
random.seed(SEED)


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    return text.strip()


def conversation_hash(messages):
    text = "\n".join(
        f"{message['role']}:{message['content']}"
        for message in messages
    )

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


def convert_messages(conversations):
    messages = []

    for message in conversations:

        role = message.get("role")
        content = normalize_text(message.get("content"))

        if not content:
            continue

        if role == "human":
            new_role = "user"

        elif role == "gpt":
            new_role = "assistant"

        elif role == "system":
            new_role = "system"

        else:
            continue

        messages.append({
            "role": new_role,
            "content": content
        })

    return messages


def valid_conversation(messages):

    if len(messages) < 2:
        return False

    # Tem de existir pelo menos uma mensagem do utilizador
    # e uma resposta do assistente.
    roles = {message["role"] for message in messages}

    if "user" not in roles or "assistant" not in roles:
        return False

    # Evitar exemplos praticamente vazios.
    for message in messages:
        if (
            message["role"] in {"user", "assistant"}
            and len(message["content"]) < MIN_CONTENT_CHARS
        ):
            return False

    total_chars = sum(
        len(message["content"])
        for message in messages
    )

    if total_chars > MAX_CONVERSATION_CHARS:
        return False

    return True


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "inaciose/AMALIA-LLM-0626-SFT-Dataset-classlangcode",
        data_files="amalia_stf_pt_0000.parquet",
        split="train",
        streaming=True
    )

    seen = set()

    stats = Counter()
    sources = Counter()

    with (
        open(TRAIN_FILE, "w", encoding="utf-8") as train_file,
        open(VAL_FILE, "w", encoding="utf-8") as val_file
    ):

        for example in dataset:

            stats["seen"] += 1

            # 1. Português
            if example.get("language") != "pt":
                stats["rejected_language"] += 1
                continue

            # 2. Quality score
            score = example.get("score")

            if score is None or score < MIN_SCORE:
                stats["rejected_score"] += 1
                continue

            # 3. Converter estrutura AMALIA
            conversations = example.get("conversations")

            if not conversations:
                stats["rejected_no_conversation"] += 1
                continue

            messages = convert_messages(conversations)

            if not valid_conversation(messages):
                stats["rejected_structure"] += 1
                continue

            # 4. Deduplicação exata
            fingerprint = conversation_hash(messages)

            if fingerprint in seen:
                stats["duplicates"] += 1
                continue

            seen.add(fingerprint)

            source = example.get("dataset", "unknown")
            sources[source] += 1

            record = {
                "messages": messages,
                "source": source,
                "score": score
            }

            line = json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"

            # 5. Train / validation
            if random.random() < VAL_RATIO:
                val_file.write(line)
                stats["validation"] += 1
            else:
                train_file.write(line)
                stats["train"] += 1

            stats["accepted"] += 1

            if stats["seen"] % 100_000 == 0:
                print(
                    f"Processados: {stats['seen']:,} | "
                    f"Aceites: {stats['accepted']:,}"
                )

    print("\n" + "=" * 70)
    print("FINAL")
    print("=" * 70)

    for key, value in stats.items():
        print(f"{key}: {value:,}")

    print("\nTOP SOURCES:")

    for source, count in sources.most_common(30):
        print(f"{count:>8,}  {source}")

    print("\nFicheiros:")
    print(TRAIN_FILE)
    print(VAL_FILE)


if __name__ == "__main__":
    main()