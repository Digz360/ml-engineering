import json
import random
from collections import defaultdict

INPUT_PATH = "/workspace/llm/chat_large_train.jsonl"
OUTPUT_PATH = "/workspace/llm/chat_large_curated.jsonl"

SEED = 42
random.seed(SEED)

# Máximo de conversas que queremos de cada source.
# Sources pequenas que não aparecem aqui são mantidas integralmente.
LIMITS = {
    "amalia-llm/persona_general": 50_000,
    "amalia-llm/persona_nemotron": 40_000,
    "amalia-llm/wikipedia_conversations": 40_000,
    "amalia-llm/amalia-smol_summarize_pt": 25_000,
    "amalia-llm/persona_math": 10_000,
    "amalia-llm/amalia-smol-rewrite-PT": 20_000,
    "amalia-llm/PT-Culture_Data": 20_000,
    "amalia-llm/amalia-Nemotron-SFT-Instruction-Following-Chat-v2": 20_000,
}

samples = defaultdict(list)
counts = defaultdict(int)

print("A selecionar exemplos...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)

        source = example.get("source", "UNKNOWN")
        counts[source] += 1

        # Source sem limite -> guardamos tudo.
        if source not in LIMITS:
            samples[source].append(example)
            continue

        limit = LIMITS[source]
        reservoir = samples[source]
        count = counts[source]

        # Reservoir sampling para escolher exemplos
        # aleatoriamente ao longo da source inteira.
        if len(reservoir) < limit:
            reservoir.append(example)
        else:
            j = random.randrange(count)

            if j < limit:
                reservoir[j] = example


# Junta todas as sources.
dataset = []

for source_examples in samples.values():
    dataset.extend(source_examples)

# Mistura as sources.
random.shuffle(dataset)

print("\nDistribuição final:")
print("-" * 70)

total = 0

for source in sorted(samples):
    n = len(samples[source])
    total += n

    print(
        f"{source:<65} "
        f"{counts[source]:>8,} -> {n:>8,}"
    )

print("-" * 70)
print(f"TOTAL ORIGINAL: {sum(counts.values()):,}")
print(f"TOTAL CURATED:  {total:,}")


print("\nA escrever dataset...")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for example in dataset:
        f.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )

print(f"\nDataset guardado em:")
print(OUTPUT_PATH)