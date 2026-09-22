import json
import random
from collections import defaultdict

DATASET_PATH = "/workspace/llm/chat_large_train.jsonl"

SAMPLES_PER_SOURCE = 3
SEED = 42

random.seed(SEED)

samples_by_source = defaultdict(list)
counts_by_source = defaultdict(int)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)

        source = example.get("source", "UNKNOWN")
        counts_by_source[source] += 1

        count = counts_by_source[source]
        reservoir = samples_by_source[source]

        if len(reservoir) < SAMPLES_PER_SOURCE:
            reservoir.append(example)
        else:
            # Escolhe uma posição entre 0 e count-1
            j = random.randrange(count)

            # Probabilidade = SAMPLES_PER_SOURCE / count
            if j < SAMPLES_PER_SOURCE:
                reservoir[j] = example


print("\n" + "=" * 100)
print("AMOSTRAS DO DATASET DE CHAT")
print("=" * 100)

for source in sorted(samples_by_source):
    print(f"\n\n{'#' * 100}")
    print(f"SOURCE: {source}")
    print(f"TOTAL: {counts_by_source[source]:,}")
    print("#" * 100)

    for i, example in enumerate(samples_by_source[source], 1):
        print(f"\n--- EXEMPLO {i} ---")
        print(f"score: {example.get('score')}")

        conversation = example.get("conversation", [])

        for message in conversation:
            role = message.get("role", "?")
            content = message.get("content", "")

            print(f"\n[{role.upper()}]")
            print(content)

        print()