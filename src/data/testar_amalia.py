from datasets import load_dataset
from collections import Counter

ds = load_dataset(
    "inaciose/AMALIA-LLM-0626-SFT-Dataset-classlangcode",
    data_files="amalia_stf_pt_0000.parquet",
    split="train",
    streaming=True
)

languages = Counter()
types = Counter()
sources = Counter()

total = 0

for example in ds:

    total += 1

    languages[example.get("language")] += 1
    types[example.get("code")] += 1
    sources[example.get("dataset")] += 1

    if total % 100_000 == 0:
        print(f"Processados: {total:,}")

print("\nTOTAL:", total)

print("\nLANGUAGES:")
for key, value in languages.most_common():
    print(key, value)

print("\nTIPOS:")
for key, value in types.most_common():
    print(key, value)

print("\nTOP DATASETS:")
for key, value in sources.most_common(30):
    print(key, value)