from datasets import load_dataset

ds = load_dataset(
    "inaciose/AMALIA-LLM-0626-SFT-Dataset-classlangcode",
    data_files="amalia_stf_pt_0000.parquet",
    split="train",
    streaming=True
)

found = 0

for example in ds:

    if example.get("language") != "pt":
        continue

    print("\n" + "=" * 100)
    print("DATASET:", example.get("dataset"))
    print("TIPO:", example.get("code"))
    print("SCORE:", example.get("score"))
    print()

    for message in example["conversations"]:
        print(
            f"{message['role'].upper()}: "
            f"{message['content']}"
        )
        print()

    found += 1

    if found == 20:
        break