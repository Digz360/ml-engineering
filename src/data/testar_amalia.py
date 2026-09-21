from datasets import load_dataset

ds = load_dataset(
    "inaciose/AMALIA-LLM-0626-SFT-Dataset-classlangcode",
    data_files="amalia_stf_pt_0000.parquet",
    split="train",
    streaming=True
)

for i, example in enumerate(ds):
    print(example)
    print("=" * 100)

    if i == 4:
        break