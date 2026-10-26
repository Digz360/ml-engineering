from pathlib import Path
import json

DATA_DIR = Path("data/minigpt")
REPORT_PATH = Path("reports/data_validation.json")

EXPECTED_FILES = [
    "pretrain_large_train.bin",
    "pretrain_large_val.bin",
    "pretrain_v2_train.bin",
    "pretrain_v2_val.bin",
    "chat_large_curated.jsonl",
    "chat_large_val.jsonl",
]

report = {}

for filename in EXPECTED_FILES:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    size_bytes = path.stat().st_size

    report[filename] = {
        "exists": True,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / 1024**2, 2),
    }

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("MiniGPT datasets validated successfully.")
print(f"Report saved to {REPORT_PATH}")