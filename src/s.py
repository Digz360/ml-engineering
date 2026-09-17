from pathlib import Path

filename = "666.jpg"

for split in ["train", "val", "test"]:
    path = Path("src/data") / split / "cat" / filename

    if path.exists():
        print("ENCONTRADA:", path)