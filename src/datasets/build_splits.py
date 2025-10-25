"""
Build Arrow datasets for training (doc_type, risk).
Reads input manifests (JSONL or JSON) from data/interim/startups/.
Saves Arrow splits under data/datasets/{train,val,test}.
"""
from pathlib import Path
import json, random
from datasets import Dataset
from tqdm.auto import tqdm


def build_splits(src_dir: Path = Path("data/interim/startups")):
    from tqdm.auto import tqdm
# ...
    rows = []
    for p in tqdm(sorted(src_dir.glob("*.json")), desc="reading manifests", dynamic_ncols=True):
        rows += json.loads(p.read_text())

    # shuffle with a visible step count (not necessary but explicit)
    tqdm.write(f"[dataset] total rows before split: {len(rows)}")
    random.shuffle(rows)
    n = len(rows)
    train, val, test = rows[:int(0.8*n)], rows[int(0.8*n):int(0.9*n)], rows[int(0.9*n):]

    Dataset.from_list(train).save_to_disk("data/datasets/train")
    Dataset.from_list(val).save_to_disk("data/datasets/val")
    Dataset.from_list(test).save_to_disk("data/datasets/test")

    print(f"[dataset] train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    build_splits()
