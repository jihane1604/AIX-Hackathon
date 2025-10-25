"""
Build Arrow datasets for training (doc_type, risk).
Reads input manifests (JSONL or JSON) from data/interim/startups/.
Saves Arrow splits under data/datasets/{train,val,test}.
"""
from pathlib import Path
import json, random
from datasets import Dataset
from tqdm.auto import tqdm

# --- FIX START: Define the ROOT_DIR reliably ---

# Find the project root by going up from the script's location (src/datasets)
ROOT_DIR = Path(__file__).resolve().parents[2] 
# This script is at 'C:/src/datasets/build_splits.py'. parents[2] points to 'C:/' (the project root).

# Define paths relative to the ROOT_DIR
INPUT_MANIFESTS_DIR = ROOT_DIR / "data" / "interim" / "startups"
OUTPUT_DATA_DIR = ROOT_DIR / "data" / "datasets"

# --- FIX END ---


def build_splits(src_dir: Path = INPUT_MANIFESTS_DIR):
    # Ensure the output directory exists before saving
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    # --- FIX: Use JSONL reading for memory efficiency (assuming you adopt this) ---
    # for p in tqdm(sorted(src_dir.glob("*.jsonl")), desc="reading manifests", dynamic_ncols=True):
    #     with p.open("r", encoding="utf-8") as f:
    #         for line in f:
    #             rows.append(json.loads(line))
    for p in tqdm(sorted(src_dir.glob("*.json")), desc="reading manifests", dynamic_ncols=True):
        rows += json.loads(p.read_text())

    # shuffle with a visible step count (not necessary but explicit)
    tqdm.write(f"[dataset] total rows before split: {len(rows)}")
    random.shuffle(rows)
    n = len(rows)
    train, val, test = rows[:int(0.8*n)], rows[int(0.8*n):int(0.9*n)], rows[int(0.9*n):]

    # --- FIX: Use Path objects for saving ---
    Dataset.from_list(train).save_to_disk(str(OUTPUT_DATA_DIR / "train"))
    Dataset.from_list(val).save_to_disk(str(OUTPUT_DATA_DIR / "val"))
    Dataset.from_list(test).save_to_disk(str(OUTPUT_DATA_DIR / "test"))

    print(f"[dataset] train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    # --- FIX: No need to pass an argument if the default is set globally ---
    build_splits()