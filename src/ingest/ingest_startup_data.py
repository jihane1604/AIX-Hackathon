"""
Pulls labeled startup document data from the database and saves a structured 
JSONL manifest to data/interim/startups/.
"""
import json
from pathlib import Path
from typing import List, Dict
from tqdm.auto import tqdm

from models.config.defaults import ROOT_DIR

# --- Placeholder for Database Connection ---
def get_startup_data_from_db() -> List[Dict]:
    """
    Connects to the database and pulls rows needed for the startup document manifest.
    
    Replace this function with your actual database connection/query logic.
    Each returned item must include: raw_text, doc_id, regulator_ns, 
    doc_type_label, and risk_label.
    """
    print("[DB] Connecting and pulling startup document manifests...")
    # NOTE: This is mock data. Replace with your actual DB call.
    return [
        {
            "raw_text": "This is a sample business plan text...",
            "doc_id": "sha1:mock_bp_1",
            "regulator_ns": "qcb",
            "doc_type_label": "business_plan",
            "risk_label": "medium",
        },
        # Add more database rows here...
    ]
# -------------------------------------------

def ingest_startup_data(root: Path = ROOT_DIR) -> Path:
    """
    Pulls structured data from the database and creates the final JSONL manifest 
    in data/interim/startups/.
    """
    out_dir = root / "data" / "interim" / "startups"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "startup_manifest.jsonl"
    
    # 1. Pull data from the database
    db_rows = get_startup_data_from_db()
    
    final_rows: List[str] = []
    
    print(f"[ingest] Processing {len(db_rows)} startup documents...")
    
    # 2. Reformat data into the expected JSONL manifest schema
    for row in tqdm(db_rows, desc="building manifest", dynamic_ncols=True):
        manifest_row = {
            "doc_id": row["doc_id"],
            "regulator_ns": row["regulator_ns"],
            "text": row["raw_text"], 
            "targets": {
                "doc_type": row["doc_type_label"],
                "risk": row["risk_label"],
            },
            "meta": {
                "source": "database_pull",
            }
        }
        # Save as a JSON string with a newline (JSONL format)
        final_rows.append(json.dumps(manifest_row, ensure_ascii=False))

    # 3. Save the final JSONL manifest
    manifest_path.write_text("\n".join(final_rows), encoding="utf-8")
    
    print(f"[ingest] Startup manifest saved to {manifest_path}")
    return manifest_path

if __name__ == "__main__":
    ingest_startup_data()