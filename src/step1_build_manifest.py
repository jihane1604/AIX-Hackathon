"""
Step 1 executable: scans `data/raw`, registers documents into a manifest,
extracts normalized text into `data/interim/<doc_id>.txt`.
No hard-coded filenames; metadata inferred from names only to seed doc_type.
"""

import os
import uuid
import json
from pathlib import Path
from typing import List

# Local imports via package path `app.*`
from app.config import load_config
from app.schemas import DocRecord
from app.utils.text_extraction import extract_text

import re
from typing import Optional, Dict

# Map regulator aliases → canonical namespace id used in your rule-packs
_REGULATOR_ALIASES = {
    "qcb": ["qcb", "qatar central bank"],
    "qfc": ["qfc", "qfcra", "qatar financial centre", "qatar financial center", "qf cra", "qf cra"],
    # add more regulators here without touching the rest of the code
}

# Precompile regexes once
_ALIAS_PATTERNS = {
    rid: [re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE) for alias in aliases]
    for rid, aliases in _REGULATOR_ALIASES.items()
}

def _detect_regulator_from_text(text: str) -> Optional[str]:
    """Best-effort detection from any text (filename or folder name)."""
    for rid, patterns in _ALIAS_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            return rid
    return None

def infer_doc_metadata(filename: str, rel_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """
    Heuristic, zero-config metadata inference.
    - doc_type seeded from filename keywords.
    - regulator_ns inferred from filename and parent folders (e.g., data/raw/qfc/...).
    """
    name = filename.lower()
    parent_hint = (str(rel_path.parent).lower() if rel_path else "")

    # --- doc_type seeding (filename-only, fast) ---
    if any(k in name for k in ["regulatory", "framework", "circular", "rule", "rulebook"]):
        doc_type = "regulator_rule"
    elif "business plan" in name or "business_plan" in name:
        doc_type = "startup_plan"
    elif "policy" in name or "internal compliance" in name:
        doc_type = "policy"
    elif "legal structure" in name or "articles of association" in name or "aoa" in name:
        doc_type = "legal"
    elif "resource mapping" in name or "resources" in name:
        doc_type = "resources"
    else:
        doc_type = "other"

    # --- regulator_ns from filename or folder names (qcb/qfc/qfcra/...) ---
    regulator_ns = (
        _detect_regulator_from_text(name)
        or _detect_regulator_from_text(parent_hint)  # e.g., data/raw/qfc/...
    )

    return {"doc_type": doc_type, "regulator_ns": regulator_ns}



def main() -> None:
    # Project root = .../regnav (script lives in regnav/src)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(str(root / "config" / "config.yaml"))

    # Prepare directories
    raw_dir = (root / cfg.raw_dir).resolve()
    interim_dir = (root / cfg.interim_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Expect the user’s provided docs already copied into data/raw/
    sources = [p for p in raw_dir.iterdir() if p.is_file()]

    manifest: List[DocRecord] = []
    for src in sources:
        # Generate short stable ID and register the source relative path
        uid = str(uuid.uuid4())[:8]
        meta = infer_doc_metadata(src.name)

        # Create a record; no copying/moving here to keep idempotency
        rec = DocRecord(
            doc_id=uid,
            path=str(src.relative_to(root)),
            doc_type=meta.get("doc_type", "other"),
            regulator_ns=meta.get("regulator_ns"),
            language=None,
            metadata={"original_name": src.name},
        )
        manifest.append(rec)

        # Extract normalized plain text into interim for downstream steps
        text, _ = extract_text(str(src))
        (interim_dir / f"{uid}.txt").write_text(text, encoding="utf-8")

    # Write manifest with typed fields — single source of truth for later stages
    mf_path = (interim_dir / "manifest.json")
    mf_path.write_text(
        json.dumps([m.model_dump() for m in manifest], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Manifest written: {mf_path}")
    print(f"Files registered: {len(manifest)}")


if __name__ == "__main__":
    main()
