"""
Step 2 runner:
1) Enrich manifest with content-based fields.
2) Build/merge rule-packs from regulator documents into config/regulators/<ns>.yaml.
3) Chunk all interim texts into data/processed/chunks/<doc_id>.jsonl (for embeddings later).
"""
from __future__ import annotations
import json
from pathlib import Path

from app.enrich import enrich_manifest
from app.rulepack_from_docs import build_rulepacks_from_enriched
from app.chunker import split_into_chunks

def _chunk_all(root: Path) -> None:
    interim = root / "data" / "interim"
    processed = root / "data" / "processed" / "chunks"
    processed.mkdir(parents=True, exist_ok=True)

    enriched = json.loads((interim / "manifest.enriched.json").read_text(encoding="utf-8"))
    for rec in enriched:
        doc_id = rec["doc_id"]
        txt_path = interim / f"{doc_id}.txt"
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)  # defaults are fine; adjust later if needed
        out = processed / f"{doc_id}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps({"doc_id": doc_id, **ch}, ensure_ascii=False) + "\n")

def main():
    root = Path(__file__).resolve().parents[1]
    enriched_path = enrich_manifest(root)
    print(f"[enrich] wrote {enriched_path}")

    rp_paths = build_rulepacks_from_enriched(root)
    for p in rp_paths:
        print(f"[rulepack] updated {p}")

    _chunk_all(root)
    print("[chunk] wrote jsonl chunks under data/processed/chunks")

if __name__ == "__main__":
    main()
