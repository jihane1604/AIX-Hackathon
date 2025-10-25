"""
Step 2b runner – model-driven rule-pack generation (no hardcoding).

Inputs:
- data/interim/manifest.enriched.json   (from Step 2 enrichment)
- data/interim/<doc_id>.txt             (regulator documents' plain text)

Process:
- Group regulator_rule docs by namespace.
- Use RulepackGenerator (zero-shot semantic) to classify segments and assemble rule-packs.
- Merge into config/regulators/<ns>.yaml (preserve manual edits, append new articles).

Run:
    python src/step2b_generate_rulepacks_model.py
"""
from __future__ import annotations
import json, yaml
from pathlib import Path
from typing import Dict, List

from app.model_rulepack import RulepackGenerator
from app.rulepack_merge import merge_rulepacks
from app.aliases import detect_regulator

def _load_yaml(p: Path) -> Dict:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _dump_yaml(p: Path, obj: Dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(obj, p.open("w", encoding="utf-8"), sort_keys=False, allow_unicode=True)

def main():
    root = Path(__file__).resolve().parents[1]
    interim = root / "data" / "interim"
    mf_enriched = interim / "manifest.enriched.json"
    if not mf_enriched.exists():
        raise FileNotFoundError("data/interim/manifest.enriched.json not found. Run Step 2 enrichment first.")

    manifest = json.loads(mf_enriched.read_text(encoding="utf-8"))

    grouped: Dict[str, List[str]] = {}
    for rec in manifest:
        if rec.get("doc_type_refined") != "regulator_rule":
            continue
        ns = rec.get("regulator_ns") or detect_regulator(rec.get("path", "")) or "unknown"
        txt_path = interim / f"{rec['doc_id']}.txt"
        text = txt_path.read_text(encoding="utf-8")
        grouped.setdefault(ns, []).append(text)
        print(f"[collect] ns={ns} doc_id={rec['doc_id']} chars={len(text)}")

    if not grouped:
        print("[collect] No regulator_rule documents detected. Check enrichment and filenames.")
        return

    gen = RulepackGenerator()

    for ns, texts in grouped.items():
        # quick visibility on segmentation count
        segs = []
        for t in texts:
            segs.extend(gen.segment(t))
        print(f"[segment] ns={ns} segments={len(segs)}")
        if not segs:
            print(f"[warn] ns={ns} produced 0 segments. Check heading patterns or extraction.")
        # classify and assemble rulepack via standard path
        rp_new = gen.build_rulepack(ns=ns, jurisdiction="", name=ns.upper(), doc_texts=texts)

        rp_path = root / "config" / "regulators" / f"{ns}.yaml"
        rp_old = _load_yaml(rp_path)
        print(f"[merge] ns={ns} existing_articles={len(rp_old.get('articles', []))} new_articles={len(rp_new.get('articles', []))}")
        rp_merged = merge_rulepacks(rp_old, rp_new)
        print(f"[write] ns={ns} merged_articles={len(rp_merged.get('articles', []))}")

        _dump_yaml(rp_path, rp_merged)
        print(f"[rulepack/model] updated {rp_path}")

if __name__ == "__main__":
    main()
