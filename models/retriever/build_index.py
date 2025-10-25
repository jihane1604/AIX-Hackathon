"""
Build a FAISS index for a regulator's rule-pack articles.
- Reads config/regulators/<ns>.yaml
- Encodes each article text with the retriever model
- Saves FAISS index + mapping under indices/<ns>/

Run:
  python -m models.retriever.build_index --ns qcb
"""
import argparse, yaml, json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from models.config.defaults import ROOT_DIR, MODEL_OUT_DIR, DEVICE

def load_rulepack(ns: str):
    p = ROOT_DIR / "config" / "regulators" / f"{ns}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Rule-pack not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", required=True)
    args = ap.parse_args()
    ns = args.ns.lower()

    rp = load_rulepack(ns)
    arts = rp.get("articles", [])
    if not arts:
        raise SystemExit(f"No articles in rule-pack for ns={ns}")

    # load retriever
    model_dir = MODEL_OUT_DIR / "retriever"
    model = SentenceTransformer(str(model_dir), device=DEVICE)

    texts = [f"{a.get('title','')}\n\n{a.get('text','')}" for a in arts]
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    emb = emb.astype("float32")


    # build FAISS index
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors → use Inner Product
    index.add(emb)

    out_dir = ROOT_DIR / "indices" / ns
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "articles.index"))

    # store id mapping
    (out_dir / "mapping.json").write_text(json.dumps(arts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[index] ns={ns} dim={dim} added={emb.shape[0]} → {out_dir}")

if __name__ == "__main__":
    main()
