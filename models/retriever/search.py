"""
Runtime search against a regulator's FAISS index.
- Loads FAISS and article mapping.
- Encodes a query (document snippet) and returns top-k article hits.
"""
from typing import List, Dict
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from models.config.defaults import ROOT_DIR, MODEL_OUT_DIR, DEVICE

class RegulatorSearcher:
    def __init__(self, ns: str):
        self.ns = ns
        self.idx_dir = ROOT_DIR / "indices" / ns
        self.mapping = json.loads((self.idx_dir / "mapping.json").read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(self.idx_dir / "articles.index"))
        self.model = SentenceTransformer(str(MODEL_OUT_DIR / "retriever"), device=DEVICE)

    def search(self, text: str, k: int = 5) -> List[Dict]:
        q = self.model.encode([text], normalize_embeddings=True).astype("float32")
        scores, idx = self.index.search(q, k)
        out = []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0]), start=1):
            art = self.mapping[i]
            out.append({
                "rank": rank,
                "score": float(s),
                "article_id": art.get("article_id"),
                "title": art.get("title"),
                "domain": art.get("domain"),
                "confidence": art.get("confidence", None),
                "text": art.get("text")[:4000]
            })
        return out
