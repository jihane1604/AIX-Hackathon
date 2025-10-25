"""
Fine-tune a bi-encoder retriever on (startup_chunk ↔ regulation_article) pairs.
For MVP, this script can also run zero-shot (no training) and just save the base model.

Inputs expected later: prepared pairs dataset. For now it saves backbone for indexing.
"""
from sentence_transformers import SentenceTransformer
from models.config.defaults import RETRIEVER_BACKBONE, MODEL_OUT_DIR, DEVICE

def main():
    model = SentenceTransformer(RETRIEVER_BACKBONE, device=DEVICE)
    out_dir = MODEL_OUT_DIR / "retriever"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"[retriever] saved base model to {out_dir}")

if __name__ == "__main__":
    main()
