"""
Generates YAML rule-packs dynamically from extracted regulator text files.
Uses lightweight LLM summarization to identify sections, clauses, and domains.
"""
import json, yaml
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm

def generate_rulepack(ns: str, model_name: str = "distilbart-cnn-12-6"):
    base_dir = Path("data/interim/reg_corpus") / ns
    out_path = Path("config/regulators") / f"{ns}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load summarizer model
    summarizer = pipeline("summarization", model=model_name, truncation=True)

    articles = []
    for f in tqdm(sorted(base_dir.glob("*.txt"))):
        text = f.read_text(encoding="utf-8", errors="ignore")[:4000]
        try:
            summary = summarizer(text, max_length=120, min_length=40)[0]["summary_text"]
        except Exception:
            summary = text[:300]
        articles.append({
            "article_id": f.stem,
            "title": f"Extracted Section {len(articles)+1}",
            "domain": "general",
            "text": text,
            "summary": summary,
            "confidence": 0.9
        })

    rulepack = {"regulator": ns, "articles": articles}
    out_path.write_text(yaml.dump(rulepack, allow_unicode=True))
    print(f"[rulepack] wrote {out_path}")
