"""
Extract text from regulator PDFs/DOCXs → data/interim/reg_corpus/<ns>/
Also builds an Arrow index for each regulator.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from docx import Document
from datasets import Dataset
from tqdm.auto import tqdm

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def read_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    return "\n".join(p.get_text("text") for p in doc)

def read_docx_text(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf_text(path)
    if path.suffix.lower() == ".docx":
        return read_docx_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")

def ingest_regulator(ns: str, root: Path = Path(".")) -> Path:
    src = root / "data" / "regulatory_corpus" / ns
    out_dir = root / "data" / "interim" / "reg_corpus" / ns
    out_dir.mkdir(parents=True, exist_ok=True)


# ...
    rows: List[Dict] = []
    for p in tqdm(sorted(src.glob("*")), desc=f"ingest {ns}", dynamic_ncols=True):
        if not p.is_file() or p.suffix.lower() not in {".pdf", ".docx"}:
            continue
        b = p.read_bytes()
        sid = sha1_bytes(b)
        txt = extract_text(p)
        (out_dir / f"{sid}.txt").write_text(txt, encoding="utf-8")
        rows.append({
            "ns": ns,
            "path": str(p),
            "sha1": sid,
            "chars": len(txt),
            "txt_path": str(out_dir / f"{sid}.txt")
        })

    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(root / "data" / "datasets" / f"regulator_{ns}"))
    manifest = root / "data" / "interim" / f"manifest.regulatory.{ns}.json"
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return manifest
