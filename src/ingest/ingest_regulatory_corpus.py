"""
Extract text from regulator PDFs/DOCXs in the raw data folder 
and saves them as structured Arrow datasets.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from docx import Document
from datasets import Dataset
from tqdm.auto import tqdm

from models.config.defaults import ROOT_DIR

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def read_pdf_text(path: Path) -> str:
    """Extracts text from a PDF using PyMuPDF (fitz)."""
    doc = fitz.open(path)
    return "\n".join(p.get_text("text") for p in doc)

def read_docx_text(path: Path) -> str:
    """Extracts text from a DOCX using python-docx."""
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(path: Path) -> str:
    """Routes file reading based on extension."""
    if path.suffix.lower() == ".pdf":
        return read_pdf_text(path)
    if path.suffix.lower() == ".docx":
        return read_docx_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")

def ingest_regulatory_data(ns: str, root: Path = ROOT_DIR) -> Path:
    """
    Ingests raw regulatory files for a given namespace (ns), 
    extracts text, and saves structured data.
    """
    # Define source and output paths relative to the project root
    src = root / "data" / "regulatory corpus" / ns
    out_dir = root / "data" / "interim" / "reg_corpus" / ns
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rows: List[Dict] = []
    
    for p in tqdm(sorted(src.glob("*")), desc=f"ingest {ns}", dynamic_ncols=True):
        if not p.is_file() or p.suffix.lower() not in {".pdf", ".docx"}:
            continue
            
        b = p.read_bytes()
        sid = sha1_bytes(b)
        txt = extract_text(p)
        
        # Save extracted text for long-term reference
        txt_out_path = out_dir / f"{sid}.txt"
        txt_out_path.write_text(txt, encoding="utf-8")
        
        rows.append({
            "ns": ns,
            "path": str(p),
            "sha1": sid,
            "chars": len(txt),
            "txt_path": str(txt_out_path)
        })

    # Save final Arrow dataset for the index builder
    ds = Dataset.from_list(rows)
    dataset_path = root / "data" / "datasets" / f"regulator_{ns}"
    ds.save_to_disk(str(dataset_path))
    
    manifest = root / "data" / "interim" / f"manifest.regulatory.{ns}.json"
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    
    print(f"[ingest] Regulatory manifest and dataset saved for ns={ns}")
    return manifest

if __name__ == "__main__":
    # Example usage for your two regulators:
    ingest_regulatory_data("qcb")
    ingest_regulatory_data("qfc")