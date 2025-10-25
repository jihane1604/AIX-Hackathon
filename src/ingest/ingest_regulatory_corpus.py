"""
Pulls regulatory documents from Supabase Storage based on metadata 
from the 'regulatory_documents' table, extracts text, and saves structured 
Arrow datasets for indexing.
"""
from __future__ import annotations
import hashlib, json
import os
import requests
from pathlib import Path
from typing import List, Dict, Any
from tqdm.auto import tqdm
from supabase import create_client, Client
from io import BytesIO
from docx import Document
import fitz # PyMuPDF
from datasets import Dataset

# --- Import ROOT_DIR from your configuration ---
from models.config.defaults import ROOT_DIR 

# --- Supabase Configuration (UPDATE THIS LINE!) ---
# Ensure BUCKET_NAME matches the name you set in Supabase (e.g., 'finovate-bucket')
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "https://annmfzwxyipqwdvnvfpt.supabase.co")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFubm1mend4eWlwcXdkdm52ZnB0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEzOTc2NjYsImV4cCI6MjA3Njk3MzY2Nn0.jzMNmMh-lkbCsKNrSIMUT2aur0NMG4JYRajTC8U8gug")
BUCKET_NAME: str = "Finovate%20Bucket" # MUST BE URL SAFE! Example: 'finovate-bucket'
TABLE_NAME: str = "regulatory_documents" # Your table name for regulations


# --- Helper Functions (Modified to use bytes, NOT file paths) ---
def read_pdf_text_from_bytes(content: bytes) -> str:
    """Extracts text from PDF bytes using PyMuPDF (fitz)."""
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(p.get_text("text") for p in doc)

def read_docx_text_from_bytes(content: bytes) -> str:
    """Extracts text from DOCX bytes using python-docx."""
    doc = Document(BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_bytes(file_path: str, content: bytes) -> str:
    """Routes text extraction based on file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf_text_from_bytes(content)
    if suffix == ".docx":
        return read_docx_text_from_bytes(content)
    # Default for text-based files
    return content.decode("utf-8", errors="ignore")

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# -------------------------------------------------------------
# CORE INGESTION FUNCTION
# -------------------------------------------------------------

def ingest_regulatory_data(root: Path = ROOT_DIR):
    """
    Pulls ALL regulatory data from Supabase, processes it, and saves structured 
    Arrow datasets grouped by regulator_ns.
    """
    
    # 🚨 CRITICAL PATHS (Must pre-exist in deployment)
    datasets_out_dir = root / "data" / "datasets"
    interim_out_dir = root / "data" / "interim" / "reg_corpus"
    
    # NOTE: REMOVE the following two lines for the FINAL deployment container
    datasets_out_dir.mkdir(parents=True, exist_ok=True) 
    interim_out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Initialize Supabase Client
        print("[DB] Connecting to Supabase for regulatory data...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 2. Query the metadata table for all regulatory documents
        # Must select 'document_path' (or link) and 'regulator_ns'
        response = supabase.table(TABLE_NAME).select("document_path, regulator_ns, id").order("id").execute()
        metadata_rows: List[Dict[str, Any]] = response.data
        
        if not metadata_rows:
            print("[DB] Warning: No rows found in the regulatory_documents table.")
            return

        print(f"[DB] Found {len(metadata_rows)} regulatory documents across all namespaces.")
        
        # Group documents by regulator namespace (ns)
        ns_map: Dict[str, List[Dict]] = {}
        for row in metadata_rows:
            # Use 'regulator_ns' as the grouping key
            ns = row.get("regulator_ns", "unknown").lower()
            if ns and row.get("document_path"):
                ns_map.setdefault(ns, []).append(row)

        
        # 3. Process documents for each regulator (ns)
        for ns, rows in ns_map.items():
            print(f"\n[ingest] Processing {len(rows)} documents for regulator: {ns}")
            
            # Define output path for text files for this specific NS
            ns_out_dir = interim_out_dir / ns
            # NOTE: REMOVE this line for the FINAL deployment container
            ns_out_dir.mkdir(parents=True, exist_ok=True) 
            
            rows_for_dataset: List[Dict] = []
            
            for row in tqdm(rows, desc=f"download & extract {ns}", dynamic_ncols=True):
                file_link = row.get("document_path") 

                # --- A. Download the file content from the storage bucket ---
                download_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_link}"
                response = requests.get(download_url, headers={"Authorization": f"Bearer {SUPABASE_KEY}"})

                if response.status_code != 200:
                    tqdm.write(f"Error downloading {file_link}. Status: {response.status_code}")
                    continue

                file_content = response.content
                
                # --- B. Extract Text & Hash ---
                raw_text = extract_text_from_bytes(file_link, file_content)
                doc_hash = sha1_bytes(file_content)
                
                # Save extracted text for long-term reference
                txt_out_path = ns_out_dir / f"{doc_hash}.txt"
                txt_out_path.write_text(raw_text, encoding="utf-8")
                
                # --- C. Build Dataset Row ---
                rows_for_dataset.append({
                    "ns": ns,
                    "path": file_link,
                    "sha1": doc_hash,
                    "chars": len(raw_text),
                    "txt_path": str(txt_out_path)
                })

            # 4. Save final artifacts for the current regulator (ns)
            ds = Dataset.from_list(rows_for_dataset)
            dataset_path = datasets_out_dir / f"regulator_{ns}"
            ds.save_to_disk(str(dataset_path))
            
            manifest = interim_out_dir / f"manifest.regulatory.{ns}.json"
            manifest.write_text(json.dumps(rows_for_dataset, indent=2, ensure_ascii=False))
            
            print(f"[ingest] Regulatory data and dataset saved for ns={ns}")
        
    except Exception as e:
        print(f"[ERROR] Supabase Regulatory Ingestion Failed: {e}")
        # Return on failure
        return

if __name__ == "__main__":
    # Runs ingestion for all documents found in the regulatory_documents table
    ingest_regulatory_data()