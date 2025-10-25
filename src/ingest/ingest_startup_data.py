"""
Pulls labeled startup document data from Supabase (metadata from table, 
raw text from storage bucket) and saves a structured JSONL manifest.
"""
import json
import os
import requests
from pathlib import Path
from typing import List, Dict, Any
from tqdm.auto import tqdm
from supabase import create_client, Client
from io import BytesIO
from docx import Document
import fitz # PyMuPDF (assuming you still use this for PDF/DocX)
import hashlib

# --- Import ROOT_DIR from your configuration ---
from models.config.defaults import ROOT_DIR 

# --- Supabase Configuration ---
# NOTE: In a production environment, NEVER hardcode secrets like this. 
# Use environment variables (e.g., os.environ['SUPABASE_URL']).
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "https://annmfzwxyipqwdvnvfpt.supabase.co")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFubm1mend4eWlwcXdkdm52ZnB0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjEzOTc2NjYsImV4cCI6MjA3Njk3MzY2Nn0.jzMNmMh-lkbCsKNrSIMUT2aur0NMG4JYRajTC8U8gug")
BUCKET_NAME: str = "Finovate%20Bucket" # Replace with your actual bucket name
TABLE_NAME: str = "startup_training" # Replace with your actual table name


# --- Helper Functions (copied/modified from ingest_regulatory_data.py) ---
def read_pdf_text_from_bytes(content: bytes) -> str:
    """Extracts text from PDF bytes."""
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(p.get_text("text") for p in doc)

def read_docx_text_from_bytes(content: bytes) -> str:
    """Extracts text from DOCX bytes."""
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

def ingest_startup_data(root: Path = ROOT_DIR) -> Path:
    """
    Pulls structured data from Supabase and creates the final JSONL manifest 
    in data/interim/startups/.
    """
    out_dir = root / "data" / "interim" / "startups"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "startup_manifest.jsonl"
    
    try:
        # 1. Initialize Supabase Client
        print("[DB] Connecting to Supabase...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

        # 2. Query the metadata table
        # Select all rows, ordering by ID (or any other field)
        response = supabase.table(TABLE_NAME).select("*").order("id").execute()
        metadata_rows: List[Dict[str, Any]] = response.data
        
        if not metadata_rows:
            print("[DB] Warning: No rows found in the startup_documents table.")
            return manifest_path
            
        print(f"[DB] Found {len(metadata_rows)} documents to process.")

        # 3. Process each document
        final_rows: List[str] = []
        
        for row in tqdm(metadata_rows, desc="ingesting documents", dynamic_ncols=True):
            file_link = row.get("document") # e.g., 'documents/business_plan.pdf'
            
            if not file_link:
                tqdm.write(f"Skipping row with missing document link: {row.get('id')}")
                continue

            # --- A. Download the file content from the storage bucket ---
            # Supabase download uses POST request to a special endpoint
            download_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{file_link}"
            
            response = requests.get(download_url, headers={"Authorization": f"Bearer {SUPABASE_KEY}"})

            if response.status_code != 200:
                tqdm.write(f"Error downloading {file_link}. Status: {response.status_code}")
                continue

            file_content = response.content
            
            # --- B. Extract Text ---
            raw_text = extract_text_from_bytes(file_link, file_content)
            doc_hash = sha1_bytes(file_content)

            # --- C. Build Manifest Row ---
            manifest_row = {
                "doc_id": doc_hash,
                "regulator_ns": row.get("entity", "unknown"), # Use 'entity' field as regulator_ns
                "path": file_link, 
                "text": raw_text, 
                "targets": {
                    "doc_type": row.get("document_type"),
                    "risk": row.get("risk_score"),
                },
                "meta": {
                    "source": "supabase_database",
                }
            }
            final_rows.append(json.dumps(manifest_row, ensure_ascii=False))

        # 4. Save the final JSONL manifest
        manifest_path.write_text("\n".join(final_rows), encoding="utf-8")
        
        print(f"[ingest] Startup manifest saved to {manifest_path} with {len(final_rows)} rows.")
        return manifest_path

    except Exception as e:
        print(f"[ERROR] Supabase Ingestion Failed: {e}")
        # Return empty path on failure
        return manifest_path


if __name__ == "__main__":
    ingest_startup_data()