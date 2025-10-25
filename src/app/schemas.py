"""
Typed schemas used across the pipeline.
Centralizes shapes for config, documents, and extracted text.
Extensible without touching downstream code.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Dict


class ChunkingConfig(BaseModel):
    target_tokens: int = 800         # default chunk length for later stages
    overlap_tokens: int = 120        # default overlap


class RegulatorRef(BaseModel):
    id: str                          # e.g., "qcb"
    name: str                        # human-readable name
    rulepack: str                    # path to YAML rule-pack


class AppConfig(BaseModel):
    project_name: str = "regnav"
    data_dir: str = "./data"
    interim_dir: str = "./data/interim"
    processed_dir: str = "./data/processed"
    raw_dir: str = "./data/raw"
    regulators: List[RegulatorRef]   # list allows easy multi-regulator support
    chunking: ChunkingConfig = ChunkingConfig()


class DocRecord(BaseModel):
    """
    One logical document registered in the manifest.
    No file I/O logic here — pure data contract for portability.
    """
    doc_id: str                      # stable ID (uuid8 in builder)
    path: str                        # repo-relative path to the source file
    doc_type: str                    # "regulator_rule" | "startup_plan" | "policy" | "legal" | "resources" | "other"
    regulator_ns: Optional[str] = None   # e.g., "qcb" when it’s a regulator file
    language: Optional[str] = None       # set later if you add langid
    metadata: Dict[str, str] = {}        # free-form (original filename, etc.)


class ExtractedText(BaseModel):
    """
    Normalize extracted content into a single shape for downstream modules.
    """
    doc_id: str
    text: str
    pages: Optional[int] = None
    extra: Dict[str, str] = {}           # future: layout stats, parse quality, etc.
