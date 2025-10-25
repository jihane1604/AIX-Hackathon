"""
Deterministic text chunker for later embeddings.
Token-agnostic but sized for ~800 tokens with overlap ~120 by default.
"""
from __future__ import annotations
from typing import List, Dict

def split_into_chunks(text: str, target_chars: int = 4000, overlap_chars: int = 600) -> List[Dict]:
    text = text or ""
    n = len(text)
    if n == 0:
        return []

    start = 0
    chunks = []
    while start < n:
        end = min(n, start + target_chars)
        chunk = text[start:end]
        chunks.append({"start": start, "end": end, "text": chunk})
        if end == n:
            break
        start = max(0, end - overlap_chars)
    return chunks
