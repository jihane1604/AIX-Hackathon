# Dataset format

Each row describes one startup document and its labels.
Use HuggingFace Datasets (Arrow) **or** JSONL (optionally `.zst`-compressed).

## JSONL row (schema)
```json
{
  "doc_id": "sha1:4c2a....",                // stable content hash
  "regulator_ns": "qcb",                    // target regulator namespace
  "path": "/abs/or/rel/path/to/file.pdf",   // location of the source file (pdf/docx/txt)
  "text": "",                               // optional: keep empty if file is large; pipeline will extract on-the-fly
  "targets": {
    "doc_type": "business_plan",            // one of: business_plan | policy | legal | other
    "risk": "high"                          // one of: none | low | medium | high
  },
  "meta": {
    "source": "synthetic|real|…",
    "created_at": "2025-10-25",
    "notes": ""
  }
}
