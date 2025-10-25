"""
Dataset helpers for Arrow (HuggingFace Datasets):
- load_splits(): loads train/val/test Arrow dirs
- tokenize_*(): prepares tokenized datasets for classifier heads
- lazy_text(): allows deferred file reading for large docs (paths in rows)
"""
from typing import Tuple, Dict, Any
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from models.config.defaults import (
    DATA_DIR, DOC_TYPE_BACKBONE, RISK_BACKBONE, MAX_LENGTH,
    DOC_TYPE_LABELS, RISK_LABELS
)

def load_splits() -> DatasetDict:
    """
    Expects:
      data/datasets/train/
      data/datasets/val/
      data/datasets/test/
    Each is an Arrow dataset saved via save_to_disk.
    """
    dd = DatasetDict()
    for split, folder in [("train", "train"), ("validation", "val"), ("test", "test")]:
        path = DATA_DIR / folder
        if path.exists():
            dd[split] = load_from_disk(str(path))
    return dd

def lazy_text(example, base_dir: Path = Path("data")):
    """
    Lazy loader for large document text files.

    Used when dataset rows only store file paths instead of full text.

    Expected:
      example["path"] → relative or absolute path to .txt/.pdf/.docx
    Returns:
      dict with {"text": "<loaded content>"} or empty string if failed.
    """
    path = Path(example.get("path", ""))
    if not path.exists():
        path = base_dir / path  # try relative
    try:
        # Load only first ~10k chars to save memory for huge files
        text = path.read_text(encoding="utf-8", errors="ignore")[:10000]
    except Exception:
        text = ""
    return {"text": text}

def _tokenize(ds, tokenizer, target_key: str, label_list):
    label2id = {k: i for i, k in enumerate(label_list)}

    def _prep(ex):
        # Use 'text' field if present; otherwise, you can add a hook to load text from ex['path']
        text = ex.get("text") or ""
        enc = tokenizer(
            text if text else (ex.get("path") or ""),
            truncation=True, max_length=MAX_LENGTH
        )
        label_name = ex.get("targets", {}).get(target_key)
        if label_name is None:
            raise ValueError(f"Missing target '{target_key}' in row: {ex}")
        enc["labels"] = label2id[label_name]
        return enc

    # datasets.map shows its own progress bar; add a descriptive label
    return ds.map(
        _prep,
        batched=False,
        desc=f"Tokenizing for '{target_key}'",
        remove_columns=[c for c in ds.column_names if c not in ("text","targets","path")]
    )


def tokenize_for_doc_type(dd: DatasetDict):
    tok = AutoTokenizer.from_pretrained(DOC_TYPE_BACKBONE, use_fast=True)
    out = DatasetDict()
    for split, ds in dd.items():
        out[split] = _tokenize(ds, tok, "doc_type", DOC_TYPE_LABELS)
    return out, tok

def tokenize_for_risk(dd: DatasetDict):
    tok = AutoTokenizer.from_pretrained(RISK_BACKBONE, use_fast=True)
    out = DatasetDict()
    for split, ds in dd.items():
        out[split] = _tokenize(ds, tok, "risk", RISK_LABELS)
    return out, tok
