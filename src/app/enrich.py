"""
Content-based enrichment over Step-1 manifest.
- Infers regulator_ns from content and folder names.
- Refines doc_type based on headings and keywords.
- Extracts startup key facts (capital, data location, officer presence, activities).
Outputs: manifest.enriched.json (idempotent).
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.aliases import detect_regulator

# --- patterns (extend only here) ---
RE_QAR = re.compile(r"\bqar\s*([0-9][0-9\.,_]*)\b", re.I)
RE_PAID_UP = re.compile(r"\b(paid[-\s]?up|paid up)\s+capital[:\s]\s*qar\s*([0-9][0-9\.,_]*)", re.I)
RE_DATA_LOC = re.compile(r"\b(aws|azure|gcp|data\s+(center|centre)|region|hosted|stored|residency)\b.*?(qatar|ireland|singapore|eu|us|me-[a-z]+-\d)\b", re.I)
RE_OFFICER = re.compile(r"\b(compliance\s+officer|mlro|money\s+laundering\s+reporting\s+officer)\b", re.I)
RE_ASSIGNED_ROLE = re.compile(r"\b(head\s+of\s+finance|cto|ceo)\b.*(will\s+act|acts)\s+as\s+(compliance\s+officer|mlro)\b", re.I)

ACTIVITY_PATTERNS = {
    "p2p_lending": re.compile(r"\b(p2p|peer[-\s]?to[-\s]?peer|marketplace\s+lending|crowdfunding)\b", re.I),
    "payments": re.compile(r"\b(payment|remittance|transfer|psp|acquiring)\b", re.I),
    "wallet": re.compile(r"\b(digital\s+wallet|e[-\s]?money|stored\s+value)\b", re.I),
    "cross_border": re.compile(r"\b(cross[-\s]?border|international)\b", re.I),
}

HEADINGS = {
    "policy": re.compile(r"\b(policy|procedure|kyc|aml|privacy)\b", re.I),
    "legal": re.compile(r"\b(articles\s+of\s+association|aoa|incorporation)\b", re.I),
    "rule": re.compile(r"\b(article\s+\d+(\.\d+)?|section\s+\d+)\b", re.I),
    "plan": re.compile(r"\b(executive\s+summary|business\s+model|market|financials)\b", re.I),
}

def refine_doc_type(filename: str, text: str, seed: str) -> str:
    name = filename.lower()
    if HEADINGS["rule"].search(text):
        return "regulator_rule"
    if HEADINGS["policy"].search(text):
        return "policy"
    if HEADINGS["legal"].search(text):
        return "legal"
    if HEADINGS["plan"].search(text) or "business plan" in name:
        return "startup_plan"
    return seed or "other"

def extract_startup_facts(text: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    # capital
    m_cap = RE_PAID_UP.search(text)
    if m_cap:
        facts["paid_up_capital_qar"] = float(m_cap.group(2).replace(",", "").replace("_", ""))
    # data location
    m_loc = RE_DATA_LOC.search(text)
    if m_loc:
        facts["data_location_mention"] = m_loc.group(0)
    # officer presence
    facts["mentions_compliance_officer"] = bool(RE_OFFICER.search(text))
    facts["improper_role_assignment"] = bool(RE_ASSIGNED_ROLE.search(text))
    # activities
    acts = [k for k, rx in ACTIVITY_PATTERNS.items() if rx.search(text)]
    if acts:
        facts["activities"] = acts
    return facts

def enrich_manifest(root: Path) -> Path:
    interim = root / "data" / "interim"
    mf = interim / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError("data/interim/manifest.json not found. Run Step-1 first.")

    records = json.loads(mf.read_text(encoding="utf-8"))
    enriched: List[Dict[str, Any]] = []
    for rec in records:
        doc_id = rec["doc_id"]
        txt_path = interim / f"{doc_id}.txt"
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        # regulator_ns inference from content and path
        regulator_ns = rec.get("regulator_ns")
        if not regulator_ns:
            regulator_ns = detect_regulator(text) or detect_regulator(rec["path"])
        # refine doc_type
        doc_type_refined = refine_doc_type(rec["metadata"]["original_name"], text, rec.get("doc_type", "other"))
        # startup facts (only for potential startup docs)
        facts = extract_startup_facts(text) if doc_type_refined in {"startup_plan", "policy", "legal"} else {}

        new_rec = dict(rec)
        new_rec["doc_type_refined"] = doc_type_refined
        new_rec["regulator_ns"] = regulator_ns
        new_rec["facts"] = facts
        enriched.append(new_rec)

    out = interim / "manifest.enriched.json"
    out.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
