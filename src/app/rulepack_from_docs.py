"""
Parse regulator documents (doc_type_refined='regulator_rule') into structured rule articles.
Heuristics:
- Detect 'SECTION <n>:' and 'Article x.y:' headers.
- Assign domain by keyword.
- Preserve existing articles; merge by article_id.
Outputs: updates config/regulators/<ns>.yaml.
"""
from __future__ import annotations
import re, yaml, json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from app.aliases import detect_regulator

RE_SECTION = re.compile(r"^\s*(SECTION|Section)\s+(\d+)\s*:\s*(.+)$", re.M)
RE_ARTICLE = re.compile(r"^\s*(Article|ARTICLE)\s+(\d+(?:\.\d+)*)\s*:\s*(.+)$", re.M)

DOMAIN_KEYWORDS = {
    "aml_kyc": re.compile(r"\b(aml|cft|kyc|due\s+diligence|transaction\s+monitoring)\b", re.I),
    "data_residency": re.compile(r"\b(data\s+protection|residency|pii|consent|privacy)\b", re.I),
    "governance": re.compile(r"\b(compliance\s+officer|audit|board|governance|fit\s+and\s+proper)\b", re.I),
    "licensing_capital": re.compile(r"\b(licensing|capital|required|min(imum)?\s+capital|category)\b", re.I),
}

def assign_domain(title: str, body: str) -> Optional[str]:
    blob = f"{title}\n{body}"
    for dom, rx in DOMAIN_KEYWORDS.items():
        if rx.search(blob):
            return dom
    return None

def load_rulepack(path: Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def dump_rulepack(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(obj, path.open("w", encoding="utf-8"), sort_keys=False, allow_unicode=True)

def parse_articles(text: str) -> List[Dict[str, Any]]:
    """
    Returns list of {id, title, text, section, domain?}
    """
    arts: List[Dict[str, Any]] = []
    # Find all article headers with positions
    matches = list(RE_ARTICLE.finditer(text))
    for i, m in enumerate(matches):
        art_id = m.group(2).strip()
        title = m.group(3).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        dom = assign_domain(title, body)
        arts.append({"article_id": art_id, "title": title, "text": body, "domain": dom})
    return arts

def merge_articles(existing: List[Dict[str, Any]], new_arts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = {a["article_id"]: a for a in existing}
    for a in new_arts:
        index[a["article_id"]] = {**index.get(a["article_id"], {}), **a}  # last write wins
    # Stabilize ordering by article_id
    return sorted(index.values(), key=lambda x: tuple(int(t) if t.isdigit() else 0 for t in x["article_id"].split(".")))

def build_rulepacks_from_enriched(root: Path) -> List[Path]:
    interim = root / "data" / "interim"
    mf = interim / "manifest.enriched.json"
    if not mf.exists():
        raise FileNotFoundError("data/interim/manifest.enriched.json not found. Run enrichment first.")

    records = json.loads(mf.read_text(encoding="utf-8"))
    # Group regulator_rule docs by regulator_ns
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if rec.get("doc_type_refined") != "regulator_rule":
            continue
        ns = rec.get("regulator_ns") or detect_regulator(rec.get("path", "")) or "unknown"
        grouped.setdefault(ns, []).append(rec)

    out_paths: List[Path] = []
    for ns, recs in grouped.items():
        # Load concatenated text
        texts: List[str] = []
        for rec in recs:
            txt = (interim / f"{rec['doc_id']}.txt").read_text(encoding="utf-8")
            texts.append(txt)
        merged_text = "\n\n".join(texts)
        articles = parse_articles(merged_text)

        # Load existing rulepack and merge
        rp_path = root / "config" / "regulators" / f"{ns}.yaml"
        rp = load_rulepack(rp_path)
        rp.setdefault("id", ns)
        rp.setdefault("jurisdiction", rp.get("jurisdiction", ""))
        rp.setdefault("name", rp.get("name", ns.upper()))
        rp.setdefault("domains", rp.get("domains", []))
        rp["articles"] = merge_articles(rp.get("articles", []), articles)

        dump_rulepack(rp_path, rp)
        out_paths.append(rp_path)

    return out_paths
