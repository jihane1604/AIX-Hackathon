"""
Merge a newly generated rule-pack into an existing YAML (if any), keyed by article_id.
Keeps manual edits; updates changed segments; stable order by natural article numbering.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List

def _natkey(aid: str):
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", aid))

def merge_rulepacks(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(existing) if existing else {}
    out.setdefault("id", new.get("id"))
    out.setdefault("jurisdiction", new.get("jurisdiction", ""))
    out.setdefault("name", new.get("name", out.get("id", "").upper()))
    out["domains"] = out.get("domains") or new.get("domains", [])

    # Merge articles by article_id
    ex_index = {a["article_id"]: a for a in out.get("articles", []) if "article_id" in a}
    for art in new.get("articles", []):
        aid = art.get("article_id")
        if not aid:
            continue
        ex_index[aid] = {**ex_index.get(aid, {}), **art}

    out["articles"] = sorted(ex_index.values(), key=lambda a: _natkey(a.get("article_id", "")))
    return out
