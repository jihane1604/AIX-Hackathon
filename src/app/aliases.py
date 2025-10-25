"""
Shared alias maps and tiny helpers used across enrichment and rule-pack parsing.
Extend aliases here only; no other code changes needed.
"""
import re
from typing import Optional

REGULATOR_ALIASES = {
    "qcb": ["qcb", "qatar central bank"],
    "qfc": ["qfc", "qfcra", "qatar financial centre", "qatar financial center", "qfc regulatory authority"],
}
ALIAS_PATTERNS = {
    rid: [re.compile(rf"\b{re.escape(a)}\b", re.I) for a in aliases]
    for rid, aliases in REGULATOR_ALIASES.items()
}

def detect_regulator(text: str) -> Optional[str]:
    for rid, pats in ALIAS_PATTERNS.items():
        if any(p.search(text) for p in pats):
            return rid
    return None
