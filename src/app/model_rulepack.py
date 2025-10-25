"""
Model-driven rule-pack generator.

Pipeline:
1) Segment regulator documents into "articles" (regex headings; fallback to semantic splits).
2) Embed segments with a sentence-transformer.
3) Classify each segment into a domain via label-embedding similarity (zero-shot).
4) Extract structured fields (article_id, title, text). Keep plain text for auditability.
5) Return a rule-pack dict that can be merged into config/regulators/<ns>.yaml.

Notes:
- Zero-shot: no training needed; uses label semantics.
- Deterministic fallbacks for IDs/titles if headings are noisy.
- Extend DOMAIN_LABELS to scale across jurisdictions.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Segmentation regexes ---
RE_ARTICLE = re.compile(
    r"""(?mx)                      # multiline, verbose
    ^\s*
    (?:
       (?:Article|ART\.?|ARTICLE)  # Article / ART / ARTICLE
       \s*
    )?
    (?P<num>\d+(?:\.\d+)*)
    (?:\s*[:\-–]\s*|\s+)           # colon or dash or space
    (?P<title>[^\n]{1,120})
    $
    """
)

RE_SECTION = re.compile(
    r"""(?mx)
    ^\s*(?:Section|SECTION)\s+(?P<num>\d+)\s*[:\-–]\s*(?P<title>[^\n]{1,120})$
    """
)


# --- Domain label surface forms (extend once; model handles synonyms) ---
DOMAIN_LABELS: Dict[str, List[str]] = {
    "aml_kyc": [
        "AML and KYC obligations",
        "anti-money laundering",
        "customer due diligence",
        "transaction monitoring",
        "suspicious activity reporting",
    ],
    "data_residency": [
        "data protection and residency",
        "personal data storage location",
        "privacy and consent",
        "data localization",
        "PII residency",
    ],
    "governance": [
        "corporate governance and audit",
        "internal audit",
        "compliance officer role",
        "board oversight",
        "fit and proper requirements",
    ],
    "licensing_capital": [
        "licensing and minimum capital",
        "licensing categories",
        "paid-up capital",
        "application requirements",
        "authorization conditions",
    ],
}

@dataclass
class Segment:
    article_id: str
    title: str
    text: str
    start: int
    end: int

class RulepackGenerator:
    """
    Zero-shot semantic classifier over text segments to assign regulatory domains.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name) # self.model = SentenceTransformer(model_name, device="cuda")
        # Build label embedding matrix once
        self._label_texts: List[Tuple[str, str]] = []  # [(domain_id, label_text)]
        for dom, labels in DOMAIN_LABELS.items():
            for lab in labels:
                self._label_texts.append((dom, lab))
        _, label_texts = zip(*self._label_texts)
        self._label_emb = self.model.encode(list(label_texts), normalize_embeddings=True)

    # --------- Segmentation ---------
    def segment(self, text: str) -> list[Segment]:
        if not text or not text.strip():
            return []

        # 1) Try article-style and numeric headings
        matches = list(RE_ARTICLE.finditer(text))
        segs: list[Segment] = []
        if matches:
            for i, m in enumerate(matches):
                art_id = m.group("num").strip()
                title = m.group("title").strip()
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                if body:
                    segs.append(Segment(article_id=art_id, title=title, text=body, start=start, end=end))
            if segs:
                return segs

        # 2) Section fallback → split paragraphs into smaller segments
        secm = list(RE_SECTION.finditer(text))
        if secm:
            for i, m in enumerate(secm):
                sec_id = m.group("num").strip()
                sec_title = m.group("title").strip()
                start = m.end()
                end = secm[i + 1].start() if i + 1 < len(secm) else len(text)
                body = text[start:end].strip()
                paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
                for j, p in enumerate(paras):
                    art_id = f"{sec_id}.{j+1}"
                    title = sec_title if j == 0 else f"{sec_title} (part {j+1})"
                    segs.append(Segment(article_id=art_id, title=title, text=p, start=start, end=end))
            if segs:
                return segs

        # 3) Last resort: fixed windows
        return self._fixed_windows(text, win_chars=2000, overlap=200)

    def _fixed_windows(self, text: str, win_chars: int, overlap: int) -> List[Segment]:
        segs: List[Segment] = []
        n = len(text)
        s = 0
        k = 1
        while s < n:
            e = min(n, s + win_chars)
            chunk = text[s:e].strip()
            if chunk:
                segs.append(Segment(article_id=f"F{k}", title=f"Fragment {k}", text=chunk, start=s, end=e))
                k += 1
            if e >= n:
                break
            s = max(0, e - overlap)
        return segs

    # --------- Classification ---------
    def classify_domain(self, segs: list[Segment]) -> list[tuple[Segment, Optional[str], float]]:
        if not segs:
            return []
        texts = []
        keep = []
        for s in segs:
            t = (s.title or "").strip() + "\n\n" + (s.text or "").strip()
            if t.strip():
                texts.append(t)
                keep.append(s)
        if not texts:
            return []
        seg_emb = self.model.encode(texts, normalize_embeddings=True)
        sims = np.matmul(seg_emb, self._label_emb.T)
        out = []
        for i, s in enumerate(keep):
            j = int(np.argmax(sims[i]))
            dom = self._label_texts[j][0]
            score = float(sims[i, j])
            assigned = dom if score >= 0.35 else None   # threshold down from 0.45
            out.append((s, assigned, score))
        return out

    # --------- Rule-pack assembly ---------
    def build_rulepack(self, ns: str, jurisdiction: str, name: str, doc_texts: List[str]) -> Dict:
        """
        Given all merged text for a regulator namespace, produce rule-pack dict.
        """
        # 1) segment all texts
        segs: List[Segment] = []
        for t in doc_texts:
            segs.extend(self.segment(t))

        # 2) classify domains
        labeled = self.classify_domain(segs)

        # 3) assemble articles
        arts: List[Dict] = []
        for seg, dom, score in labeled:
            arts.append({
                "article_id": seg.article_id,
                "title": seg.title,
                "text": seg.text,
                "domain": dom,
                "confidence": round(score, 3),
            })

        # 4) compose final rulepack
        rulepack = {
            "id": ns,
            "jurisdiction": jurisdiction,
            "name": name,
            # domains are referenced by key; UI can show friendly names
            "domains": [{"id": k, "name": " ".join(w.capitalize() for w in k.split("_"))} for k in DOMAIN_LABELS.keys()],
            "articles": arts,
        }
        return rulepack
