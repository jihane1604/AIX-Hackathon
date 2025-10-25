"""
End-to-end inference utilities:
- load classifiers and retriever
- classify doc_type and risk
- retrieve top-k regulatory articles for explanations/gaps
- return a unified result dict ready for the web UI
"""
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.config.defaults import MODEL_OUT_DIR, DOC_TYPE_LABELS, RISK_LABELS, DEVICE
from models.retriever.search import RegulatorSearcher
import torch

def _load_clf(name: str):
    path = MODEL_OUT_DIR / name
    tok = AutoTokenizer.from_pretrained(str(path))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(path)).to(DEVICE)
    mdl.eval()
    return tok, mdl

@torch.no_grad()
def _predict_cls(text: str, tok, mdl, labels: List[str]):
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    out = mdl(**enc).logits
    probs = out.softmax(-1)[0].detach().cpu().tolist()
    idx = int(out.argmax(-1).item())
    return {"label": labels[idx], "probs": {labels[i]: float(p) for i, p in enumerate(probs)}}

class InferencePipeline:
    def __init__(self, regulator_ns: str):
        # Load classifiers
        self.dt_tok, self.dt_mdl = _load_clf("doc_type_clf")
        self.risk_tok, self.risk_mdl = _load_clf("risk_clf")
        # Load retriever for selected regulator
        self.searcher = RegulatorSearcher(regulator_ns)

    def run(self, text: str, k: int = 5) -> Dict[str, Any]:
        # 1) Document type
        doc_type = _predict_cls(text, self.dt_tok, self.dt_mdl, DOC_TYPE_LABELS)

        # 2) Retrieve top-k relevant regulatory articles
        hits = self.searcher.search(text, k=k)

        # 3) Risk prediction (baseline uses raw text; you can concatenate hits texts for stronger signal)
        risk = _predict_cls(text, self.risk_tok, self.risk_mdl, RISK_LABELS)

        return {
            "doc_type": doc_type,
            "risk": risk,
            "retrieved": hits
        }
    
    def set_regulator(self, regulator_ns: str):
        """Allows updating the search context without reloading heavy classifiers."""
        if self.searcher.regulator_ns != regulator_ns:
            self.searcher = RegulatorSearcher(regulator_ns) 
            # Note: This requires RegulatorSearcher to handle efficient index switching.
            self.searcher.regulator_ns = regulator_ns # Assuming RegulatorSearcher doesn't have a built-in property for this
