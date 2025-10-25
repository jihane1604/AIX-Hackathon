"""
Metrics helpers for multi-class classification:
- compute_doc_type_metrics
- compute_risk_metrics
Both return dicts compatible with HF Trainer.
"""
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

def _common(preds, refs):
    acc = accuracy_score(refs, preds)
    f1m = f1_score(refs, preds, average="macro")
    p, r, f1w, _ = precision_recall_fscore_support(refs, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1m, "precision_weighted": p, "recall_weighted": r, "f1_weighted": f1w}

def compute_doc_type_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return _common(preds, labels)

def compute_risk_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return _common(preds, labels)
