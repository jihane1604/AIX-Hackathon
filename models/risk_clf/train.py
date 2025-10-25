"""
Train risk classifier with visible tqdm progress.
"""
from transformers import AutoModelForSequenceClassification, Trainer
from models.preprocessing.datasets import load_splits, tokenize_for_risk
from models.training.utils import build_training_args, seed_everything
from models.training.callbacks import TqdmLogger
from models.evaluation.metrics import compute_risk_metrics
from models.config.defaults import RISK_BACKBONE, NUM_RISK_LABELS, MODEL_OUT_DIR

def main():
    seed_everything(42)
    dd = load_splits()
    tokenized, tok = tokenize_for_risk(dd)

    model = AutoModelForSequenceClassification.from_pretrained(
        RISK_BACKBONE, num_labels=NUM_RISK_LABELS
    )

    args = build_training_args("risk_clf")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
        tokenizer=tok,
        compute_metrics=compute_risk_metrics,
        callbacks=[TqdmLogger()],  # <-- progress bar callback
    )
    trainer.train()
    out = MODEL_OUT_DIR / "risk_clf"
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[risk] saved to {out}")

if __name__ == "__main__":
    main()
