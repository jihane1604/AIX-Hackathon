"""
Train document type classifier with visible tqdm progress.
"""
from transformers import AutoModelForSequenceClassification, Trainer
from models.preprocessing.datasets import load_splits, tokenize_for_doc_type
from models.training.utils import build_training_args, seed_everything
from models.training.callbacks import TqdmLogger
from models.evaluation.metrics import compute_doc_type_metrics
from models.config.defaults import DOC_TYPE_BACKBONE, NUM_DOC_TYPE_LABELS, MODEL_OUT_DIR

def main():
    seed_everything(42)
    dd = load_splits()
    tokenized, tok = tokenize_for_doc_type(dd)

    model = AutoModelForSequenceClassification.from_pretrained(
        DOC_TYPE_BACKBONE, num_labels=NUM_DOC_TYPE_LABELS
    )

    args = build_training_args("doc_type_clf")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
        tokenizer=tok,
        compute_metrics=compute_doc_type_metrics,
        callbacks=[TqdmLogger()],  # <-- progress bar callback
    )
    trainer.train()
    out = MODEL_OUT_DIR / "doc_type_clf"
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[doc_type] saved to {out}")

if __name__ == "__main__":
    main()
