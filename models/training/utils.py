"""
Shared training utilities:
- seed_everything()
- default TrainingArguments builder with tqdm enabled
"""
import random, numpy as np, torch
from transformers import TrainingArguments
from models.config.defaults import (
    MODEL_OUT_DIR, LOG_DIR, EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, DEVICE
)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_training_args(name: str) -> TrainingArguments:
    out_dir = (MODEL_OUT_DIR / name)
    # out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = (LOG_DIR / name)
    # log_dir.mkdir(parents=True, exist_ok=True)

    return TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(log_dir),
        # >>> progress bar settings <<<
        disable_tqdm=False,          # ensure tqdm is enabled
        logging_steps=10,            # emit log events frequently for smooth bar
        fp16=(DEVICE == "cuda"),
        report_to=["none"],          # add "tensorboard" if you want TB logs
        metric_for_best_model="f1",
    )

