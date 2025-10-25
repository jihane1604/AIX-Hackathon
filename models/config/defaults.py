"""
Global defaults for models: device, hyperparameters, paths, and label maps.
All modules import from here to stay consistent.
"""
from pathlib import Path
import torch
import json

# Root repo directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Paths
DATA_DIR = ROOT_DIR / "data" / "datasets"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODEL_OUT_DIR = ROOT_DIR / "models" / "artifacts"
LOG_DIR = ROOT_DIR / "models" / "logs"
REPORTS_DIR = ROOT_DIR / "reports"

# MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
# LOG_DIR.mkdir(parents=True, exist_ok=True)
# REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Labels (doc_type, risk) – loaded from artifacts
with open(ARTIFACTS_DIR / "id2label.json", "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)

DOC_TYPE_LABELS = LABEL_MAP["doc_type"]
RISK_LABELS = LABEL_MAP["risk"]

NUM_DOC_TYPE_LABELS = len(DOC_TYPE_LABELS)
NUM_RISK_LABELS = len(RISK_LABELS)

# Backbone defaults
DOC_TYPE_BACKBONE = "distilroberta-base"
RISK_BACKBONE = "distilroberta-base"
RETRIEVER_BACKBONE = "intfloat/e5-base"  # strong baseline for retrieval

# Tokenization / batching
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

# Optim
LEARNING_RATE = 5e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

# Device – default GPU if present
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
