"""
Config loader helpers. No hard-coded paths anywhere else.
"""
import os
import yaml
from .schemas import AppConfig


def load_config(path: str) -> AppConfig:
    """Load YAML config into typed AppConfig."""
    with open(path, "r", encoding="utf-8") as f:
        return AppConfig(**yaml.safe_load(f))


def resolve_path(root: str, maybe_rel: str) -> str:
    """Turn a repo-relative path into absolute; leave absolute paths intact."""
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.normpath(os.path.join(root, maybe_rel))
