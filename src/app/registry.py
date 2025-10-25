"""
Functions to create/update regulator rule-packs and config entries.
No hard-coded paths: everything resolves from config/config.yaml.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml

from .config import load_config
from .schemas_regulator import RegisterRegulatorRequest

def _load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _dump_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(obj, p.open("w", encoding="utf-8"), sort_keys=False, allow_unicode=True)

def repo_root() -> Path:
    # Assumes API runs from repo root or any subdir; resolves to top-level containing /config
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "config.yaml").exists():
            return parent
    # Fallback to cwd
    return Path.cwd()

def regulator_dir(root: Path) -> Path:
    return root / "config" / "regulators"

def config_path(root: Path) -> Path:
    return root / "config" / "config.yaml"

def ensure_rulepack(req: RegisterRegulatorRequest) -> str:
    """
    Create or update config/regulators/<id>.yaml.
    Returns repository-relative rulepack path (POSIX style).
    """
    root = repo_root()
    rp_path = regulator_dir(root) / f"{req.id}.yaml"

    # Merge with existing content if present; preserve any 'articles' already curated.
    rp = _load_yaml(rp_path)
    rp.setdefault("id", req.id)
    rp["jurisdiction"] = req.jurisdiction
    rp["name"] = req.name
    rp["domains"] = [{"id": d.id, "name": d.name} for d in req.domains]
    rp.setdefault("articles", rp.get("articles", []))  # keep if already populated

    _dump_yaml(rp_path, rp)
    return str(rp_path.relative_to(root)).replace("\\", "/")

def ensure_config_entry(req: RegisterRegulatorRequest, rulepack_rel: str) -> bool:
    """
    Ensure config/config.yaml contains the regulator entry (id, name, rulepack).
    Returns True if config file changed.
    """
    root = repo_root()
    cfg = _load_yaml(config_path(root))

    # If config file is empty, bootstrap minimal structure
    if not cfg:
        cfg = {
            "project_name": "AIX-HACKATHON",
            "data_dir": "./data",
            "interim_dir": "./data/interim",
            "processed_dir": "./data/processed",
            "raw_dir": "./data/raw",
            "regulators": [],
            "chunking": {"target_tokens": 800, "overlap_tokens": 120},
        }

    regs = cfg.get("regulators", [])
    changed = False
    for r in regs:
        if r.get("id") == req.id:
            # Update in place
            if r.get("name") != req.name or r.get("rulepack") != f"./{rulepack_rel}":
                r["name"] = req.name
                r["rulepack"] = f"./{rulepack_rel}"
                changed = True
            break
    else:
        # Append new entry
        regs.append({"id": req.id, "name": req.name, "rulepack": f"./{rulepack_rel}"})
        changed = True

    cfg["regulators"] = regs
    if changed:
        _dump_yaml(config_path(root), cfg)
    return changed
