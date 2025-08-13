from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional


def _models_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "..", "models")
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    return root


def _symbol_dir(symbol_norm: str) -> str:
    d = os.path.join(_models_root(), symbol_norm)
    os.makedirs(d, exist_ok=True)
    return d


def model_bundle_path(symbol_norm: str, horizon: int, kind: str = "return_lgbm") -> str:
    fn = f"{kind}_h{int(horizon)}.pkl"
    return os.path.join(_symbol_dir(symbol_norm), fn)


def save_model_bundle(symbol_norm: str, horizon: int, bundle: Dict[str, Any], kind: str = "return_lgbm") -> str:
    path = model_bundle_path(symbol_norm, horizon, kind=kind)
    meta = dict(bundle.get("meta") or {})
    meta["saved_at"] = datetime.utcnow().isoformat()
    bundle["meta"] = meta
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    return path


def load_model_bundle(symbol_norm: str, horizon: int, kind: str = "return_lgbm") -> Optional[Dict[str, Any]]:
    path = model_bundle_path(symbol_norm, horizon, kind=kind)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


