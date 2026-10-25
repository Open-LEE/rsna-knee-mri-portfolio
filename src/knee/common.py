import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = ["Sagittal", "Coronal", "Axial"]


def read_csv(path):
    df = pd.read_csv(path, dtype={UID: str, "SeriesInstanceUID": str})
    if UID not in df or df[UID].isna().any():
        raise ValueError(f"Missing {UID}: {path}")
    if not df[UID].map(lambda s: bool(s) and all(c.isalnum() or c in '.-_' for c in s)).all():
        raise ValueError("Unsafe study identifier")
    return df


def labels(df):
    y = df.reindex(columns=TARGETS).apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if np.isinf(y).any() or ((y[np.isfinite(y)] < 0) | (y[np.isfinite(y)] > 1)).any():
        raise ValueError("Labels must be in [0,1] or missing")
    return y


def scores(y, p):
    per_class = {}
    for i, name in enumerate(TARGETS):
        valid = np.isfinite(y[:, i])
        unique = np.unique(y[valid, i])
        per_class[name] = (float(roc_auc_score(y[valid, i], p[valid, i]))
                           if len(unique) == 2 and set(unique) == {0., 1.} else None)
    good = [v for v in per_class.values() if v is not None]
    return {"macro_auc_available": float(np.mean(good)) if good else None,
            "valid_classes": len(good), "per_class": per_class}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
