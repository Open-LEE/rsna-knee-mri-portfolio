"""Inspect labels and make study/group-level folds. Does not infer report labels."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, GroupKFold

from .common import UID, TARGETS, read_csv, labels, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/data")
    ap.add_argument("--weak-labels", help="CSV: StudyInstanceUID and 12 targets; blanks remain unknown")
    ap.add_argument("--groups", help="Optional CSV: StudyInstanceUID, group (patient/site)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = read_csv(Path(args.data) / "train.csv")
    if df[UID].duplicated().any():
        raise ValueError("Duplicate training study IDs")
    gold = labels(df)
    y = gold.copy()
    if args.weak_labels:
        weak = read_csv(args.weak_labels)
        if weak[UID].duplicated().any() or not set(weak[UID]).issubset(set(df[UID])):
            raise ValueError("Weak label IDs must be unique and belong to training set")
        w = labels(weak.set_index(UID).reindex(df[UID]).reset_index())
        y = np.where(np.isfinite(gold), gold, w)
    # Gold entries always take precedence. Preserve source per target.
    result = df[[UID]].copy()
    for i, name in enumerate(TARGETS):
        result[name] = y[:, i]
        result[f"gold::{name}"] = np.isfinite(gold[:, i]).astype(int)
    result["fold"] = -1
    if args.groups:
        groups = read_csv(args.groups)
        if groups[UID].duplicated().any() or "group" not in groups:
            raise ValueError("Groups require unique study IDs and group column")
        g = df[[UID]].merge(groups[[UID, "group"]], on=UID, how="left", validate="one_to_one")
        if g["group"].isna().any():
            raise ValueError("Every study needs a group")
        split = GroupKFold(args.folds).split(result, groups=g["group"])
    else:
        split = KFold(args.folds, shuffle=True, random_state=args.seed).split(result)
    for fold, (_, idx) in enumerate(split):
        result.loc[idx, "fold"] = fold
    result.to_csv(out / "folds.csv", index=False)
    summary = {"studies": len(df), "reports": int(df["Report"].notna().sum()) if "Report" in df else 0,
               "gold_known": dict(zip(TARGETS, np.isfinite(gold).sum(0).tolist())),
               "training_known": dict(zip(TARGETS, np.isfinite(y).sum(0).tolist())),
               "grouped": bool(args.groups), "seed": args.seed,
               "note": "Random study folds cannot guarantee patient/site isolation; gold-only AUC may be unstable."}
    save_json(out / "audit.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
