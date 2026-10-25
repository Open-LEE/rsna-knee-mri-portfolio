"""Offline prediction from cached test MRI; preserves sample submission row order."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import UID, TARGETS, read_csv
from .model import KneeModel
from .train import Studies, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    df = read_csv(args.sample)
    if list(df.columns) != [UID] + TARGETS or df[UID].duplicated().any() or df.empty:
        raise ValueError("Unexpected sample submission schema or duplicate IDs")
    config = json.loads((Path(args.cache) / "config.json").read_text())
    loader = DataLoader(Studies(df, args.cache), batch_size=args.batch_size, num_workers=args.workers)
    predictions = []
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["targets"] != TARGETS:
            raise ValueError("Checkpoint target order mismatch")
        for key in ["version", "slices", "size"]:
            if ck["cache_config"][key] != config[key]:
                raise ValueError(f"Checkpoint/cache mismatch: {key}")
        model = KneeModel(ck["backbone"], pretrained=False)
        model.load_state_dict(ck["model"])
        model.to(args.device)
        p, _, _ = predict(model, loader, args.device)
        predictions.append(p)
        del model
    p = np.mean(predictions, axis=0)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid predictions")
    df[TARGETS] = p
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} studies to {args.out}")


if __name__ == "__main__":
    main()
