import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .common import UID, TARGETS, labels, read_csv, save_json, scores
from .model import KneeModel


class Studies(Dataset):
    def __init__(self, df, cache):
        self.df = df.reset_index(drop=True)
        self.cache = Path(cache)
        self.y = labels(df)
        self.gold = df.reindex(columns=[f"gold::{t}" for t in TARGETS], fill_value=0).to_numpy(dtype=bool)
        missing = [uid for uid in self.df[UID] if not (self.cache / f"{uid}.npz").exists()]
        if missing:
            raise ValueError(f"Missing cache for {len(missing)} studies; first: {missing[:3]}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.iloc[idx][UID]
        with np.load(self.cache / f"{uid}.npz") as z:
            x = torch.from_numpy(z["x"].astype(np.float32) / 127.5 - 1)
            mask = torch.from_numpy(z["mask"].copy())
        if not mask.any():
            raise ValueError(f"No valid imaging: {uid}")
        return x, mask, torch.from_numpy(self.y[idx]), torch.from_numpy(self.gold[idx])


def masked_loss(logits, y, gold, weak_weight):
    known = torch.isfinite(y)
    weights = torch.where(gold, 1., weak_weight) * known
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.nan_to_num(y), reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, ys, gs = [], [], []
    for x, mask, y, gold in loader:
        predictions.append(model(x.to(device), mask.to(device)).sigmoid().cpu().numpy())
        ys.append(y.numpy())
        gs.append(gold.numpy())
    return np.concatenate(predictions), np.concatenate(ys), np.concatenate(gs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--weak-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if min(args.epochs, args.batch_size, args.accum) < 1 or not 0 < args.weak_weight <= 1:
        raise ValueError("Invalid training settings")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # Never silently overwrite an experiment.
    save_json(out / "config.json", vars(args))
    cache_config = json.loads((Path(args.cache) / "config.json").read_text())
    df = read_csv(args.folds)
    if df[UID].duplicated().any() or "fold" not in df or args.fold not in set(df["fold"]):
        raise ValueError("Invalid fold table")
    # Only studies carrying known labels contribute to training/validation.
    known = np.isfinite(labels(df)).any(axis=1)
    train = df[(df["fold"] != args.fold) & known]
    val = df[(df["fold"] == args.fold) & known]
    if train.empty or val.empty:
        raise ValueError("Empty labeled training/validation split")
    train_loader = DataLoader(Studies(train, args.cache), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(Studies(val, args.cache), batch_size=args.batch_size, num_workers=args.workers)
    model = KneeModel(args.backbone, args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -float("inf")
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        total = 0.
        for step, (x, mask, y, gold) in enumerate(train_loader):
            x, mask, y, gold = [v.to(device) for v in (x, mask, y, gold)]
            # Normalize the final incomplete accumulation window correctly.
            start = (step // args.accum) * args.accum
            window = min(args.accum, len(train_loader) - start)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = masked_loss(model(x, mask), y, gold, args.weak_weight)
            scaler.scale(loss / window).backward()
            total += loss.item()
            if (step + 1) % args.accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        p, y, gold = predict(model, val_loader, device)
        gold_metrics = scores(np.where(gold, y, np.nan), p)
        all_metrics = scores(y, p)
        # Prefer gold AUC if any class can be evaluated; otherwise a labeled-data proxy.
        selection = gold_metrics["macro_auc_available"]
        source = "gold"
        if selection is None:
            selection, source = all_metrics["macro_auc_available"], "all_labels_proxy"
        log = {"epoch": epoch + 1, "train_loss": total / len(train_loader),
               "gold": gold_metrics, "all_labels": all_metrics, "selection_source": source}
        print(json.dumps(log), flush=True)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        checkpoint = {"model": model.state_dict(), "backbone": args.backbone, "targets": TARGETS,
                      "cache_config": cache_config, "epoch": epoch + 1, "config": vars(args)}
        torch.save(checkpoint, out / "last.pt")
        if selection is not None and selection > best:
            best = selection
            torch.save(checkpoint, out / "best.pt")
            oof = val[[UID]].copy()
            oof[TARGETS] = p
            oof.to_csv(out / "validation_predictions.csv", index=False)
    if best == -float("inf"):
        print("No evaluable binary AUC classes: best.pt was not created. Inspect labels/folds; last.pt exists.")


if __name__ == "__main__":
    main()
