"""Select one series per plane, order slices geometrically, cache uint8 arrays."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .common import UID, PLANES, read_csv, save_json

VERSION = 1


def volume(folder, slices, size):
    import pydicom
    paths = sorted(Path(folder).glob("*.dcm"))
    if not paths:
        raise ValueError(f"No DICOM files in {folder}")
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in paths]
    if all(hasattr(d, "ImageOrientationPatient") and hasattr(d, "ImagePositionPatient") for d in headers):
        orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
        if not all(np.allclose(d.ImageOrientationPatient, orientation, atol=0.01) for d in headers):
            raise ValueError("Inconsistent slice orientation")
        normal = np.cross(orientation[:3], orientation[3:])
        order = np.argsort([np.dot(d.ImagePositionPatient, normal) for d in headers])
    elif all(hasattr(d, "InstanceNumber") for d in headers):
        order = np.argsort([int(d.InstanceNumber) for d in headers])
    else:
        raise ValueError("Cannot reliably order slices")
    chosen = order[np.rint(np.linspace(0, len(order) - 1, slices)).astype(int)]
    imgs = []
    for idx in chosen:
        d = pydicom.dcmread(paths[idx])
        a = d.pixel_array.astype(np.float32)
        if a.ndim != 2:
            raise ValueError("Only single-frame grayscale DICOM supported")
        a = a * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0))
        if getattr(d, "PhotometricInterpretation", "") == "MONOCHROME1":
            a = -a
        imgs.append(np.asarray(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR)))
    a = np.stack(imgs)
    lo, hi = np.percentile(a, [1, 99])
    if not np.isfinite(a).all() or hi <= lo:
        raise ValueError("Invalid or constant image intensities")
    return np.rint(np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def one(job):
    uid, rows, root, out, slices, size = job
    dest = Path(out) / f"{uid}.npz"
    if dest.exists():
        with np.load(dest) as z:
            if z["x"].shape != (3, slices, size, size) or int(z["version"]) != VERSION:
                raise ValueError(f"Stale cache: {dest}; use a fresh cache directory")
        return {UID: uid, "status": "cached"}
    x = np.zeros((3, slices, size, size), dtype=np.uint8)
    mask = np.zeros(3, dtype=bool)
    errors, selected = [], []
    for plane_idx, plane in enumerate(PLANES):
        candidates = [r for r in rows if str(r.get("Anatomical_Plane", "")).lower() == plane.lower()]
        def priority(r):
            return (str(r.get("Fluid_Sensitive", 0)) in ("1", "1.0"),
                    str(r.get("Fat_Suppression", 0)) in ("1", "1.0"))
        candidates.sort(key=priority, reverse=True)
        for r in candidates:
            sid = str(r["SeriesInstanceUID"])
            if not sid or not all(c.isalnum() or c in '.-_' for c in sid):
                raise ValueError("Unsafe series identifier")
            try:
                x[plane_idx] = volume(Path(root) / uid / sid, slices, size)
                mask[plane_idx] = True
                selected.append(sid)
                break
            except Exception as e:
                errors.append(f"{sid}: {type(e).__name__}: {e}")
    if not mask.any():
        return {UID: uid, "status": "failed", "errors": errors}
    temp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(temp, x=x, mask=mask, version=VERSION)
    temp.replace(dest)
    return {UID: uid, "status": "ok", "planes": mask.tolist(), "series": selected, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if min(args.slices, args.size, args.workers) <= 0:
        raise ValueError("Sizes and workers must be positive")
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    ids = read_csv(data / f"{args.split}.csv")[UID].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate study IDs")
    if args.limit:
        ids = ids[:args.limit]
    meta = read_csv(data / f"{args.split}_series.csv")
    required = {"SeriesInstanceUID", "Anatomical_Plane"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Missing series columns: {required - set(meta.columns)}")
    groups = {uid: group.to_dict("records") for uid, group in meta.groupby(UID)}
    config = {"version": VERSION, "slices": args.slices, "size": args.size,
              "data": str(data.resolve()), "split": args.split}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cache configuration changed; use a new output directory")
    save_json(config_path, config)
    jobs = [(uid, groups.get(uid, []), str(data / f"{args.split}_series"), str(out), args.slices, args.size) for uid in ids]
    from tqdm import tqdm
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(tqdm(pool.map(one, jobs), total=len(jobs)))
    save_json(out / "manifest.json", records)
    failures = sum(r["status"] == "failed" for r in records)
    print(f"Studies: {len(records)}, failed: {failures}. See manifest.json for decoding errors.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
