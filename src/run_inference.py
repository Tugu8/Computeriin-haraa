"""data/images доторх бүх зураг дээр гурван MiDaS загварыг ажиллуулж,
depth map болон хурдны хэмжилтийг results/ руу хадгална.

    python -m src.run_inference                 # бүх загвар
    python -m src.run_inference --models MiDaS_small --runs 3
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.depth import (MODEL_TYPES, benchmark, count_params, depth_to_color,
                       load_model, predict_depth, read_image)

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=ROOT / "data" / "images", type=Path)
    ap.add_argument("--out", default=ROOT / "results", type=Path)
    ap.add_argument("--models", nargs="+", default=MODEL_TYPES)
    ap.add_argument("--runs", type=int, default=5, help="хурд хэмжих давталт")
    args = ap.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXT)
    if not paths:
        raise SystemExit(f"{args.images} дотор зураг алга")
    images = {p.stem: read_image(p) for p in paths}

    depth_dir = args.out / "depth_maps"
    npy_dir = args.out / "depth_npy"
    depth_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    preds = {}  # (model, image) -> depth
    speed_rows = []
    bench_img = next(iter(images.values()))

    for m in args.models:
        print(f"== {m}")
        model, transform, device = load_model(m)
        for name, img in images.items():
            d = predict_depth(model, transform, img, device)
            preds[(m, name)] = d
            np.save(npy_dir / f"{name}_{m}.npy", d.astype(np.float16))
            cv2.imwrite(str(depth_dir / f"{name}_{m}.png"),
                        cv2.cvtColor(depth_to_color(d), cv2.COLOR_RGB2BGR))
        mean, std = benchmark(model, transform, bench_img, device, runs=args.runs)
        params = count_params(model) / 1e6
        print(f"   {params:.1f}M параметр, {mean:.3f} ± {std:.3f} с/зураг ({device})")
        speed_rows.append([m, f"{params:.1f}", f"{mean:.3f}", f"{std:.3f}", f"{1 / mean:.2f}", device.type])
        del model

    with open(args.out / "speed_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "params_M", "sec_per_image", "std", "fps", "device"])
        w.writerows(speed_rows)

    # Зураг x загвар харьцуулсан grid
    n_rows, n_cols = len(images), len(args.models) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows))
    axes = np.atleast_2d(axes)
    for r, (name, img) in enumerate(images.items()):
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(name, fontsize=10)
        for c, m in enumerate(args.models, start=1):
            axes[r, c].imshow(depth_to_color(preds[(m, name)]))
            if r == 0:
                axes[r, c].set_title(m)
        if r == 0:
            axes[r, 0].set_title("Оролт")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.out / "comparison_grid.png", dpi=110)
    print("Хадгалсан:", args.out)


if __name__ == "__main__":
    main()
