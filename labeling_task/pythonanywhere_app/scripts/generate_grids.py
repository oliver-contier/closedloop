from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parents[1]
STATIC_GRIDS_DIR = APP_DIR / "static" / "grids"
DATA_DIR = APP_DIR / "data"
DEFAULT_EMBEDDING = (
    PROJECT_ROOT
    / "results"
    / "validation"
    / "subj-all_relthresh-0.60_k-17"
    / "emb_filtered.npy"
)


def square_thumb(path: Path, size: int) -> Image.Image:
    """Mimic top15 plotting: keep aspect ratio, no center-crop."""
    img = Image.open(path).convert("RGB")
    resampling = getattr(Image, "Resampling", Image)
    img.thumbnail((size, size), resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), color=(246, 248, 252))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_grid(image_paths: list[Path], out_path: Path, cell: int = 160, gap: int = 6) -> None:
    n_cols, n_rows = 8, 8
    canvas_w = n_cols * cell + (n_cols + 1) * gap
    canvas_h = n_rows * cell + (n_rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(246, 248, 252))

    for idx, p in enumerate(image_paths):
        r, c = divmod(idx, n_cols)
        x = gap + c * (cell + gap)
        y = gap + r * (cell + gap)
        thumb = square_thumb(p, size=cell)
        canvas.paste(thumb, (x, y))

    if len(image_paths) < 64:
        draw = ImageDraw.Draw(canvas)
        for idx in range(len(image_paths), 64):
            r, c = divmod(idx, n_cols)
            x = gap + c * (cell + gap)
            y = gap + r * (cell + gap)
            draw.rectangle(
                [x, y, x + cell, y + cell],
                outline=(200, 206, 222),
                width=1,
            )
    canvas.save(out_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 8x8 grids (top-weight images) for 15 dimensions."
    )
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--n-dims", type=int, default=15)
    parser.add_argument("--n-images", type=int, default=64)
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.ffadims.data import get_thingsimages_fnames

    embedding = np.load(args.embedding)
    image_fnames = get_thingsimages_fnames()

    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got shape {embedding.shape}")

    if embedding.shape[1] != len(image_fnames):
        if embedding.shape[0] == len(image_fnames):
            embedding = embedding.T
        else:
            raise ValueError(
                f"Incompatible shapes: embedding={embedding.shape}, image_fnames={len(image_fnames)}"
            )

    n_dims = min(args.n_dims, embedding.shape[0])

    STATIC_GRIDS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trials_path = DATA_DIR / "trials.csv"

    rows: list[dict[str, str]] = []
    for dim_idx in range(n_dims):
        weights = embedding[dim_idx]
        order = np.argsort(weights)[::-1]
        selected: list[Path] = []

        for image_idx in order:
            p = Path(image_fnames[image_idx])
            selected.append(p)
            if len(selected) >= args.n_images:
                break

        dim_name = f"dimension_{dim_idx + 1:02d}"
        grid_name = f"d{dim_idx + 1:02d}.png"
        out_grid_path = STATIC_GRIDS_DIR / grid_name
        make_grid(selected, out_grid_path)

        rows.append(
            {
                "dim_id": dim_name,
                "grid_path": f"grids/{grid_name}",
            }
        )
        print(f"Generated grid {dim_idx + 1}/{n_dims}: {grid_name}")

    with trials_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dim_id", "grid_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote trials to {trials_path}")


if __name__ == "__main__":
    main()
