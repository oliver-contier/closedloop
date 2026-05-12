"""
Plot top 6 images per dimension from top_images_per_dimension.csv as a mosaic.
Layout: 15 dimension panels in 3 rows x 5 columns; each panel has label + 2x3 image grid at original resolution.
Saves to plots/highres/mosaic_top_images.pdf.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from os.path import join as pjoin, dirname, abspath

# Paths (script lives in scripts/fmri_experiment/top_images/)
SCRIPT_DIR = dirname(abspath(__file__))
PROJECT_ROOT = dirname(dirname(dirname(SCRIPT_DIR)))
CSV_PATH = pjoin(SCRIPT_DIR, "top_images_per_dimension.csv")
THINGS_IMAGES_DIR = pjoin(PROJECT_ROOT, "data", "things_images", "THINGS", "images")
OUT_PATH = pjoin(PROJECT_ROOT, "plots", "highres", "mosaic_top_images.pdf")

N_TOP = 6
N_DIM_ROWS, N_DIM_COLS = 3, 5
IMG_ROWS, IMG_COLS = 2, 3


def center_crop_square(img: Image.Image) -> Image.Image:
    """Center-crop image to a square to avoid letterboxing gaps between neighbors."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return img.crop((left, top, right, bottom))


def find_image_path(basename):
    """Resolve image basename to full path under THINGS images (any subdir)."""
    pattern = pjoin(THINGS_IMAGES_DIR, "*", basename)
    hits = glob(pattern)
    if not hits:
        return None
    return hits[0]


def main():
    df = pd.read_csv(CSV_PATH, index_col=0)
    # Clean column names (strip trailing comma)
    df.columns = [c.strip().rstrip(",") for c in df.columns]
    dims = list(df.columns)
    assert len(dims) == N_DIM_ROWS * N_DIM_COLS, f"Expected 15 dimensions, got {len(dims)}"

    # One figure: 3 rows of dimension panels, 5 columns.
    # Use an outer gridspec for the 15 panels and an inner subgridspec per panel
    # so we can set hspace/wspace=0 within a panel (no gaps between its 2x3 images)
    fig = plt.figure(figsize=(N_DIM_COLS * IMG_COLS * 1.1, N_DIM_ROWS * (1 + IMG_ROWS) * 1.1))
    outer_gs = fig.add_gridspec(N_DIM_ROWS, N_DIM_COLS, hspace=0.6, wspace=0.4)

    for dim_idx, dim_label in enumerate(dims):
        row_block = dim_idx // N_DIM_COLS
        col_block = dim_idx % N_DIM_COLS

        # Create a subgridspec inside this panel: 1 row for label + 2 rows x 3 cols for images
        panel_spec = outer_gs[row_block, col_block].subgridspec(
            1 + IMG_ROWS, IMG_COLS, hspace=0.0, wspace=0.0
        )

        # Label row spans all columns in this panel
        ax_label = fig.add_subplot(panel_spec[0, :])
        ax_label.text(0.5, 0.5, dim_label, ha="center", va="center", fontsize=10, fontweight="bold")
        ax_label.set_axis_off()
        # Image grid 2x3 with no spacing inside the panel
        for k in range(N_TOP):
            ir, ic = k // IMG_COLS, k % IMG_COLS
            ax = fig.add_subplot(panel_spec[1 + ir, ic])
            fname = df.iloc[k][dim_label]
            if isinstance(fname, str):
                path = find_image_path(fname.strip())
            else:
                path = None
            if path and os.path.isfile(path):
                img = Image.open(path).convert("RGB")
                img = center_crop_square(img)
                ax.imshow(img, aspect="auto")
            else:
                ax.text(0.5, 0.5, fname or "?", ha="center", va="center", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_axis_off()

    os.makedirs(dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
