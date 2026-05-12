from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io as skio
from matplotlib.patches import Patch
from pathlib import Path
import matplotlib.colors as mcolors
from math import ceil


def load_image_thumbnail(
    image_path,
    max_size=(128, 128),
):
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        return img


def make_categorical_cmap(n_colors):
    """
    Make a matplotlib cmap for n categories that look distinct from each other in HSV space.
    """
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    hsv_colors = [(hue, 1, 1) for hue in hues]
    rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsv_colors]
    cmap = mcolors.ListedColormap(rgb_colors)
    return cmap


def plot_topk_rows(
    emb: np.ndarray,
    things_fnames: list,
    topk: int = 10,
    imwidth: float = 0.8,
    plot_lowest: bool = False,
):
    fig, rows = plt.subplots(
        nrows=emb.shape[1], ncols=topk, figsize=(imwidth * topk, imwidth * emb.shape[1])
    )
    for dimi, dim in enumerate(emb.T):
        sortinds = np.argsort(dim)
        if not plot_lowest:
            sortinds = np.flip(sortinds)
        ims_sorted = things_fnames[sortinds]
        topk_ims = ims_sorted[:topk]

        row = rows[dimi]
        for a, im in zip(row, topk_ims):
            a.imshow(load_image_thumbnail(im))
            a.set_xticks([])
            a.set_yticks([])
            a.set_axis_off()
        row[0].set_title(f"dim-{dimi+1}")
    return fig


def plot_bothends_k_rows(
    emb: np.ndarray,
    things_fnames: list,
    k: int = 10,
    imwidth: float = 1.0,
    fontsize_dots: int = 20,
):
    nrows = emb.shape[1]
    ncols = 2 * k + 1  # leave one blank in the middle
    fig, rows = plt.subplots(nrows, ncols, figsize=(imwidth * ncols, imwidth * nrows))
    # each row is a dimension
    for dimi, (dim, row) in enumerate(zip(emb.T, rows)):
        # sort images by dimension weights
        sortinds = np.flip(np.argsort(dim))
        ims_sorted = things_fnames[sortinds]
        # find positive and negative examples
        pos_fs, neg_fs = ims_sorted[:k], ims_sorted[-k:]
        # divide axis objects in this row for positive and negative examples
        pos_axs, neg_axs = row[:k], row[-k:]
        # add blank image inbetween positive and negative examples
        row[k].text(0.5, 0.5, f"...", ha="center", va="center", fontsize=fontsize_dots)
        row[k].axis("off")
        # plot actual images examples
        for fs, axs in zip([pos_fs, neg_fs], [pos_axs, neg_axs]):
            for f, a in zip(fs, axs):
                a.imshow(load_image_thumbnail(f))
                a.axis("off")
        # row titles
        row[0].set_title(f"dim-{dimi+1}: highest")
        row[-k].set_title(f"dim-{dimi+1}: lowest")
    return fig


def plot_topk_separate(
    embedding: np.ndarray,
    things_fnames: list,
    outf_basename: Path,
    topk: int = 500,
    plot_ncols: int = 25,
    plot_lowest: int = False,
):
    ndims = embedding.shape[1]
    inds = np.argsort(embedding, axis=0)
    if not plot_lowest:
        inds = np.flip(inds, axis=0)
    for dimi in range(ndims):
        impaths = things_fnames[inds[:topk, dimi]]
        fig = plot_list_of_images(impaths, use_thumbnail=True, plot_ncols=plot_ncols)
        fname = outf_basename + f"_dim-{dimi+1}.jpg"
        fig.savefig(fname)
    plt.close()
    return None


def plot_reliability(
    reliability_per_dim: np.ndarray,
    n_pruned: int,
    n_embeddings: int,
    out_dir: str,
    outf: str,
) -> None:
    """Plot the reliability of each dimension across all models."""
    ndim = len(reliability_per_dim)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(ndim), reliability_per_dim)
    ax.set_xlabel("Dimension number")
    ax.set_ylabel("Reproducibility of dimensions (Pearson's r)".format(n_embeddings))
    ax.set_title(f"{outf}")
    ax.axvline(x=n_pruned, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, outf),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def plot_list_of_images(
    img_paths: list,
    plot_imwdith: int = 2,
    plot_ncols: int = 10,
    use_thumbnail: bool = True,
):
    """quickly plot a images in a grid with fixed column number and variable number of rows"""
    plot_nrows = ceil(len(img_paths) / plot_ncols)
    fig, axs = plt.subplots(
        ncols=plot_ncols,
        nrows=plot_nrows,
        figsize=(plot_ncols * plot_imwdith, plot_nrows * plot_imwdith),
    )
    for a, f in zip(axs.flatten(), img_paths):
        im = load_image_thumbnail(f) if use_thumbnail else skio.imread(f)
        a.imshow(im)
    # separate loop disables all axes even if `len(img_paths)` is not divisible by `plot_ncols`
    for a in axs.flatten():
        a.axis("off")
    return fig


def plot_facebars(
    emb,
    faces_onehot,
    figwidth=7,
    rowheight=2,
    labelcolors=("red", "lightgrey"),
    bar_kws={"edgecolor": "none", "width": 1.0, "align": "edge"},
):
    nims, ndims = emb.shape
    fig, rows = plt.subplots(
        ndims, 1, figsize=(figwidth, rowheight * ndims), sharex=True
    )
    for dimi, (dim, row) in enumerate(zip(emb.T, rows)):
        # sort by weight
        sortinds = np.flip(np.argsort(dim))
        weights_sorted, faces_sorted = dim[sortinds], faces_onehot[sortinds]
        # colors for labels
        colors = [
            labelcolors[0] if label == True else labelcolors[1]
            for label in faces_sorted
        ]
        # plot
        row.bar(np.arange(nims), weights_sorted, color=colors, **bar_kws)
        row.set_title(f"dim-{dimi+1}", loc="center")
    # legend
    legend_handles = [
        Patch(facecolor=c, label=l) for c, l in zip(labelcolors, ("face", "not face"))
    ]
    rows[0].legend(handles=legend_handles)
    plt.xlim(0, nims)
    return fig


def plot_facebins(
    emb,
    faces_onehot,
    num_bins=100,
    figwidth=7,
    rowheight=2,
    title_y=0.8,
    bar_kws={"color": "grey", "edgecolor": "none", "width": 1.0, "align": "edge"},
):
    nims, ndims = emb.shape
    bin_size = nims // num_bins

    fig, rows = plt.subplots(
        ndims, 1, figsize=(figwidth, rowheight * ndims), sharex=True, sharey=True
    )
    for dimi, (dim, row) in enumerate(zip(emb.T, rows)):
        # sort by weight
        sortinds = np.flip(np.argsort(dim))
        weights_sorted, faces_sorted = dim[sortinds], faces_onehot[sortinds]
        # count faces in each bin
        bin_counts = np.zeros(num_bins, dtype=int)
        for i in range(num_bins):
            start_index = i * bin_size
            end_index = start_index + bin_size
            bin_counts[i] = np.sum(faces_sorted[start_index:end_index])
        # plot
        row.bar(np.arange(num_bins), bin_counts, **bar_kws)
        row.set_title(f"dim-{dimi+1}", loc="center", y=title_y)
        row.set_ylabel("count(faces)")
    rows[-1].set_xlabel("ranked images (binned)")
    return fig
