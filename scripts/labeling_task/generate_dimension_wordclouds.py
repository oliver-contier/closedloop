from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.image import imread
import simplemma

try:
    from wordcloud import WordCloud
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise ImportError(
        "Missing dependency 'wordcloud'. Install it with:\n"
        "  source .venv/bin/activate && pip install wordcloud"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "labeling_task_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots" / "labeling_task_wordclouds"
DEFAULT_EMBEDDING_PATH = PROJECT_ROOT / "data" / "embedding" / "emb_filtered.npy"
FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
]


TERM_NORMALIZATION = {
    "faces": "face",
    "facial": "face",
    "humans": "human",
    "heads": "head",
    "trees": "tree",
    "clothes": "clothing",
    "bodz": "body",
    "rectangualr": "rectangular",
    "centered": "center",
    "centred": "center",
    "verticals": "vertical",
    "roundness": "round",
}

STOP_TERMS = {
    "image",
    "images",
    "picture",
    "pictures",
    "thing",
    "things",
    "object",
    "objects",
    "one",
    "two",
    "three",
    "and",
    "the",
    "a",
    "an",
}


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_latest_labels_all_tokens(data_dir: Path) -> pd.DataFrame:
    responses = pd.read_csv(data_dir / "responses_long.csv")

    if responses.empty:
        raise ValueError("No responses found in responses_long.csv.")

    responses["updated_at"] = _to_datetime(responses["updated_at"])
    responses = responses.sort_values("updated_at").dropna(subset=["updated_at"])

    # Keep latest entry per token x dimension, regardless of submission state.
    final_df = responses.drop_duplicates(subset=["token", "dim_id"], keep="last")
    final_df = final_df[["token", "dim_id", "trial_index", "label", "updated_at"]].copy()
    final_df["label"] = final_df["label"].fillna("").astype(str)
    final_df = final_df[final_df["label"].str.strip().astype(bool)].copy()
    if final_df.empty:
        raise ValueError("No non-empty labels found in responses_long.csv.")
    return final_df.sort_values(["dim_id", "token"])


def normalize_term(term: str) -> str | None:
    term = term.lower().strip()
    term = term.replace("_", " ")
    term = re.sub(r"[\(\)\[\]\{\}\"'`]", "", term)
    term = re.sub(r"[^a-z\s\-]", " ", term)
    term = re.sub(r"\s+", " ", term).strip()
    if not term:
        return None

    # Lemmatize word-by-word to reduce inflection variants (e.g., bodies -> body).
    term = " ".join(simplemma.lemmatize(tok, lang="en") for tok in term.split())

    term = TERM_NORMALIZATION.get(term, term)
    if term in STOP_TERMS:
        return None
    if len(term) < 3:
        return None

    letters = re.sub(r"[^a-z]", "", term)
    if letters:
        vowels = sum(ch in "aeiou" for ch in letters)
        if len(letters) >= 5 and vowels / len(letters) < 0.15:
            return None
    return term


def split_label_to_terms(label: str) -> list[str]:
    rough_terms = re.split(r"[,;/\|]+", label)
    cleaned: list[str] = []
    for term in rough_terms:
        norm = normalize_term(term)
        if norm:
            cleaned.append(norm)
    return cleaned


def build_term_counts(final_df: pd.DataFrame) -> tuple[dict[str, Counter], pd.DataFrame]:
    rows = []
    by_dim: dict[str, Counter] = {}
    for dim_id, sub_df in final_df.groupby("dim_id"):
        counter: Counter = Counter()
        for label in sub_df["label"]:
            counter.update(split_label_to_terms(label))
        by_dim[dim_id] = counter
        for term, count in counter.most_common():
            rows.append({"dim_id": dim_id, "term": term, "count": count})
    counts_df = pd.DataFrame(rows)
    return by_dim, counts_df


def build_color_kwargs(colormap: str | None) -> dict:
    if colormap:
        return {"colormap": colormap}
    return {"color_func": lambda *args, **kwargs: "black"}


def resolve_font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if path.exists():
            return str(path)
    return None


def get_thingsimages_fnames() -> np.ndarray:
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from closedloop.data import get_thingsimages_fnames as _get_thingsimages_fnames

    return _get_thingsimages_fnames()


def plot_single_cloud(
    freqs: Counter,
    title: str,
    out_path: Path,
    max_words: int,
    colormap: str | None,
    font_path: str | None,
) -> None:
    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        max_words=max_words,
        collocations=False,
        font_path=font_path,
        **build_color_kwargs(colormap),
    ).generate_from_frequencies(dict(freqs))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_wordcloud_image(
    freqs: Counter,
    max_words: int,
    colormap: str | None,
    font_path: str | None,
    width: int = 1000,
    height: int = 600,
) -> np.ndarray:
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        max_words=max_words,
        collocations=False,
        font_path=font_path,
        **build_color_kwargs(colormap),
    ).generate_from_frequencies(dict(freqs))
    return wc.to_array()


def plot_grid_clouds(
    term_counts: dict[str, Counter],
    n_raters: dict[str, int],
    out_path: Path,
    max_words: int,
    colormap: str | None,
    font_path: str | None,
) -> None:
    dim_ids = sorted(term_counts.keys())
    n = len(dim_ids)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 4))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, dim_id in enumerate(dim_ids):
        ax = axes_flat[i]
        freqs = term_counts[dim_id]
        if freqs:
            wc_img = make_wordcloud_image(
                freqs=freqs,
                max_words=max_words,
                colormap=colormap,
                font_path=font_path,
                width=900,
                height=600,
            )
            ax.imshow(wc_img, interpolation="bilinear")
        ax.set_title(f"{dim_id} (n={n_raters.get(dim_id, 0)})", fontsize=11)
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_wordcloud_plus_top_images_overview(
    term_counts: dict[str, Counter],
    n_raters: dict[str, int],
    embedding_path: Path,
    out_path: Path,
    max_words: int,
    top_k_images: int = 8,
    wordcloud_colormap: str = "viridis",
    font_path: str | None = None,
) -> None:
    embedding = np.load(embedding_path)
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {embedding.shape}")

    image_fnames = np.array(get_thingsimages_fnames(), dtype=str)
    if embedding.shape[1] != len(image_fnames):
        if embedding.shape[0] == len(image_fnames):
            embedding = embedding.T
        else:
            raise ValueError(
                "Embedding shape does not match THINGS image list. "
                f"Embedding shape={embedding.shape}, n_images={len(image_fnames)}"
            )

    dim_ids = sorted(term_counts.keys())
    if embedding.shape[0] < len(dim_ids):
        raise ValueError(
            f"Embedding has {embedding.shape[0]} dims but labeling has {len(dim_ids)} dims."
        )

    n_rows = len(dim_ids)
    n_cols = 1 + top_k_images
    width_ratios = [3.8] + [1.0] * top_k_images
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(2.0 * sum(width_ratios), 2.0 * n_rows),
        gridspec_kw={"width_ratios": width_ratios},
    )

    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, dim_id in enumerate(dim_ids):
        freqs = term_counts[dim_id]
        cloud_ax = axes[row_idx, 0]
        if freqs:
            wc_img = make_wordcloud_image(
                freqs=freqs,
                max_words=max_words,
                colormap=wordcloud_colormap,
                font_path=font_path,
            )
            cloud_ax.imshow(wc_img, interpolation="bilinear")
        cloud_ax.set_title(f"{dim_id} (n={n_raters.get(dim_id, 0)})", fontsize=10)
        cloud_ax.axis("off")

        dim_idx = int(dim_id.split("_")[-1]) - 1
        top_inds = np.argsort(embedding[dim_idx])[-top_k_images:][::-1]
        top_paths = image_fnames[top_inds]

        for col_offset, img_path in enumerate(top_paths, start=1):
            ax = axes[row_idx, col_offset]
            img = imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"Top {col_offset}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-dimension word clouds from labeling task outputs."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--embedding-path", type=Path, default=DEFAULT_EMBEDDING_PATH)
    parser.add_argument("--top-k-images", type=int, default=8)
    parser.add_argument(
        "--colormap",
        type=str,
        default=None,
        help=(
            "Matplotlib colormap name (e.g., turbo, cubehelix, magma). "
            "Default is black text on white."
        ),
    )
    args = parser.parse_args()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = resolve_font_path()

    final_df = load_latest_labels_all_tokens(data_dir)

    n_tokens = final_df["token"].nunique()
    n_dims = final_df["dim_id"].nunique()
    max_expected_rows = n_tokens * n_dims
    if len(final_df) < max_expected_rows:
        print(
            f"Info: Found {len(final_df)} token x dimension labels across {n_tokens} tokens "
            f"(max possible for observed dims: {max_expected_rows})."
        )

    term_counts, counts_df = build_term_counts(final_df)
    raters_per_dim = final_df.groupby("dim_id")["token"].nunique().to_dict()

    for dim_id, freqs in term_counts.items():
        if not freqs:
            continue
        plot_single_cloud(
            freqs=freqs,
            title=f"{dim_id} (n={raters_per_dim.get(dim_id, 0)})",
            out_path=output_dir / f"{dim_id}_wordcloud.pdf",
            max_words=args.max_words,
            colormap=args.colormap,
            font_path=font_path,
        )

    plot_grid_clouds(
        term_counts=term_counts,
        n_raters=raters_per_dim,
        out_path=output_dir / "all_dimensions_wordclouds.png",
        max_words=args.max_words,
        colormap=args.colormap,
        font_path=font_path,
    )
    plot_wordcloud_plus_top_images_overview(
        term_counts=term_counts,
        n_raters=raters_per_dim,
        embedding_path=args.embedding_path.resolve(),
        out_path=output_dir / "all_dimensions_wordclouds_plus_top8_images.png",
        max_words=args.max_words,
        top_k_images=args.top_k_images,
        wordcloud_colormap=None,
        font_path=font_path,
    )

    counts_df = counts_df.sort_values(
        ["dim_id", "count", "term"], ascending=[True, False, True]
    )
    counts_df.to_csv(output_dir / "term_counts_by_dimension.csv", index=False)
    final_df.to_csv(output_dir / "final_submitted_labels_reconstructed.csv", index=False)

    print(f"Tokens with labels: {n_tokens}")
    print(f"Dimensions found: {n_dims}")
    print(f"Final rows used: {len(final_df)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
