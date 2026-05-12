from closedloop.validate import validate_group_embedding, plot_validation
from closedloop.plotting import plot_topk_separate, plot_bothends_k_rows, plot_facebins
from os.path import join as pjoin
import os
from closedloop.inspect_spose import ResultsLoader
from closedloop.data import get_thingsimages_fnames
from closedloop.facelabels import load_facelabels
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def _filter_emb_by_val(emb, validation_df, r2_thresh, n_subjs=4):
    eval_df = validation_df.loc[validation_df["subj"] != "all"]
    eval_df["keep"] = eval_df["r2_relative"] > r2_thresh
    sums = eval_df.groupby("dim")["keep"].sum()
    mask = np.array(sums == n_subjs)
    emb_filtered = emb[:, mask]
    return emb_filtered


def main(args):
    plotdir = pjoin(
        args.plot_basedir,
        f"subj-{args.subj}_relthresh-{args.rel_thresh:.2f}_k-{args.k}",
    )
    outdir = pjoin(
        args.out_basedir,
        f"subj-{args.subj}_relthresh-{args.rel_thresh:.2f}_k-{args.k}",
    )
    for d in [plotdir, outdir]:
        if not os.path.exists(d):
            os.makedirs(d)
    # Load embedding
    rl = ResultsLoader()
    if args.subj == "inter":
        emb = rl.load_groupavg_emb(rel_thresh=args.rel_thresh)
    else:
        emb = rl.load_consensus_emb(
            subj=args.subj, rel_thresh=args.rel_thresh, k=args.k
        )
    # Run validation and save results
    validation_df = validate_group_embedding(emb, ncv=args.ncv)
    validation_df.to_csv(pjoin(outdir, "pred_dims_from_subjs.csv"))

    # save filtered embedding
    # only keep dimension for which all subjects exceed r-2 threshold
    emb_filtered = _filter_emb_by_val(emb, validation_df, args.r2_thresh)
    np.save(pjoin(outdir, "emb_filtered.npy"), emb_filtered)
    # Make plot and save
    grid = plot_validation(validation_df, plot_thresh=args.r2_thresh)
    grid.savefig(pjoin(plotdir, "pred_dims_from_subjs.pdf"))

    print("plot top 500 images for each dimension")
    things_fnames = get_thingsimages_fnames()
    fig = plot_bothends_k_rows(emb_filtered, things_fnames, k=15)
    fig.savefig(pjoin(plotdir, f"sub-{args.subj}_top15.png"))
    outf_basename = pjoin(plotdir, f"sub-{args.subj}_top500")
    _ = plot_topk_separate(
        emb_filtered,
        things_fnames,
        outf_basename,
        topk=500,
        plot_ncols=25,
        plot_lowest=False,
    )

    print("plot face bins")
    _, _, _, faces_onehot = load_facelabels()
    fig = plot_facebins(emb_filtered, faces_onehot)
    fig.savefig(pjoin(plotdir, f"sub-{args.subj}_facebins.png"))

    # get correlation between dimensions
    corrs = np.corrcoef(emb_filtered.T)
    corrs[np.triu_indices_from(corrs)] = np.nan
    # plot
    ticklabels = np.arange(1, corrs.shape[0] + 1)
    fig = plt.figure(figsize=(13, 13))
    ax = sns.heatmap(
        corrs,
        square=True,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        vmax=1,
        vmin=-1,
        center=0,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        cbar=True,
    )
    plt.savefig(pjoin(plotdir, f"sub-{args.subj}_corrs.png"))
    return None


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj",
        type=str,
        default="all",
        help="""
        subject number. if 1-4, all, or pooled is chosen, this will use the consensus embedding approach. 
        If inter is chosen, this will use the inter-subject consistency approach like in Khosla et al.
        """,
        choices=["1", "2", "3", "4", "all", "pooled", "inter"],
    )
    parser.add_argument(
        "--k", type=int, required=True, help="number of (consensus) dimensions"
    )
    parser.add_argument(
        "--r2_thresh",
        type=float,
        default=0.6,
        help="""
        Threshold for filtering dimensions based on their predictivity by individual subjects. 
        Only those dimensions will be kept which can be predicted with at least this R-squared value 
        by each individual subject.""",
    )
    parser.add_argument(
        "--rel_thresh",
        type=float,
        default=0.6,
        help="reliability threshold for loaded consensus embeddings. Only relevant for defining which data to load.",
    )
    parser.add_argument(
        "--plot_basedir",
        type=str,
        default="../../plots/validation",
        help="directory to save plots",
    )
    parser.add_argument(
        "--out_basedir",
        type=str,
        default="../../results/validation",
        help="directory to save results data frame",
    )
    parser.add_argument(
        "--ncv",
        type=int,
        default=100,
        help="number of cross-validation folds used in prediction",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
