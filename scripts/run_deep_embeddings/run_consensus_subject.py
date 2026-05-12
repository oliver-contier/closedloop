from closedloop.inspect_spose import ResultsLoader
from closedloop.consensus import (
    split_half_reliability,
    make_npruned_dict,
    find_mean_reliability,
    cluster_evaluate_ks,
)
from closedloop.plotting import (
    plot_reliability,
    plot_topk_rows,
    plot_bothends_k_rows,
    plot_facebins,
    plot_list_of_images,
    make_categorical_cmap,
    plot_topk_separate,
)
from closedloop.data import get_thingsimages_fnames
from closedloop.facelabels import load_facelabels
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE, MDS
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
import warnings
import os
from os.path import join as pjoin
import json
from colorcet import glasbey
import argparse
import matplotlib.colors as mcolors


class ConfigFinalK:
    """Number of clusters (k) used to derive the group-level consensus embedding.

    The paper-reported analysis used the 4-subject pooled in-silico data (subj='all')
    and k=17 median cluster centroids (cf. Methods: Reproducible group-level embedding).
    """

    def __init__(self):
        self.subj1 = 13
        self.subj2 = 10
        self.subj3 = 10
        self.subj4 = 16
        self.subjall = 17
        self.subjpooled = 17


def _make_outdirs(args, final_k):
    out_dir = pjoin(
        args.out_basedir,
        f"subj-{args.subj}",
        f"rel_thresh-{args.rel_thresh:.2f}",
        f"k-{final_k}",
    )
    plot_dir = pjoin(
        args.plot_basedir,
        f"subj-{args.subj}",
        f"rel_thresh-{args.rel_thresh:.2f}",
        f"k-{final_k}",
    )
    for d in [out_dir, plot_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    return out_dir, plot_dir


def _fit_gmm(rel_df, gmm_kws, p=0.95):
    gmm = BayesianGaussianMixture(**gmm_kws)
    gmm.fit(rel_df.reliability.to_numpy().reshape(-1, 1))
    relrange = np.linspace(0, 1, 1000)
    probas = gmm.predict_proba(relrange.reshape(-1, 1))
    # which label has higher values on x? -> signal
    signal_clust_i = np.argmax(gmm.means_)
    # what reliability value is 95% prob to be in signal distribution?
    thresh_thresh_ind = np.where(probas[:, signal_clust_i] > p)[0].min()
    rel_thresh = relrange[thresh_thresh_ind]
    return rel_thresh


def main(
    args,
    ks=list(range(2, 35)) + list(range(35, 71, 5)),
    gmm_kws=dict(
        n_components=2, random_state=0, max_iter=1_000, covariance_type="tied"
    ),
):
    ## parameters for gaussian mixture model used to filter out unreproducible dimensions
    things_fnames = get_thingsimages_fnames()
    final_k = ConfigFinalK().__dict__[f"subj{args.subj}"]
    # output directories
    print("creating output directories")
    out_dir, plot_dir = _make_outdirs(args, final_k)

    print("load embeddings")
    rl = ResultsLoader()
    filterdict = {"ntrips_mio": args.ntrips_mio, "beta": args.lmdb, "subj": args.subj}
    if args.subj == "pooled":
        # pooled means we load embeddings from all individual subjects
        results = []
        for subj in range(1, 5):
            filterdict["subj"] = f"{subj}"
            results += rl.load_results(filterdict)
    else:
        results = rl.load_results(filterdict)
    print(f"found {len(results)} embeddings")
    embeddings_l = [np.load(res["embfile"])["weights"] for res in results]
    embeddings = np.stack(embeddings_l)
    identifiers = [
        f'subj-{res["subj"]}_lmdb-{res["beta"]:.4f}_seed-{res["seed"]}'
        for res in results
    ]

    print("compute reliability of each dimension in each embedding")
    reliability_per_dim = split_half_reliability(embeddings, identifiers)

    print("plot reliability per dim separately for each seed")
    for res in results:
        ident = f'subj-{res["subj"]}_lmdb-{res["beta"]:.4f}_seed-{res["seed"]}'
        outf = f"{ident}_reliabilities.png"
        n_pruned = np.load(res["embfile"])["pruned_weights"].shape[1]
        dimrels = reliability_per_dim[ident]
        plot_reliability(
            dimrels,
            n_pruned,
            embeddings.shape[0],
            plot_dir,
            outf,
        )

    print("get and plot mean reliability per seed")
    n_pruned = make_npruned_dict(
        results
    )  # dict containing n_pruned for each identifier
    mean_reliabilities = find_mean_reliability(reliability_per_dim, n_pruned)
    # identifiers = list(mean_reliabilities.keys())
    meanrels = [mean_reliabilities[ident] for ident in identifiers]
    plt.bar(range(len(identifiers)), meanrels)
    plt.xticks([])
    plt.xlabel("seed")
    plt.title("mean reliability of pruned embedding")
    plt.ylabel("mean(pearson_r)")
    plt.savefig(pjoin(plot_dir, f"sub-{args.subj}_meanrels_per_emb.png"))
    plt.close()

    # collect all seeds in one dataframe
    dfs_ = []
    for ident, rels in reliability_per_dim.items():
        df_ = pd.DataFrame({})
        df_["reliability"] = rels
        df_["ident"] = str(ident)
        dfs_.append(df_)
    rel_df = pd.concat(dfs_)

    if not args.rel_thresh:
        print(
            "Identify reliability threshold based on gaussian mixture model for filtering dimensions"
        )
        rel_thresh = _fit_gmm(rel_df, gmm_kws)
        print("Identified reliability threshold: ", rel_thresh)
    else:
        rel_thresh = args.rel_thresh
        print("Using specified reliability threshold: ", rel_thresh)

    print("plot distribution of reliabilities")
    sns.displot(
        hue="ident", x="reliability", data=rel_df, kind="kde", palette="viridis"
    )
    plt.title("distributions of dimension reliabilities per seed")
    # add vertical line for reliabilith threshold with text label of rel_thresh
    plt.axvline(x=rel_thresh, color="red", linestyle="--")
    plt.text(
        rel_thresh + 0.01,
        0.5,
        f"reliability threshold: {rel_thresh:.2f}",
        rotation=90,
        verticalalignment="center",
    )
    # save
    plt.savefig(pjoin(plot_dir, f"sub-{args.subj}_rels_distplot.png"))
    plt.close()

    print("filter out dimensions below reliability threshold")
    reliability_per_dim.keys()
    embeddings_filtered = []
    for res in results:
        ident = res["ident"]
        rels = reliability_per_dim[ident]
        mask = rels > rel_thresh
        emb = np.load(res["embfile"])["weights"]
        embeddings_filtered.append(emb[:, mask])
    dims_filtered = np.hstack(embeddings_filtered)
    print("kept number of dimensions:", dims_filtered.shape[1])

    print("standardizing dimensions")
    scaler = StandardScaler()  # zscore filtered dimensions
    dims_filtered = scaler.fit_transform(dims_filtered)
    dims_filtered = dims_filtered.T  # we want to cluster dimensions, not objects
    if args.subj == "pooled":
        # apply additional zscoring per image if dimensions are pooled across subjects
        dims_filtered = scaler.fit_transform(dims_filtered)

    print("evaluate clustering")
    if args.clustering_model == "KMeans":
        clustering_model = KMeans()
    elif args.clustering_model == "SpectralClustering":
        clustering_model = SpectralClustering()
    elif args.clustering_model == "AgglomerativeClustering":
        clustering_model = AgglomerativeClustering()
    else:
        raise ValueError("Clustering model not recognized.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores_df = cluster_evaluate_ks(
            dims_filtered,
            ks,
            clustering_model,
        )

    print("plot clustering fit scores")
    metrics = [c for c in scores_df.columns if c != "k"]
    fig, axs = plt.subplots(
        nrows=len(metrics), figsize=(8, 2 * len(metrics)), sharex=True
    )
    for ax, metric in zip(axs, metrics):
        ax.plot(ks, scores_df[metric])
        ax.set_title(metric, y=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(color="gray", linestyle="--")
        ax.axvline(x=final_k, color="red", linestyle="--")
    axs[-1].set_xlabel("k")
    plt.xticks(ks, fontsize="xx-small")
    plt.suptitle(f"Subject {args.subj}")
    plt.tight_layout()
    fig.savefig(pjoin(plot_dir, f"sub-{args.subj}_kmeans_metrics.png"))
    plt.close()

    print("cluster dimensions")
    final_cluster = clustering_model.set_params(n_clusters=final_k)
    labels = final_cluster.fit_predict(dims_filtered)

    print("plot cluster labels in tSNE space")
    # custom_cmap = make_categorical_cmap(final_k)
    custom_cmap = mcolors.ListedColormap(glasbey)
    df_tsne = TSNE(n_components=2, n_jobs=3).fit_transform(dims_filtered)
    plt.scatter(df_tsne[:, 0], df_tsne[:, 1], alpha=0.8, c=labels, cmap=custom_cmap)
    plt.savefig(pjoin(plot_dir, f"sub-{args.subj}_tsne_labeled.png"))
    plt.close()

    print("plot cluster labels in MDS space")
    df_mds = MDS(n_components=2, n_jobs=3).fit_transform(dims_filtered)
    plt.scatter(df_mds[:, 0], df_mds[:, 1], alpha=0.8, c=labels, cmap=custom_cmap)
    plt.savefig(pjoin(plot_dir, f"sub-{args.subj}_mds_labeled.png"))
    plt.close()

    print("taking cluster centroids as new dimensions")
    _, nims = dims_filtered.shape
    consensus = np.zeros((final_k, nims))
    n_singular_clusters = 0  # count clusters with only one sample
    centroid_funcs = {"mean": np.mean, "median": np.median}
    centroid_func = centroid_funcs[args.centroid_estimate]
    for clust_i in range(final_k):
        cluster_is = final_cluster.labels_ == clust_i
        if cluster_is.sum() == 1:
            cluster_center = dims_filtered[cluster_is].squeeze()
            n_singular_clusters += 1
        else:
            cluster_data = dims_filtered[cluster_is]
            cluster_center = centroid_func(cluster_data, axis=0)
        consensus[clust_i] = cluster_center
    print(f"number of singular clusters found: {n_singular_clusters}")

    print("plot consensus embedding")
    # fig = plot_topk_embedding(consensus.T, things_fnames, topk=20)
    fig = plot_bothends_k_rows(consensus.T, things_fnames, k=15)
    fig.savefig(pjoin(plot_dir, f"sub-{args.subj}_topk_imgs.png"))
    plt.close()

    # print("plot top 500 images for each dimension")
    # outf_basename = pjoin(plot_dir, f"sub-{args.subj}_top500")
    # _ = plot_topk_separate(consensus.T, things_fnames, outf_basename, topk=500, plot_ncols=25, plot_lowest=False)

    print("plot distribution of faces across dimensions")
    _, _, _, faces_onehot = load_facelabels()
    fig = plot_facebins(consensus.T, faces_onehot)
    fig.savefig(pjoin(plot_dir, f"sub-{args.subj}_facebins.png"))

    print("save consensus embedding")
    outfn = pjoin(out_dir, f"sub-{args.subj}_consensus_embedding.npy")
    np.save(outfn, consensus)

    print("save all parameters in results dict to text file")
    outconf = pjoin(out_dir, f"sub-{args.subj}_consensus_args.txt")
    args_dict = vars(args)
    args_dict["final_k"] = final_k
    args_dict["rel_thresh"] = rel_thresh
    with open(outconf, "w") as json_f:
        json.dump(args_dict, json_f, indent=4)
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj",
        type=str,
        default="all",
        help="subject number",
    )
    parser.add_argument(
        "--clustering_model",
        type=str,
        default="AgglomerativeClustering",
        help="Clustering model to use. Options: KMeans, SpectralClustering, AgglomerativeClustering",
    )
    parser.add_argument(
        "--centroid_estimate",
        type=str,
        default="median",
        help="How to select the centroid of each dimensions cluster. Options: median, mean",
    )
    parser.add_argument(
        "--rel_thresh",
        type=float,
        default=0.6,
        help="reliability threshold. If set to 0., will be estimated from data based on gaussian mixture model",
    )
    parser.add_argument(
        "--plot_basedir",
        type=str,
        default="../../plots/consensus/subject",
        help="directory to save plots",
    )
    parser.add_argument(
        "--out_basedir",
        type=str,
        default="../../results/consensus/subject",
        help="directory to save embedding",
    )
    parser.add_argument(
        "--ntrips_mio",
        type=int,
        default=4,
        help="number of triplets (in millions)",
    )
    parser.add_argument(
        "--lmdb",
        type=float,
        default=0.0003,
        help="lambda parameter (L1 penalty beta)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
