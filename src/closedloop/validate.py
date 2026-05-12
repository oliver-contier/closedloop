from os.path import join as pjoin
from os.path import pardir, dirname
from closedloop.data import load_data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from closedloop.encoding import FracRidgeVoxelwise
import pandas as pd


def validate_group_embedding(
    emb,
    ncv=100,
    datadir=pjoin(dirname(__file__), pardir, pardir, "data"),
):
    """
    Validate a group embedding obtained from all 4 of Ratan's subjects 
    (either by concatenating their data or aggregating subject-specific embeddings).
    To that end, we predict the amplitude of each dimension individually from the original voxel data of each subject.
    That way, we get a sense of how strongly associated each group dimension is to each individual subject.
    """
    ndims = emb.shape[1]
    subjs = ["1", "2", "3", "4", "all"]
    # results are saved as list of dicts
    scores = []
    # iterate through subjects
    for subj in tqdm(subjs, desc="subjects", leave=True):
        # load voxel data for subject
        X = load_data(subj, datadir)
        # fit ridge regression
        model = FracRidgeVoxelwise(n_splits=ncv, test_size=0.0)
        _, _, r2_test, _, _ = model.tune_and_eval(X, emb)
        for dimi, r2 in enumerate(r2_test):
            scores.append(dict(subj=subj, dim=dimi, r2=r2))
    df = pd.DataFrame(scores)
    # divide by noise ceiling
    for subj in subjs[:-1]:
        for dimi in range(ndims):
            # could use more elegant pandas logic here
            nc_row = df.loc[(df["subj"] == "all") & (df["dim"] == dimi)]
            target_row = df.loc[(df["subj"] == subj) & (df["dim"] == dimi)]
            r2_relative_ = target_row["r2"].to_numpy() / nc_row["r2"].to_numpy()
            r2_relative = r2_relative_[0]
            df.loc[
                (df["subj"] == subj) & (df["dim"] == dimi), "r2_relative"
            ] = r2_relative
    return df


def plot_validation(validation_df, plot_thresh=.7):
    ndims = validation_df["dim"].max() + 1
    plot_df = validation_df[validation_df["subj"] != "all"]
    grid = sns.catplot(
        data=plot_df,
        y="r2_relative",
        x="dim",
        hue="subj",
        kind="point",
        aspect=2,
        join=False,
    )
    plt.axhline(y=plot_thresh, linestyle="--", color='black')
    grid.set(
        xticklabels=np.arange(1, ndims + 1),
        ylabel="$R^2$ (proportional to noise ceiling)",
        xlabel="Dimension",
        ylim=(0, 1),
    )
    return grid
