import h5py
import numpy as np
from os.path import join as pjoin
from glob import glob
from os.path import pardir, dirname, exists
import pandas as pd
from os import makedirs


def load_data(
    subj="all",
    datadir=pjoin(dirname(__file__), pardir, pardir, "data"),
):
    """Load preprocessed voxel-wise response predictions for all THINGS images"""
    valid_subjs = ("1", "2", "3", "4")
    assert subj in list(valid_subjs) + ["all"]
    if subj == "all":
        data = np.hstack(
            [
                np.hstack(list(load_data_raw(subj=subj, datadir=datadir)))
                for subj in valid_subjs
            ]
        )
    else:
        lffa, rffa = load_data_raw(subj=subj, datadir=datadir)
        data = np.hstack([lffa, rffa])
    return data


def load_data_raw(subj="1", datadir=pjoin(dirname(__file__), pardir, pardir, "data")):
    """Load raw data"""
    lffa_h5 = pjoin(datadir, "raw", f"s{subj}_lffa.h5")
    rffa_h5 = pjoin(datadir, "raw", f"s{subj}_lffa.h5")
    lffa = read_h5(lffa_h5)
    rffa = read_h5(rffa_h5)
    return lffa, rffa


def read_h5(fname):
    """
    Returns array of shape (nimages x nvoxel)
    """
    with h5py.File(fname, "r") as f:
        a_group_key = list(f.keys())[0]
        arr = f[a_group_key][()]
    return arr


def remove_outlier_features(data, simthresh=0.6):
    """data should be (images x voxels)"""
    # voxelXvoxel correlation
    s = np.corrcoef(data.T)
    # ignore diagonal when calculating means
    imask = np.eye(s.shape[0], dtype=bool)
    s[imask] = np.nan
    means = np.nanmean(s, 0)
    # only keep features with mean similarity above threshold
    valinds = means >= simthresh
    cleaned = data[:, valinds]
    return cleaned


def get_thingsimages_fnames(
    things_dir=pjoin(
        dirname(__file__), pardir, pardir, "data", "things_images", "THINGS", "images"
    )
):
    fs = glob(pjoin(things_dir, "*", "*.jpg"))
    assert fs
    fs.sort()
    return np.array(fs, dtype=str)


def npsave_mkdir(arr, fname):
    """Save array to npy file, creating directory if it doesn't exist"""
    if not exists(dirname(fname)):
        makedirs(dirname(fname))
    np.save(fname, arr)
    return None


def trip_npy2txt(tripdir):
    npys = glob(pjoin(tripdir, "*.npy"))
    for npy in npys:
        d = np.load(npy)
        np.savetxt(npy.replace(".npy", ".txt"), d, fmt="%d")
    return None


class ThingsmriLoader:
    def __init__(self, thingsmri_dir=pjoin(pardir, "data", "thingsmri")):
        self.thingsmri_dir = thingsmri_dir
        self.betas_dir = pjoin(thingsmri_dir, "betas_csv")
        self.brainmasks_dir = pjoin(thingsmri_dir, "brainmasks")

    def load_responses(self, subject):
        stimdata = pd.read_csv(
            pjoin(self.betas_dir, f"sub-{subject}_StimulusMetadata.csv")
        )
        voxdata = pd.read_csv(pjoin(self.betas_dir, f"sub-{subject}_VoxelMetadata.csv"))
        responses = pd.read_hdf(pjoin(self.betas_dir, f"sub-{subject}_ResponseData.h5"))
        return responses, stimdata, voxdata

    def get_brainmask(self, subject):
        return pjoin(self.brainmasks_dir, f"sub-{subject}_space-T1w_brainmask.nii.gz")
