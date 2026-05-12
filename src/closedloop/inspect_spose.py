from glob import glob
import os
from os.path import join as pjoin
from os.path import basename, dirname, pardir, normpath, abspath
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from closedloop.utils import _dictinfo_to_string
import toml


def _parse_epochs_ran_CPP(results_dir):
    jsons = glob(pjoin(results_dir, "*.json"))
    epochs = [
        int(basename(j).replace(".json", "").replace("results_epoch", ""))
        for j in jsons
    ]
    return max(epochs)


def load_perf(results_dir):
    epochs_ran = _parse_epochs_ran_CPP(results_dir)
    jdicts = []
    for epoch in range(5, epochs_ran + 1, 5):
        jfile = pjoin(results_dir, f"results_epoch{epoch:04d}.json")
        with open(jfile, "r") as f:
            jdicts.append(json.load(f))
    perf_df = pd.DataFrame(jdicts)
    return perf_df


def load_latest_embedding(results_dir):
    epochs_ran = _parse_epochs_ran_CPP(results_dir)
    emb_txt = pjoin(results_dir, f"weights_sorted_epoch{epochs_ran:04d}.txt")
    return np.loadtxt(emb_txt)


def get_all_subdirs_CPP(results_dir):
    """
    assumes folder structure:
    /path/to/results/cpp/simthresh-*_trans-*_tripdist-*/sposedist-*_lmdb-*/subj-*
    """
    txts = glob(pjoin(results_dir, "*", "*", "*", "*.txt"))
    return np.unique([dirname(txt) for txt in txts])


class ResultsLoaderCPP:
    def __init__(self):
        self.results_basedir = pjoin(dirname(__file__), pardir, pardir, "results")
        self.results = {}
        self.relevant_params = [
            "simthresh",
            "trans",
            "tripdist",
            "sposedist",
            "lmdb",
            "subj",
        ]

    def _parse_subdir(self, dirname):
        paramdict = {}
        dirstrs = dirname.split("/")
        for dirstr in dirstrs:
            relevant = False
            for p in self.relevant_params:
                if p in dirstr:
                    relevant = True
            if not relevant:
                continue
            tmpdict = {}
            els = dirstr.split("_")
            for el in els:
                k, v = el.split("-")
                tmpdict[k] = v
            paramdict.update(tmpdict)
        return paramdict

    def load_all_results_CPP(self):
        results_dirs = glob(pjoin(self.results_basedir, "*"))
        perf, embs, embinfo = [], [], []
        for results_dir in results_dirs:
            imp = basename(results_dir)
            subdirs = get_all_subdirs_CPP(results_dir)
            for subdir in tqdm(subdirs, desc="loading results subdirectories"):
                # load model parameters and performance
                params = self._parse_subdir(subdir)
                perf_ = load_perf(subdir)
                perf_["implementation"] = imp
                for k, v in params.items():
                    perf_[k] = v
                perf.append(perf_)
                # load embedding
                embs.append(load_latest_embedding(subdir))
                embinfo.append(perf_.copy())
        perf = pd.concat(perf)
        for c in ["lmdb", "simthresh"]:  # convert from string to float
            perf[c] = perf[c].astype(float)
        return perf, embs, embinfo


class ResultsLoader:
    """
    Load deep_embeddings results
    """

    def __init__(
        self,
        results_basedir=pjoin(dirname(__file__), pardir, pardir, "results"),
    ):
        self.results_basedir = abspath(results_basedir)
        print(f"results directory set: {self.results_basedir}")
        self.consensus_basedir = pjoin(self.results_basedir, "consensus")
        self.groupavg_dir = pjoin(self.consensus_basedir, "group")
        self.encoding_dir = pjoin(self.results_basedir, "encoding")
        self.subjects = ("1", "2", "3", "4", "all")

    def get_all_resultsdirs(self):
        drs = glob(
            pjoin(
                self.results_basedir,
                "deep_embeddings",
                f"simthresh-*_trans-*_tripdist-*",
                "*",
                "*",
                f"subj-*",
                "behavior",
                f"*.mio",
                "uniform",
                f"*",
                f"*",
                "*",
                "*",
            )
        )
        return drs

    def params_from_resultsdir(self, resultsdir):
        params = {}
        # get parameters not saved in deep_embeddings config
        pathels = normpath(resultsdir).split(os.sep)
        for el in pathels:
            if "simthresh" in el or "subj" in el:
                subels = el.split("_")
                for subel in subels:
                    k, v = subel.split("-")
                    params[k] = v
            if ".mio" in el:
                params["ntrips_mio"] = int(el.replace(".mio", ""))
            if "sposedist" in el:
                params["sposedist"] = el.split("_")[0].split("-")[-1]
        # get deep_embeddings config
        conf_f = pjoin(resultsdir, "config.toml")
        params.update(toml.load(conf_f))
        return params

    def parse_epoch(self, embfile):
        return int(embfile.split("/")[-1].split("_")[-1].replace(".npz", ""))

    def get_embfile_and_last_epoch(self, resultsdir):
        npzs = glob(pjoin(resultsdir, "params", "params_epoch_*.npz"))
        if not npzs:
            return None, 0
        else:
            epochs = [self.parse_epoch(n) for n in npzs]
            sortinds = np.argsort(epochs)
            lastind = sortinds[-1]
            return npzs[lastind], epochs[lastind]

    def get_consensus_dir(self, subj, rel_thresh, k):
        return pjoin(
            self.results_basedir,
            "consensus",
            "subject",
            f"subj-{subj}",
            f"rel_thresh-{rel_thresh:.2f}",
            f"k-{k}",
        )

    def load_results(self, filterdict=None):
        resultsdirs = self.get_all_resultsdirs()
        results = []
        for rd in resultsdirs:
            result = self.params_from_resultsdir(rd)
            result["embfile"], result["epoch"] = self.get_embfile_and_last_epoch(rd)
            results.append(result)
        if filterdict:
            results = [
                r for r in results if np.all([r[k] == v for k, v in filterdict.items()])
            ]
        return results

    def load_consensus_emb(self, subj, rel_thresh, k):
        consensus_dir = self.get_consensus_dir(subj, rel_thresh, k)
        emb_npy = pjoin(consensus_dir, f"sub-{subj}_consensus_embedding.npy")
        emb = np.load(emb_npy).T
        return emb

    def load_groupavg_emb(self, rel_thresh=0.6):
        groupavg_npy = pjoin(
            self.results_basedir,
            "consensus",
            "group",
            f"rel_thresh-{rel_thresh:.2f}",
            "group_avg.npy",
        )
        groupavg = np.load(groupavg_npy)
        return groupavg

    def load_validated_embedding(self, subj, rel_thresh, k):
        validated_npy = pjoin(
            self.results_basedir,
            "validation",
            f"subj-{subj}_relthresh-{rel_thresh:.2f}_k-{k}",
            "emb_filtered.npy",
        )
        validated_emb = np.load(validated_npy)
        return validated_emb

    def _encodingfiles_from_dir(
        self,
        resdir,
        thingsmri_sub,
    ):
        best_fracs_f = pjoin(resdir, f"sub-{thingsmri_sub}_best_fracs.nii.gz")
        betas_f = pjoin(resdir, f"sub-{thingsmri_sub}_betas.nii.gz")
        r2_test_f = pjoin(resdir, f"sub-{thingsmri_sub}_r2_test.nii.gz")
        r2_train_f = pjoin(resdir, f"sub-{thingsmri_sub}_r2_train.nii.gz")
        files_dict = dict(
            best_fracs=best_fracs_f,
            betas=betas_f,
            r2_test=r2_test_f,
            r2_train=r2_train_f,
        )
        return files_dict

    # def load_encoding_files_filterdict(
    #     self,
    #     filterdict,
    #     thingsmri_sub,
    # ):
    #     resdir = pjoin(self.encoding_dir, _dictinfo_to_string(filterdict))
    #     files_dict = self._encodingfiles_from_dir(resdir, thingsmri_sub)
    #     return files_dict

    def load_encoding_files_groupavgdims(
        self,
        thingsmri_sub,
    ):
        resdir = pjoin(self.encoding_dir, "groupavgdims", f"sub-{thingsmri_sub}")
        files_dict = self._encodingfiles_from_dir(resdir, thingsmri_sub)
        return files_dict
