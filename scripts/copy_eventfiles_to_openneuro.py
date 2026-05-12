"""Copy events.tsv files from experiment output dirs into the openneuro BIDS dataset.

- blockdesign: source runs 01-08 -> BIDS runs 03-10
- floc: rename task-localizerOneBack -> task-floc, normalize ses-0N -> ses-N
- skip sub-01 (pilot)
"""
import shutil
from glob import glob
from os.path import basename, dirname, join as pjoin

REPO = "/Users/contier/projects/ffadims/ffadims_1"
OUTDIR = pjoin(REPO, "data/fmri_experiment/openneuro")
SKIP_SUBS = {"sub-01"}


def copy_blockdesign():
    indir = pjoin(REPO, "fMRI_experiment_code/ffadims_block/results")
    infiles = sorted(glob(pjoin(indir, "sub-*", "ses-*", "*events.tsv")))
    assert infiles, "no blockdesign events found"
    for infile in infiles:
        parts = infile.split("/")
        sub = parts[-3]
        ses = parts[-2]
        if sub in SKIP_SUBS:
            continue
        fname = parts[-1]
        run_src = int(fname.split("_")[-2].split("-")[-1])
        run_dst = run_src + 2
        new_fname = fname.replace(f"run-{run_src:02d}", f"run-{run_dst:02d}")
        # insert ses-N into the basename
        bn_parts = new_fname.split("_")
        bn_parts.insert(1, ses)
        new_fname = "_".join(bn_parts)
        outfile = pjoin(OUTDIR, sub, ses, "func", new_fname)
        assert not infile == outfile
        shutil.copy(infile, outfile)
        print(f"{infile}\n  -> {outfile}")


def copy_floc():
    indir = pjoin(REPO, "fMRI_experiment_code/pyfLoc/data")
    infiles = sorted(glob(pjoin(indir, "*events.tsv")))
    assert infiles, "no floc events found"
    for infile in infiles:
        elements = basename(infile).split("_")
        sub_i = int(elements[0].split("-")[-1])
        ses_i = int(elements[1].split("-")[-1])
        run_i = int(elements[-2].split("-")[-1])
        sub = f"sub-{sub_i:02d}"
        if sub in SKIP_SUBS:
            continue
        outfile = pjoin(
            OUTDIR, sub, f"ses-{ses_i}", "func",
            f"{sub}_ses-{ses_i}_task-floc_run-{run_i:02d}_events.tsv",
        )
        shutil.copy(infile, outfile)
        print(f"{infile}\n  -> {outfile}")


if __name__ == "__main__":
    copy_blockdesign()
    copy_floc()
