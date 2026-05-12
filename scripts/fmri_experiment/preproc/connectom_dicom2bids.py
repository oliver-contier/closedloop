import dicom2nifti
import nibabel as nib
from pathlib import Path
import numpy as np
import shutil
from nipype.interfaces.dcm2nii import Dcm2niix
import logging
from os.path import join as pjoin
import glob
import sys
import os


def make_filemapping_connectom(out_basedir, sub, ses):
    """
    Generate a dictionary that maps nifti file patterns to BIDS-compliant file paths for a given subject and session.
    The function supports mapping for different MRI modalities and sessions including anatomical, field mapping, and functional imaging.

    Parameters:
        out_basedir (str): The base directory where the output BIDS-formatted files will be stored.
        sub (str): The subject identifier, as a string.
        ses (str): The session identifier, as a string.

    Returns:
        dict: A dictionary where keys are glob patterns to match input DICOM files and values are the corresponding output BIDS file paths.
    """
    mapping = {}
    for suffix in [".nii.gz", ".json"]:
        # t1w
        mapping[f"*MPRAGE*{suffix}"] = pjoin(
            out_basedir,
            f"sub-{sub}",
            f"ses-{ses}",
            "anat",
            f"sub-{sub}_ses-{ses}_T1w{suffix}",
        )
        # fmap
        mapping[f"*field_mapping*e1{suffix}"] = pjoin(
            out_basedir,
            f"sub-{sub}",
            f"ses-{ses}",
            "fmap",
            f"sub-{sub}_ses-{ses}_magnitude1{suffix}",
        )
        mapping[f"*field_mapping*e2{suffix}"] = pjoin(
            out_basedir,
            f"sub-{sub}",
            f"ses-{ses}",
            "fmap",
            f"sub-{sub}_ses-{ses}_magnitude2{suffix}",
        )
        mapping[f"*field_mapping*ph{suffix}"] = pjoin(
            out_basedir,
            f"sub-{sub}",
            f"ses-{ses}",
            "fmap",
            f"sub-{sub}_ses-{ses}_phasediff{suffix}",
        )
        # floc
        for runi in range(1, 3):
            mapping[f"*floc_{runi}*{suffix}"] = pjoin(
                out_basedir,
                f"sub-{sub}",
                f"ses-{ses}",
                "func",
                f"sub-{sub}_ses-{ses}_task-floc_run-{runi:02d}_bold{suffix}",
            )
        # blockdesign
        for runi in range(1, 9):
            mapping[f"*block_{runi}*{suffix}"] = pjoin(
                out_basedir,
                f"sub-{sub}",
                f"ses-{ses}",
                "func",
                f"sub-{sub}_ses-{ses}_task-blockdesign_run-{runi+2:02d}_bold{suffix}",
            )
    return mapping


def convert_files_connectom(
    sub: str,
    ses: str,
    mri_incoming_basedir: Path,
    out_basedir: Path,
    conv_wdir: Path,
    rename_only: bool = False,
):
    """
    Converts DICOM files for a given subject and session into NIFTI format and renames them according to BIDS standards.
    If rename_only is set to True, skips conversion and only renames files.
    This is helpful if you have to manually intervene after conversion and then run again.

    Parameters:
        sub (str): Subject identifier.
        ses (str): Session identifier.
        mri_incoming_basedir (Path): Directory containing incoming DICOM files.
        out_basedir (Path): Output directory for the BIDS-compliant dataset.
        conv_wdir (Path): Working directory for conversion.
        rename_only (bool, optional): If True, skips the DICOM to NIFTI conversion step and only performs the renaming of files. Defaults to False.

    Raises:
        AssertionError: If more than one directory is found in the expected DICOM directory, or if the directory does not exist.
        AssertionError: If an unexpected number of files are found for a given mapping pattern during file copying.
    """
    os.makedirs(conv_wdir, exist_ok=True)
    # find the dicomdir I manually copied the data to
    # should be something like ../data/mri_incoming/sub-01/ses-1/18373.ae_9838448499_CONNECTOM
    found_dicomdirs = glob.glob(
        pjoin(mri_incoming_basedir, "sub-{}".format(sub), "ses-{}".format(ses), "*")
    )
    assert len(found_dicomdirs) == 1
    dicomdir = found_dicomdirs[0]

    # run dcm2niix, skip if conversion already
    if not rename_only:
        converter = Dcm2niix(compress="y")
        converter.inputs.source_dir = dicomdir
        converter.inputs.output_dir = conv_wdir
        converter.run()

    # copy nifti files to the bids directory
    mapping = make_filemapping_connectom(out_basedir, sub, ses)
    for in_template, out_file in mapping.items():
        in_files = glob.glob(pjoin(conv_wdir, in_template))
        # we acquire two field maps (beginning and end of session), so we save them as run 1 and run 2
        if "field_mapping" in in_template:
            assert len(in_files) == 2, f"Found {len(in_files)} files for {in_template}"
            in_files = sorted(in_files)
            for run_i, in_file in enumerate(in_files):
                out_file_run = out_file.replace(
                    f"sub-{sub}_ses-{ses}", f"sub-{sub}_ses-{ses}_run-{run_i+1:02d}"
                )
                os.makedirs(os.path.dirname(out_file_run), exist_ok=True)
                shutil.copy(in_file, out_file_run)
            continue
        # There are two versions of the anatomical scan: B0-corrected and uncorrected.
        # the B0-corrected version is the one with a higher running scan number
        elif "MPRAGE" in in_template:
            assert len(in_files) == 2, f"Found {len(in_files)} files for {in_template}"
            in_files = sorted(in_files)
            in_file = in_files[-1]
        # remaining functional runs
        else:
            assert len(in_files) == 1, f"Found {len(in_files)} files for {in_template}"
            in_file = in_files[0]
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        shutil.copy(in_file, out_file)
    return None


def _define_dirnames(sub: int, ses: int, conv_wdir_bdir: Path, out_basedir: Path):
    conv_wdir = conv_wdir_bdir / f"sub-{sub:02d}_ses-{ses}"
    bids_modalities = ("anat", "fmap", "func")
    bids_dirs = [
        pjoin(out_basedir, f"sub-{sub:02d}", f"ses-{ses}", modality)
        for modality in bids_modalities
    ]
    return conv_wdir, bids_dirs


if __name__ == "__main__":

    subjects = [6, 7]
    sessions = [1, 2]
    # script is now in scripts/fmri_experiment/preproc, so go three levels up to project root
    project_root = Path(__file__).resolve().parents[3]
    mri_incoming_basedir = project_root / "data" / "mri_incoming"
    out_basedir = project_root / "data" / "fmri_experiment" / "bids"
    conv_wdir_bdir = Path("./conv_wdir")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    for sub in subjects:
        for ses in sessions:
            conv_wdir, bids_dirs = _define_dirnames(
                sub, ses, conv_wdir_bdir, out_basedir
            )
            for d in bids_dirs + [conv_wdir]:
                os.makedirs(d, exist_ok=True)
            try:
                convert_files_connectom(
                    f"{sub:02d}",
                    str(ses),
                    mri_incoming_basedir,
                    out_basedir,
                    conv_wdir,
                    # rename_only=True,
                )
            except Exception as e:
                logging.error(f"Failed for sub-{sub} ses-{ses}: {e}")
                sys.exit()
