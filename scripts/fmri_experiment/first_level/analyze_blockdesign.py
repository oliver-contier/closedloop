import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import nibabel as nib
import numpy as np
from os.path import abspath
from pathlib import Path
import sys
from typing import List

# Script is in scripts/fmri_experiment/first_level; project root is 3 levels up
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root / "src"))

from closedloop.fmri.analyze_experiment import DataLoaderExperiment, make_glm_wf, BlockDesignConfig
from nipype.pipeline.engine.workflows import Workflow


def to_abspaths(files):
    return [abspath(f) for f in files]


def parse_subjects(subjects_arg: str) -> List[str]:
    tokens = [s.strip() for s in subjects_arg.replace(",", " ").split() if s.strip()]
    return [f"{int(s):02d}" if s.isdigit() else s for s in tokens]


@dataclass(frozen=True)
class RunSettings:
    subjects: List[str]
    denoising_approach: str
    smoothing_fwhm_mm: int
    cluster_thr: float
    cluster_pthr: float
    n_procs: int
    dry_run: bool
    allow_missing_nuisance: bool
    stage1_mask_roi: str
    task: str = "blockdesign"


def validate_and_resolve_run_files(
    event_files: List[str],
    bold_files: List[str],
    nuisance_files: List[str],
    allow_missing_nuisance: bool,
) -> List[str]:
    if len(event_files) != len(bold_files):
        raise ValueError(
            f"Number of event files ({len(event_files)}) does not match number "
            f"of BOLD files ({len(bold_files)})."
        )

    if len(nuisance_files) != len(event_files):
        msg = (
            f"Number of nuisance files ({len(nuisance_files)}) does not match number "
            f"of event files ({len(event_files)})."
        )
        if not allow_missing_nuisance:
            raise ValueError(
                f"{msg} Rerun with --allow-missing-nuisance to continue without "
                "nuisance regressors."
            )
        print(f"WARNING: {msg} Continuing without nuisance regressors.")
        return []

    return nuisance_files


def create_and_save_config(out_basedir: Path, config: BlockDesignConfig, analysis_params: dict, denoising_approach: str) -> None:
    """
    Create and save analysis configuration file.

    Args:
        out_basedir: Path to output base directory
        config: BlockDesignConfig instance
        denoising_approach: String specifying which denoising approach to use
    """
    analysis_config = {
        "created_date": datetime.now().isoformat(),
        "script_name": "analyze_blockdesign.py",
        "analysis_type": "block_design_glm",
        "denoising_approach": denoising_approach,
        # config attributes
        "condnames": getattr(config, "condnames", None),
        "contrasts": getattr(config, "contrasts", None),
        "contrast_names": getattr(config, "contrast_names", None),
        # analysis_params dict elements
        **analysis_params,
        "noiseregs_count": len(analysis_params["noiseregs"]) if isinstance(analysis_params["noiseregs"], list) else "dynamic",
    }

    # Save configuration file
    config_file = out_basedir / "analysis_config.json"
    with open(config_file, "w") as f:
        json.dump(analysis_config, f, indent=2)
    print(f"Analysis configuration saved to: {config_file}")


def build_analysis_params(config: BlockDesignConfig, settings: RunSettings) -> dict:
    base_params = config.get_analysis_params(settings.denoising_approach)
    return {
        **base_params,
        "fwhm": settings.smoothing_fwhm_mm,
        "cluster_thr": settings.cluster_thr,
        "cluster_pthr": settings.cluster_pthr,
    }


def build_subject_workflow(
    subj: str,
    project_root: Path,
    out_basedir: Path,
    config: BlockDesignConfig,
    analysis_params: dict,
    settings: RunSettings,
) -> Workflow:
    outdir = out_basedir / f"sub-{subj}"
    wdir = project_root / "blockdesign_glm_wdir" / f"sub-{subj}_smooth-{settings.smoothing_fwhm_mm}mm"
    outdir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)

    dl = DataLoaderExperiment(project_root=project_root, denoising_approach=settings.denoising_approach)
    event_files = to_abspaths(dl.get_event_files(subj, task=settings.task))
    bold_files = to_abspaths(dl.get_bold_files_preproc(subj, task=settings.task))
    nuisance_files = to_abspaths(dl.get_nuisance_files_fmriprep(subj, task=settings.task))
    nuisance_files = validate_and_resolve_run_files(
        event_files=event_files,
        bold_files=bold_files,
        nuisance_files=nuisance_files,
        allow_missing_nuisance=settings.allow_missing_nuisance,
    )
    brainmask = abspath(dl.get_brain_mask(subj))
    analysis_mask = resolve_analysis_mask(
        dl=dl,
        subj=subj,
        fallback_mask=brainmask,
        stage1_mask_roi=settings.stage1_mask_roi,
        output_dir=wdir,
    )

    if settings.dry_run:
        print(
            f"[DRY RUN] sub-{subj}: "
            f"runs={len(event_files)}, nuisance={len(nuisance_files)}, "
            f"fwhm={analysis_params['fwhm']}mm, "
            f"cluster(z={analysis_params['cluster_thr']}, p={analysis_params['cluster_pthr']})"
        )

    subj_wf = make_glm_wf(
        bold_files,
        event_files,
        analysis_mask,
        tr=analysis_params["tr"],
        nuisance_files=nuisance_files,
        outdir=str(outdir),
        wdir=str(wdir),
        condition_colname=analysis_params["condition_colname"],
        ignore_conditions=analysis_params["ignore_conditions"],
        contrasts=config.contrasts,
        contrast_names=config.contrast_names,
        fwhm=analysis_params["fwhm"],
        cluster_thr=analysis_params["cluster_thr"],
        cluster_pthr=analysis_params["cluster_pthr"],
        wf_name=f"wf-{subj}",
        noiseregs=analysis_params["noiseregs"],
        denoising_approach=settings.denoising_approach,
    )
    return subj_wf


def resolve_analysis_mask(
    dl: DataLoaderExperiment,
    subj: str,
    fallback_mask: str,
    stage1_mask_roi: str,
    output_dir: Path,
) -> str:
    """
    Resolve inference mask for fixed-effects and cluster steps.

    If `stage1_mask_roi` is provided, combine all matching ROI files (e.g. both
    FFA hemispheres) into a single binary mask and use it for stage-1 masking.
    Otherwise, fall back to the whole-brain mask.
    """
    if not stage1_mask_roi:
        return fallback_mask

    roi_files = dl.get_rois(subj=subj, roi_name=stage1_mask_roi, hemisphere="both")
    if len(roi_files) == 0:
        raise ValueError(f"No ROI files found for sub-{subj}, roi={stage1_mask_roi}.")

    union_data = None
    ref_img = None
    for roi_file in roi_files:
        img = nib.load(roi_file)
        data = img.get_fdata()
        bin_data = data > 0
        if union_data is None:
            union_data = bin_data
            ref_img = img
        else:
            if data.shape != union_data.shape:
                raise ValueError(
                    f"ROI mask shape mismatch for sub-{subj}, roi={stage1_mask_roi}: "
                    f"{data.shape} vs {union_data.shape}"
                )
            union_data = union_data | bin_data

    if union_data is None or ref_img is None:
        raise RuntimeError(f"Failed to build ROI union mask for sub-{subj}, roi={stage1_mask_roi}.")

    out_mask = output_dir / f"sub-{subj}_stage1mask-roi-{stage1_mask_roi}.nii.gz"
    union_img = nib.Nifti1Image(union_data.astype(np.uint8), ref_img.affine, ref_img.header)
    union_img.to_filename(str(out_mask))
    return str(out_mask)


def parse_args() -> RunSettings:
    parser = argparse.ArgumentParser(description="Run block-design first-level GLM workflows.")
    parser.add_argument("--subjects", type=str, default="02 03 04 05 06 07", help="Space/comma-separated subject IDs.")
    parser.add_argument("--denoising-approach", choices=["motion", "compcor", "aroma"], default="compcor")
    parser.add_argument("--smoothing-fwhm-mm", type=int, default=0)
    parser.add_argument("--cluster-thr", type=float, default=2.3, help="Cluster-forming threshold (z).")
    parser.add_argument("--cluster-pthr", type=float, default=0.05, help="Cluster significance threshold (p).")
    parser.add_argument("--n-procs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Validate files and print run plan only.")
    parser.add_argument(
        "--stage1-mask-roi",
        type=str,
        default="",
        help="Optional independent ROI mask for stage-1 inference (e.g., 'FFA').",
    )
    parser.add_argument(
        "--allow-missing-nuisance",
        action="store_true",
        help="Continue by dropping nuisance regressors if nuisance/event file counts differ.",
    )
    args = parser.parse_args()
    return RunSettings(
        subjects=parse_subjects(args.subjects),
        denoising_approach=args.denoising_approach,
        smoothing_fwhm_mm=args.smoothing_fwhm_mm,
        cluster_thr=args.cluster_thr,
        cluster_pthr=args.cluster_pthr,
        n_procs=args.n_procs,
        dry_run=args.dry_run,
        allow_missing_nuisance=args.allow_missing_nuisance,
        stage1_mask_roi=args.stage1_mask_roi.strip(),
    )


def main() -> None:
    settings = parse_args()
    project_root = _project_root
    mask_tag = f"_mask-{settings.stage1_mask_roi}" if settings.stage1_mask_roi else ""
    out_basedir = (
        project_root
        / "results"
        / "fmri_experiment"
        / f"blockdesign_denoised-{settings.denoising_approach}_smooth-{settings.smoothing_fwhm_mm}mm{mask_tag}"
    )

    config = BlockDesignConfig()
    analysis_params = build_analysis_params(config, settings)
    out_basedir.mkdir(parents=True, exist_ok=True)
    create_and_save_config(out_basedir, config, analysis_params, settings.denoising_approach)

    subj_workflows = []
    for subj in settings.subjects:
        subj_workflows.append(
            build_subject_workflow(
                subj=subj,
                project_root=project_root,
                out_basedir=out_basedir,
                config=config,
                analysis_params=analysis_params,
                settings=settings,
            )
        )

    if settings.dry_run:
        print(f"[DRY RUN] Validated {len(subj_workflows)} subject workflows; no execution requested.")
        return

    meta_wf = Workflow(name="analyze_blockdesign_wf")
    meta_wf.add_nodes(subj_workflows)
    meta_wf.base_dir = str(project_root / "blockdesign_glm_wdir" / "meta_wf")
    meta_wf.run(plugin="MultiProc", plugin_args={"n_procs": settings.n_procs})


if __name__ == "__main__":
    main()
