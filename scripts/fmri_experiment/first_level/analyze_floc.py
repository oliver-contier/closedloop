from os.path import abspath
from pathlib import Path
import sys

# Script is in scripts/fmri_experiment/first_level; project root is 3 levels up
_project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_project_root / "src"))

from closedloop.fmri.analyze_experiment import DataLoaderExperiment, make_glm_wf, PyflocConfig
from nipype.pipeline.engine.workflows import Workflow


def to_abspaths(files):
    return [abspath(f) for f in files]


if __name__ == "__main__":
    # subjects = [f"{i:02d}" for i in range(2,8)]
    subjects = ['04', '05']
    project_root = _project_root
    subj_workflows = []
    for subj in subjects:
        # make a workflow for each subject
        outdir = project_root / "results" / "fmri_experiment" / "fLoc" / f"sub-{subj}"
        wdir = project_root / "floc_glm_wdir" / f"sub-{subj}"
        outdir.mkdir(parents=True, exist_ok=True)
        wdir.mkdir(parents=True, exist_ok=True)
        dl = DataLoaderExperiment(project_root)
        event_files = to_abspaths(dl.get_event_files(subj, task="floc"))
        bold_files = to_abspaths(dl.get_bold_files_preproc(subj, task="floc"))
        nuisance_files = to_abspaths(dl.get_nuisance_files_fmriprep(subj, task="floc"))
        brainmask = abspath(dl.get_brain_mask(subj))
        config = PyflocConfig()
        subj_wf = make_glm_wf(
            bold_files,
            event_files,
            brainmask,
            tr=2,
            nuisance_files=nuisance_files,
            outdir=str(outdir),
            wdir=str(wdir),
            condition_colname='category',
            ignore_conditions=['baseline'],
            contrasts=config.contrasts,
            contrast_names=config.contrast_names,
            wf_name=f'wf-{subj}',
        )
        subj_workflows.append(subj_wf)
    # create overarching workflow and run it
    meat_wf = Workflow(name="analyze_floc_wf")
    meat_wf.add_nodes(subj_workflows)
    meat_wf.base_dir = str(project_root / "floc_glm_wdir" / "meta_wf")
    meat_wf.run(
        plugin='MultiProc', plugin_args={'n_procs' : 9}
        )
