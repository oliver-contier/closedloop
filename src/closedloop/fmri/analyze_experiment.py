import os
from os.path import join as pjoin
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from niflow.nipype1.workflows.fmri.fsl import create_modelfit_workflow, create_fixed_effects_flow
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.base import Bunch
from nipype.interfaces.fsl.model import SmoothEstimate, Cluster
from nipype.interfaces.fsl.preprocess import SUSAN
from nipype.interfaces.io import DataSink
import glob
from nipype.interfaces.utility import Function
import numpy as np
from nipype.pipeline.engine import Node, Workflow, MapNode
import json
from typing import List, Optional, Tuple


def get_aroma_motion_components(nuisance_tsv: str) -> List[str]:
    """
    Get the AROMA motion component column names for a given run based on the AROMAnoiseICs.csv file.
    
    Args:
        nuisance_tsv (str): Path to the nuisance TSV file.
        
    Returns:
        List[str]: List of AROMA motion component column names that are classified as motion noise.
    """
    # Derive the path to the AROMAnoiseICs.csv file from the TSV path
    tsv_path = Path(nuisance_tsv)
    aroma_noise_file = tsv_path.parent / (tsv_path.stem.replace('_desc-confounds_timeseries', '_AROMAnoiseICs') + '.csv')
    
    if not aroma_noise_file.exists():
        print(f"WARNING: AROMA noise components file not found: {aroma_noise_file}")
        return []
    
    try:
        # Read the noise component numbers
        with open(aroma_noise_file, 'r') as f:
            noise_components_str = f.read().strip()
            if not noise_components_str:
                return []
            noise_component_nums = [int(x.strip()) for x in noise_components_str.split(',')]
        
        # Convert to column names with zero-padding (aroma_motion_XX where XX is zero-padded)
        aroma_motion_columns = [f'aroma_motion_{num:02d}' for num in noise_component_nums]
        
        # Verify these columns actually exist in the TSV file
        nuisance_df = pd.read_csv(nuisance_tsv, sep='\t')
        existing_aroma_columns = [col for col in aroma_motion_columns if col in nuisance_df.columns]
        
        if len(existing_aroma_columns) != len(aroma_motion_columns):
            missing = set(aroma_motion_columns) - set(existing_aroma_columns)
            print(f"WARNING: Some AROMA motion components not found in TSV: {missing}")
        
        return existing_aroma_columns
        
    except Exception as e:
        print(f"ERROR reading AROMA noise components from {aroma_noise_file}: {e}")
        return []


def get_aroma_motion_regressors_for_run(nuisance_tsv: str) -> List[str]:
    """
    Get the complete list of noise regressors for AROMA denoising approach for a single run.
    
    Args:
        nuisance_tsv (str): Path to the nuisance TSV file for this run.
        
    Returns:
        List[str]: List of noise regressor column names including basic motion + AROMA motion components.
    """
    # Basic motion parameters
    basic_motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement']
    
    # Get AROMA motion components for this run
    aroma_components = get_aroma_motion_components(nuisance_tsv)
    
    # Combine them
    all_regressors = basic_motion + aroma_components
    
    return all_regressors


class DataLoaderExperiment:
    """
    DataLoaderExperiment handles paths and loading of fMRI data, including event files,
    preprocessed BOLD files, brain masks, and nuisance files.

    Attributes:
        datadir (Path): Path to the data directory.
        fmriprepdir (str): Path to the derivatives directory.
        bidsdir (str): Path to the BIDS directory.
        blockdesigndir (str): Path to the block design results directory.
    """

    def __init__(self, project_root: Path, denoising_approach: str = 'compcor'):
        self.project_root = project_root
        self.bidsdir = pjoin(self.project_root,'data', 'fmri_experiment', 'bids')
        self.fmriprepdir = pjoin(self.bidsdir, 'derivatives', 'fmriprep_aroma')
        self.roisdir = pjoin(self.project_root, 'results', 'fmri_experiment', 'rois')
        self.blockdesigndir = pjoin(self.project_root, 'results', 'fmri_experiment', f'blockdesign_denoised-{denoising_approach}')
        for d in [self.bidsdir, self.fmriprepdir, self.roisdir]:
            assert os.path.exists(d), f"Directory {d} does not exist"
        # blockdesigndir might not exist yet if we're running analysis

    def get_event_files(self, subj: str, ses: str = '*', task: str = 'floc'):
        """Return sorted list of event files for a given subject, session, and task."""
        return sorted(glob.glob(pjoin(self.bidsdir, f'sub-{subj}/ses-{ses}/func/*task-{task}*_events.tsv')))

    def get_bold_files_preproc(self, subj: str, ses: str = '*', task: str = 'floc', space: str = 'T1w'):
        """Return sorted list of preprocessed BOLD files for a given subject, session, task, and space."""
        return sorted(glob.glob(pjoin(self.fmriprepdir, f'sub-{subj}/ses-{ses}/func/*task-{task}*space-{space}*desc-preproc_bold.nii.gz')))

    def get_anat_file_preproc(self, subj:str):
        """Return anatomical image created by fMRIPrep"""
        return pjoin(self.fmriprepdir, f'sub-{subj}', 'anat', f'sub-{subj}_desc-preproc_T1w.nii.gz')

    def get_brain_mask(self, subj, space: str = 'T1w'):
        """Return the brain mask file for a given subject and space."""
        found_masks = glob.glob(pjoin(
            self.fmriprepdir, f'sub-{subj}/ses-*/func/*space-{space}*desc-brain_mask.nii.gz'))
        assert found_masks, f"No brain mask found for subject {subj} in space {space}"
        brainmask = found_masks[0]
        return brainmask

    def get_nuisance_files_fmriprep(self, subj: str, ses: str = '*', task: str = 'floc'):
        """Return sorted list of nuisance files for a given subject, session, and task."""
        return sorted(glob.glob(pjoin(self.fmriprepdir, f'sub-{subj}/ses-{ses}/func/*task-{task}*desc-confounds_timeseries.tsv')))
    
    def get_rois(self, subj: str, roi_name: str = 'FFA', hemisphere: str = 'both'):
        """Return sorted list of rois for a given subject, roi name, and hemisphere."""
        if hemisphere == 'both':
            niftis = sorted(glob.glob(pjoin(self.roisdir, f'sub-{subj}_*{roi_name}.nii.gz')))
        else:
            niftis = sorted(glob.glob(pjoin(self.roisdir, f'sub-{subj}*_{hemisphere}{roi_name}.nii.gz')))
        assert len(niftis) > 0, f"No files found for subj={subj}, roi_name={roi_name}, hemisphere={hemisphere}, roisdir={self.roisdir}"
        return niftis
    
    def get_blockdesign_results(self, subj: str, statstype: str = 'contrasts', ndims:int = 15, nruns:int = 16):
        niftis = []
        condinfo = []
        for dim_i in tqdm(range(1,ndims+1), desc='dims', total=ndims, leave=True):
            if statstype == 'betas':
                nifti_name_pattern = f'beta_cond-dim{dim_i}.nii.gz'
            elif statstype == 'contrasts':
                nifti_name_pattern = f'contrast-dim{dim_i}_v_control.nii.gz'
            for run_i in tqdm(range(1,nruns+1), desc=f'runs', total=nruns, leave=False):
                nii = pjoin(
                    self.blockdesigndir, f'sub-{subj}', f'runwise_{statstype}', f'run-{run_i:02d}', 
                    nifti_name_pattern
                    )
                niftis.append(nii)
                condinfo.append({'dim': dim_i, 'run': run_i, 'imageset': 'odd' if run_i % 2 == 1 else 'even'})
        condinfo = pd.DataFrame(condinfo)
        return niftis, condinfo

    def load_analysis_data(
        self,
        sub: str,
        statstype: str = 'contrasts',
        roi_name: str = 'FFA'
    ) -> Tuple[List[str], pd.DataFrame, List[str], np.ndarray]:
        """
        Loads all data needed for fMRI analysis including blockdesign results, ROI files, and condition labels.

        Args:
            sub: Subject identifier (e.g., '02')
            statstype: Type of statistic to load ('betas' or 'contrasts')
            roi_name: Name of the ROI to load

        Returns:
            Tuple containing:
            - List of NIfTI file paths
            - DataFrame with condition information (including human-readable labels)
            - List of ROI file paths
            - Array of condition labels
        """
        # Load condition labels
        labels_path = Path(self.project_root, 'plots', 'validation', 'subj-all_relthresh-0.60_k-17', 'manual_labels.txt')
        if not labels_path.exists():
            raise FileNotFoundError(f"Label file not found at {labels_path}")
        
        labels = np.loadtxt(labels_path, dtype=str)
        
        try:
            # Load ROI files
            roi_files = self.get_rois(subj=sub, roi_name=roi_name, hemisphere='both')
            
            # Load blockdesign results
            niftis, condinfo = self.get_blockdesign_results(subj=sub, statstype=statstype)
            
            # Add human-readable labels (assuming 'dim' codes are 1-indexed)
            condinfo['label'] = labels[condinfo['dim'].astype(int) - 1]
            
            return niftis, condinfo, roi_files, labels
            
        except Exception as e:
            print(f"Error loading analysis data for subject {sub}: {e}")
            raise


def _check_contrasts(contrasts, runinfos):
    """
    Make sure that all conditions in the contrasts are in the design matrix. 
    Otherwise, FSL would give falsely named conditions a contrast value of 0, leading to wrong results.
    """
    for runinfo in runinfos:
        design_conds = runinfo.conditions
        for contrast in contrasts:
            contrast_name, contrast_conds = contrast[0], contrast[2]
            assert set(contrast_conds) <= set(design_conds), f"""
                Condition not found in design but specified in contrast.
                conditions in design: {design_conds}
                contrast name: {contrast_name}
                conditions in contrast: {contrast_conds}
                """
    return None


class PyflocConfig:
    """
    PyflocConfig holds configuration details for the fLOC fMRI analysis, including condition names and contrasts.

    Attributes:
        condnames (list): List of condition names.
        contrasts (list): List of contrasts to be applied.
        contrast_names (list): List of contrast names.
    """

    def __init__(self):
        self.condnames = ['faces', 'bodies', 'places', 'characters', 'objects']
        self.contrasts = [
            ['all', 'T', self.condnames, [1] * len(self.condnames)],
            ['faces_v_all', 'T', self.condnames, [4, -1, -1, -1, -1]],
            ['faces_v_objects', 'T', self.condnames, [1, 0, 0, 0, -1]],
            ['bodies_v_all', 'T', self.condnames, [-1, 4, -1, -1, -1]],
            ['bodies_v_objects', 'T', self.condnames, [0, 1, 0, 0, -1]],
            ['places_v_all', 'T', self.condnames, [-1, -1, 4, -1, -1]],
            ['places_v_objects', 'T', self.condnames, [0, 0, 1, 0, -1]],
            ['characters_v_all', 'T', self.condnames, [-1, -1, -1, 4, -1]],
            ['characters_v_objects', 'T', self.condnames, [0, 0, 0, 1, -1]],
        ]
        self.contrast_names = [con[0] for con in self.contrasts]


class BlockDesignConfig:
    def __init__(self):
        ndims = 15
        faces_dim = 12
        animal_faces_dim = 2
        self.condnames = [
            f'dim{dimi}' for dimi in range(1, ndims+1)] + [
            'control1', 'control2', 'catch', 'response',
        ]
        contrasts = []
        for dimi in range(ndims):
            # contrast for each dimension vs. control
            contvec = [0]*ndims
            contvec[dimi] = 2
            contvec += [-1, -1, 0, 0]
            contrasts += [(f'dim{dimi+1}_v_control', 'T', self.condnames, contvec)]
            # contrast for each dimension v all other (and control)
            contvec = [-1] * ndims
            contvec[dimi] = 16
            contvec += [-1, -1, 0, 0]
            contrasts += [(f'dim{dimi+1}_v_all', 'T', self.condnames, contvec)]

        # Dimension-wise contrasts against human faces (dim12), excluding dim12 itself.
        for dimi in range(ndims):
            if (dimi + 1) == faces_dim:
                continue
            contvec = [0] * ndims
            contvec[dimi] = 1
            contvec[faces_dim - 1] = -1
            contvec += [0, 0, 0, 0]
            contrasts += [(f'dim{dimi+1}_v_faces', 'T', self.condnames, contvec)]

        # Explicit reverse contrast faces > animal_faces.
        contvec = [0] * ndims
        contvec[faces_dim - 1] = 1
        contvec[animal_faces_dim - 1] = -1
        contvec += [0, 0, 0, 0]
        contrasts += [('faces_v_animal_faces', 'T', self.condnames, contvec)]
        for contrast in contrasts:
            assert np.sum(contrast[-1]) == 0
        self.contrasts = contrasts
        self.contrast_names = [con[0] for con in self.contrasts]
        
        # General analysis parameters
        self.tr = 2.
        self.fwhm = 0
        self.cluster_thr = 3.7
        self.cluster_pthr = .001
        self.condition_colname = 'trial_type'
        self.ignore_conditions = ['baseline1', 'baseline2']
        
        # Different denoising approaches
        self.denoising_approaches = {
            'motion': {
                # basic motion parameters (motion derivatives lead to convergence error)
                'noiseregs': ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement']
            },
            'compcor': {
                'noiseregs': [
                    'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'framewise_displacement',
                    # physiological CompCor components (first 6 explain ~ 10% cumulative variance in white matter voxels)
                    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'a_comp_cor_04', 'a_comp_cor_05',
                ]
            },
            'aroma': {
                # AROMA denoising - dynamic component list determined per run
                'noiseregs': 'dynamic'  # Special marker indicating dynamic regressor list
            }
        }
    
    def get_analysis_params(self, denoising_approach):
        """
        Get analysis parameters for a specific denoising approach.
        
        Args:
            denoising_approach (str): Either 'motion', 'compcor', or 'aroma'
            
        Returns:
            dict: Complete analysis parameters including the specified denoising approach
        """
        if denoising_approach not in self.denoising_approaches:
            raise ValueError(f"Unknown denoising approach: {denoising_approach}. Available: {list(self.denoising_approaches.keys())}")
        
        return {
            'tr': self.tr,
            'fwhm': self.fwhm,
            'cluster_thr': self.cluster_thr,
            'cluster_pthr': self.cluster_pthr,
            'condition_colname': self.condition_colname,
            'ignore_conditions': self.ignore_conditions,
            'noiseregs': self.denoising_approaches[denoising_approach]['noiseregs'],
            'denoising_approach': denoising_approach  # Add this to track which approach is being used
        }


def make_runinfo_nipype(
        events_tsv: str,
        stc_reftime: float,
        condition_colname: str = 'trial_type',
        ignore_conditions: list = [],
        nuisance_tsv: Optional[str] = None,
        noiseregs: list = [],
        denoising_approach: str = 'motion',
) -> Bunch:
    """
    Create a Bunch object containing the subject information for a single run.

    Args:
        events_tsv (str): Path to the events TSV file.
        stc_reftime (float): Slice timing correction reference time.
        condition_colname (str): Column name for conditions in the events TSV file.
        ignore_conditions (list): List of conditions to skip.
        nuisance_tsv (Optional[str]): Path to the nuisance TSV file.
        noiseregs (list): List of noise regressors.
        denoising_approach (str): The denoising approach being used.

    Returns:
        Bunch: Contains conditions, onsets, durations, and optionally regressors and regressor names.
    """
    events_df = pd.read_csv(events_tsv, sep='\t')
    # take slice timing reference into account
    events_df['onset'] = events_df['onset'] - stc_reftime
    conditions = []
    onsets = []
    durations = []
    for cond_name, cond_df in events_df.groupby(condition_colname):
        if cond_name in ignore_conditions:
            continue
        conditions.append(cond_name)
        onsets.append(cond_df.onset.tolist())
        durations.append(cond_df.duration.tolist())
    
    # add noise regressors if there are any
    if not noiseregs or not nuisance_tsv:
        runinfo = Bunch(conditions=conditions,
                        onsets=onsets, durations=durations)
    else:
        # Handle dynamic AROMA regressor selection
        if denoising_approach == 'aroma':
            actual_noiseregs = get_aroma_motion_regressors_for_run(nuisance_tsv)
        else:
            actual_noiseregs = noiseregs
        
        nuisance_df = pd.read_csv(nuisance_tsv, sep='\t')
        
        # Check which regressors actually exist in the file
        available_regressors = [reg for reg in actual_noiseregs if reg in nuisance_df.columns]
        if len(available_regressors) != len(actual_noiseregs):
            missing = set(actual_noiseregs) - set(available_regressors)
            print(f"WARNING: Some noise regressors not found in {nuisance_tsv}: {missing}")
        
        if available_regressors:
            nuisance_df_selected = nuisance_df[available_regressors].copy()
            if 'framewise_displacement' in available_regressors:
                # Handle NaN values in framewise_displacement column
                fd_column = nuisance_df_selected['framewise_displacement']
                nuisance_df_selected.loc[:, 'framewise_displacement'] = pd.Series(fd_column).fillna(0)
            
            noiseregs_names = nuisance_df_selected.columns.tolist()
            noiseregs_regressors = [nuisance_df_selected[noisereg].tolist()
                                    for noisereg in noiseregs_names]
            runinfo = Bunch(
                conditions=conditions, onsets=onsets, durations=durations,
                regressor_names=noiseregs_names, regressors=noiseregs_regressors
            )
        else:
            print(f"WARNING: No valid noise regressors found for {events_tsv}, proceeding without nuisance regressors")
            runinfo = Bunch(conditions=conditions,
                            onsets=onsets, durations=durations)
    return runinfo


def sort_copes(files):
    """
    Reshape COPE or VARCOPE files for use in create_fixed_effects_flow().

    Args:
        files (list): List of files to be reshaped.

    Returns:
        list: Reshaped files.
    """
    numelements = len(files[0])
    outfiles = []
    for i in range(numelements):
        outfiles.insert(i, [])
        for j, elements in enumerate(files):
            outfiles[i].append(elements[i])
    return outfiles


def make_glm_wf(
        bold_files: list,
        event_files: list,
        mask_file: str,
        tr: float,
        contrasts: list,
        contrast_names: list,
        wdir: str = '',
        outdir: str = '',
        nuisance_files: list = [],
        condition_colname: str = 'trial_type',
        ignore_conditions: list = [],
        noiseregs: list = ['trans_x', 'trans_y', 'trans_z',
                           'rot_x', 'rot_y', 'rot_z', 'framewise_displacement'],
        denoising_approach: str = 'motion',
        stc_reftime: float = 0.,  # If set to 0, will be 1/2 of tr
        hrf: dict = {'dgamma': {'derivs': False}},
        cluster_thr: float = 3.7,
        cluster_pthr: float = .001,
        hpf: int = 100,
        fwhm: float = 4.,
        ar: bool = False,
        smoothing_brightness_threshold: float = 2000,
        wf_name: str = 'wf',
) -> Workflow:
    """
    Create a Nipype workflow for GLM analysis of fMRI data.

    Args:
        bold_files (list): List of preprocessed BOLD files.
        event_files (list): List of event files.
        mask_file (str): Path to the brain mask file.
        tr (float): Repetition time.
        contrasts (list): List of contrasts.
        contrast_names (list): List of contrast names.
        wdir (str): Working directory for the workflow.
        outdir (str): Output directory for the workflow.
        nuisance_files (list): List of nuisance files.
        condition_colname (str): Column name for conditions in the events TSV file.
        ignore_conditions (list): List of conditions to skip.
        noiseregs (list): List of noise regressors.
        denoising_approach (str): The denoising approach being used ('motion', 'compcor', 'aroma').
        stc_reftime (float): Slice timing correction reference time.
        hrf (dict): Hemodynamic response function model.
        cluster_thr (float): Cluster threshold.
        cluster_pthr (float): Cluster p-value threshold.
        hpf (int): High-pass filter cutoff.
        fwhm (int): Full-width half-maximum for smoothing. If set to 0, no smoothing is applied.
        ar (bool): Auto-regression flag.
        smoothing_brightness_threshold (float): Brightness threshold for smoothing.

    Returns:
        Workflow: Configured Nipype workflow for GLM analysis.
    """
    if not stc_reftime:
        stc_reftime = tr / 2
    if not wdir:
        wdir = pjoin(os.getcwd(), 'floc_glm', 'wdir')

    os.makedirs(pjoin(wdir, 'wf'), exist_ok=True)

    if not outdir:
        outdir = pjoin(os.getcwd(), 'floc_glm', 'out')

    wf = Workflow(name=wf_name, base_dir=wdir)

    # Define environment for FSL nodes to ensure correct PATH
    fsl_bin_path = os.environ.get('FSL_BIN', '/SOFTWARE/fsl/5.0.11/bin') # Get FSL bin path
    current_path = os.environ.get('PATH', '')
    node_path = f'{fsl_bin_path}:{current_path}'
    # Ensure FSLOUTPUTTYPE is also set if needed
    node_environ = {'FSLOUTPUTTYPE': os.environ.get('FSLOUTPUTTYPE', 'NIFTI_GZ'), 'PATH': node_path}

    if fwhm:
        smooth = MapNode(SUSAN(environ=node_environ), name='smooth', iterfield=['in_file'])
        smooth.inputs.in_file = bold_files
        smooth.inputs.fwhm = fwhm
        smooth.inputs.brightness_threshold = smoothing_brightness_threshold
    if nuisance_files:
        # The check for matching lengths is now handled in the DataLoaderExperiment class
        runinfos = [
            make_runinfo_nipype(events_tsv, nuisance_tsv=nuisance_tsv,
                                noiseregs=noiseregs, stc_reftime=stc_reftime, condition_colname=condition_colname, 
                                ignore_conditions=ignore_conditions, denoising_approach=denoising_approach)
            for events_tsv, nuisance_tsv in zip(event_files, nuisance_files)
        ]
    else:
        runinfos = [
            make_runinfo_nipype(events_tsv, stc_reftime=stc_reftime,
                                condition_colname=condition_colname, ignore_conditions=ignore_conditions,
                                denoising_approach=denoising_approach)
            for events_tsv in event_files
        ]
    _check_contrasts(contrasts, runinfos)
    modelspec = Node(
        SpecifyModel(
            subject_info=runinfos,
            high_pass_filter_cutoff=hpf,
            input_units='secs',
            time_repetition=tr
        ),
        name='modelspec')
    modelfit = create_modelfit_workflow()
    modelfit.inputs.inputspec.interscan_interval = tr
    modelfit.inputs.inputspec.contrasts = contrasts
    modelfit.inputs.inputspec.bases = hrf
    modelfit.inputs.inputspec.model_serial_correlations = ar
    if not ar:
        filmgls = modelfit.get_node('modelestimate')
        if filmgls is not None:
            filmgls.inputs.autocorr_noestimate = True
        modelgen = modelfit.get_node('modelgen')
    ffx = create_fixed_effects_flow()
    l2model = ffx.get_node('l2model')
    if l2model is not None:
        l2model.inputs.num_copes = len(bold_files)
    flameo = ffx.get_node('flameo')
    if flameo is not None:
        flameo.inputs.mask_file = mask_file
    sortcopes = Node(Function(function=sort_copes, input_names=[
                     'files'], output_names='outfiles'), name='sortcopes')
    sortvarcopes = Node(Function(function=sort_copes, input_names=['files'], output_names='outfiles'),
                        name='sortvarcopes')
    sortzs = Node(Function(function=sort_copes, input_names=[
        'files'], output_names='outfiles'), name='sortzs')
    sortpes = Node(Function(function=sort_copes, input_names=[
        'files'], output_names='outfiles'), name='sortpes')
    smoothest = MapNode(SmoothEstimate(mask_file=mask_file, environ=node_environ),
                        name='smoothest', iterfield=['zstat_file'])
    cluster = MapNode(Cluster(threshold=cluster_thr, pthreshold=cluster_pthr,
                              out_threshold_file=True, out_pval_file=True, out_index_file=True,
                              out_localmax_txt_file=True, out_localmax_vol_file=True, out_max_file=True,
                              out_mean_file=True, out_size_file=True,
                              environ=node_environ), # Set environment here
                      name='cluster', iterfield=['in_file', 'dlh', 'volume'])
    # configure data sink
    sink = Node(DataSink(), name='sink')
    sink.inputs.base_directory = outdir
    # name level2 contrasts
    sink_substitutions = [(f'/{outputtype}/_cluster{i}/', f'/contrast-{contname}/')
                          for outputtype in ['threshold_file', 'pval_file', 'index_file', 'localmax_txt_file',
                                             'localmax_vol_file', 'max_file', 'mean_file', 'size_file']
                          for i, contname in enumerate(contrast_names)]

    for run_i, runinfo in enumerate(runinfos):
        # name run-wise betas
        # # TODO: Bug! Naming of betas is corrently off, dimension 3 gets called cond-dim4 for some reason...
        for cond_i, condname in enumerate(runinfo.conditions):  # type: ignore
            sink_substitutions += [(f'/_modelestimate{run_i}/pe{cond_i+1}.nii.gz', f'/run-{run_i+1:02d}/beta_cond-{condname}.nii.gz')]
        # name run-wise contrats
        for cont_i, contname in enumerate(contrast_names):
            sink_substitutions += [(f'/_modelestimate{run_i}/zstat{cont_i+1}.nii.gz', f'/run-{run_i+1:02d}/contrast-{contname}.nii.gz')]
            
            
    sink.inputs.substitutions = sink_substitutions

    """
    NOTE to self: modelfit.outputspec.parameter_estimates contains our run-wise beta estimates 
    for all regressors in the design matrix. The first few are our regressors of interest, 
    then come nuisance regressors. The regressors of interest are sorted alphabetically 
    (e.g. 1: bodies, 2: characters, 3: faces) no matter in which order they occured in the event files.
    """
    if fwhm:
        wf.connect([
            (smooth, modelspec, [('smoothed_file', 'functional_runs')]),
            (smooth, modelfit, [
             ('smoothed_file', 'inputspec.functional_data')]),
        ])
    else:
        modelspec.inputs.functional_runs = bold_files
        modelfit.inputs.inputspec.functional_data = bold_files

    wf.connect([
        (modelspec, modelfit, [('session_info', 'inputspec.session_info')]),
        (modelfit, sortcopes, [('outputspec.copes', 'files')]),
        (modelfit, sortvarcopes, [('outputspec.varcopes', 'files')]),
        (sortcopes, ffx, [('outfiles', 'inputspec.copes')]),
        (sortvarcopes, ffx, [('outfiles', 'inputspec.varcopes')]),
        (modelfit, ffx, [('outputspec.dof_file', 'inputspec.dof_files')]),
        (ffx, smoothest, [('outputspec.zstats', 'zstat_file')]),
        (smoothest, cluster, [('dlh', 'dlh'), ('volume', 'volume')]),
        (ffx, cluster, [('outputspec.zstats', 'in_file')]),
        (cluster, sink, [
            ('threshold_file', 'threshold_file'), ('pval_file', 'pval_file'),
            ('index_file', 'index_file'), ('localmax_txt_file', 'localmax_txt_file'),
            ('localmax_vol_file', 'localmax_vol_file'), ('max_file', 'max_file'),
            ('mean_file', 'mean_file'), ('size_file', 'size_file')]),
        (ffx, sink, [('outputspec.zstats', 'ffx_contrasts')]),
        (modelfit, sink, [('outputspec.parameter_estimates', 'runwise_betas')]),
        (modelfit, sink, [('outputspec.zfiles', 'runwise_contrasts')]),
        (modelgen, sink, [('design_image', 'design_image')]),
        # (modelfit, sortpes, [('outputspec.parameter_estimates', 'files')]),
        # (modelfit, sortzs, [('outputspec.zfiles', 'files')]),
        # (sortpes, sink, [('outfiles', 'runwise_betas')]),
        # (sortzs, sink, [('outfiles', 'runwise_contrasts')]),
    ])
    return wf
