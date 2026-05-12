# Block Design Experiment

- [Block Design Experiment](#block-design-experiment)
  - [Overview](#overview)
  - [Key Features:](#key-features)
  - [Installation](#installation)
  - [Configuration](#configuration)
    - [Adding Custom Images](#adding-custom-images)
    - [YAML Configuration File](#yaml-configuration-file)
  - [Running the Experiment](#running-the-experiment)
  - [Results](#results)
    - [Data Saving](#data-saving)
    - [Data Contents](#data-contents)
  - [Note](#note)

## Overview
This Python script is designed to run a block design experiment for functional magnetic resonance imaging (fMRI). It handles the presentation of visual stimuli in blocks, collects responses, and records timing data. The script is highly configurable via a YAML configuration file, allowing for customization of stimuli presentation, timing, and screen settings.

## Key Features:
- Presentation of visual stimuli in defined blocks.
- Collection of participant responses.
- Record event and response timing.
- Customizable experiment parameters through a YAML configuration file.
- Supports adding custom images for stimuli.

## Installation
This script primarily requires you to install [PsychoPy](https://www.psychopy.org/download.html). Depending on the machine on which you want to run this experiment, you could either install the standalone version, use pip, or conda.
```bash
# Installation with conda
conda create -n psychopy_env --python=3.8
conda activate psychopy_env
conda install psychopy
```
To check if everything runs properly, you can type to following line and check out if the help message is displayed properly.
```bash
python experiment.py --help
```

## Configuration

### Adding Custom Images
- Place your images in the `.data/stimuli` directory.
- Organize images into subfolders within `./data/stimuli`. The naming convention of these subfolders (e.g., block-01, block-02) determines the name of the conditions in the experiment and will be reflected in the saved results.

### YAML Configuration File
- Use the YAML configuration file to set experiment parameters such as stimulus duration, inter-trial intervals, screen settings, etc.
- A default configuration file **config.yml** is provided, but you should definitely **edit it**! The monitor parameters are especially important to make sure all stimuli are displayed properly.

## Running the Experiment
1. Make sure that **PsychoPy** is properly installed and loaded on the system. For example, at the 7T Scanner at the MPI CBS, you'll have to type `PSYCHOPY` in the command line in order to load it as a module.
2. **Configure the YAML** file with your desired experiment settings.
3. **Run the script** from the command line:
```python
python experiment.py -c path/to/config.yml
```

## Results

### Data Saving
- The script saves experiment results in the `./results` directory.
- Data for each participant is saved in a subdirectory named `sub-[subject_id]`.
- Within each participant's directory, data from each run of the experiment is saved in separate files named `sub-[subject_id]_task-blockdesign_run-[run_number]_events.tsv`.

### Data Contents
- The `.tsv` (Tab Separated Values) files contain detailed records of each trial, including:
  - Type of trial (condition name).
  - Onset time of the stimulus.
  - Duration of the stimulus.
  - Image file name used in the trial.
  - Timestamps of participant responses.

## Note
Make sure the `data/stimuli` directory only contains image files relevant to your experiment. Non-image files or irrelevant images in this directory may cause the script to malfunction. You should also make sure that all blocks have an equal number of images.