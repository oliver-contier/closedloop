from psychopy import visual, core, event, gui, monitors
from psychopy.hardware.keyboard import Keyboard
import os
from random import shuffle, uniform
import glob
import yaml
from datetime import datetime
from os.path import join as pjoin
import pandas as pd
import argparse
from tqdm import tqdm


def block_order_is_valid(conditions, block_inds):
    # Check if 'baseline' is neither at the first nor the last position
    if conditions[block_inds[0]] == 'baseline' or conditions[block_inds[-1]] == 'baseline':
        return False
    # Check if there are no consecutive 'baseline'
    for i in range(len(block_inds) - 1):
        if conditions[block_inds[i]] == 'baseline' and conditions[block_inds[i + 1]] == 'baseline':
            return False
    return True


def assert_equal_lengths(list_of_lists):
    if list_of_lists:
        first_len = len(list_of_lists[0])
        for sublist in list_of_lists:
            assert (
                len(sublist) == first_len
            ), "Different numbers of images found for the conditions"
    else:
        print("Could not find any stimuli")


def wait_for_experimenter_input(win, kb, height, input_key='space'):
    """
    Waits for the experimenter's input to start the next run.

    Parameters:
    - win (psychopy.visual.Window): The PsychoPy window object.
    - kb (psychopy.event.Keyboard): The PsychoPy keyboard object.
    - height (int): The height of the text stimulus.
    - input_key (str, optional): The key to be pressed by the experimenter. Defaults to 'space'.

    Returns:
    None
    """
    message = f'Press "{input_key}" to start the next run'
    wait_text = visual.TextStim(win, text=message, height=height)
    wait_text.draw()
    win.flip()
    kb.waitKeys(keyList=[input_key])
    win.flip()


def wait_for_scanner_trigger(
    run_clock: core.Clock,
    kb: Keyboard,
    win: visual.Window,
    trigger_key: str,
    height: float,
    message: str,
) -> None:
    """
    Waits for a specific trigger key press and then resets the run clock.

    Parameters:
    run_clock (core.Clock): A PsychoPy clock to reset when trigger is received. Will be reset to first scanner trigger.
    kb (KeyBoard): The PsychoPy keyboard object to listen for key presses on. Keyboard clock will be reset to first scanner trigger.
    win (visual.Window): The PsychoPy window to draw text in.
    trigger_key (str): The key to wait for as a trigger.
    height (float): The height of the text to be displayed.
    message (str): The message to display while waiting for the trigger.

    Returns:
    None
    """
    wait_text = visual.TextStim(win, text=message, height=height)
    wait_text.draw()
    win.flip()
    kb.waitKeys(keyList=[trigger_key])
    run_clock.reset()
    kb.clock.reset()
    win.flip()
    return run_clock


def load_config(config_yml: str) -> dict:
    """
    Loads a YAML configuration file.

    Parameters:
    config_yml (str): Path to the YAML configuration file.

    Returns:
    dict: A dictionary containing configuration parameters.
    """
    with open(config_yml, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def show_dlg() -> dict:
    """
    Displays a dialog box for the user to input session information.

    Returns:
    dict: A dictionary containing the experiment information input by the user.
    """
    dlg = gui.Dlg(title="Block Design Experiment")
    dlg.addText("Session info")
    dlg.addField("Subject:", tip="Subject ID")
    dlg.addField("Session:", tip="Session (1 or 2)")
    dlg.addField("Start with Run:", tip="...")
    exp_info = dlg.show()
    exp_info = {
        "subject": exp_info[0],
        "session_id": exp_info[1],
        "start_run_i": int(exp_info[2]) - 1,
    }
    return exp_info


def setup_clocks() -> tuple:
    """
    Sets up the PsychoPy clocks for the experiment.

    Returns:
    tuple[core.Clock, str]: A tuple containing a PsychoPy clock object and a string representing the clock creation time.
    """
    # time clocks were set up
    clock_creation_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S_%f")[:-3]
    # time relative to first trigger of each run
    run_clock = core.Clock()
    return run_clock, clock_creation_time


def setup_timers() -> tuple:
    """
    Sets up multiple PsychoPy countdown timers for different phases of the experiment.

    Returns:
    tuple[core.CountdownTimer, ...]: A tuple of CountdownTimer objects for various timing requirements in the experiment.
    """
    img_timer = core.CountdownTimer()
    trial_timer = core.CountdownTimer()
    blockend_timer = core.CountdownTimer()
    startblank_timer = core.CountdownTimer()
    endblank_timer = core.CountdownTimer()
    fix_switch_timer = core.CountdownTimer()
    return (
        img_timer,
        trial_timer,
        blockend_timer,
        startblank_timer,
        endblank_timer,
        fix_switch_timer,
    )


def setup_monitor_and_window(
    monitor_distance: float,
    monitor_width: float,
    monitor_resolution: tuple,
    window_fullscr: bool,
    units: str = "deg",
) -> tuple:
    """
    Sets up the PsychoPy monitor and window based on given specifications.

    Parameters:
    monitor_distance (float): Distance of the monitor from the viewer.
    monitor_width (float): Width of the monitor.
    monitor_resolution (tuple[int, int]): Resolution of the monitor.
    window_fullscr (bool): Fullscreen mode flag.
    units (str): Units of measurement, defaults to "deg" (degrees).

    Returns:
    tuple[monitors.Monitor, visual.Window]: A tuple containing the monitor and window objects.
    """
    mon = monitors.Monitor(
        "beamer",
        distance=monitor_distance,
        width=monitor_width,
    )
    mon.setSizePix(monitor_resolution)
    win = visual.Window(
        size=list(monitor_resolution),
        fullscr=window_fullscr,
        units=units,
        monitor=mon,
    )
    return mon, win


def infer_block_info(stimdir: str, n_runs:int, start_run_i=0) -> tuple:
    """
    Infers block information from the stimulus directory structure.

    Parameters:
    stimdir (str): Path to the stimulus directory.

    Returns:
    tuple[list[str], list[str], list[int], int]: A tuple containing lists of block directories, conditions, block indices, and the total number of blocks.
    """
    # blockdirs = []
    # for run_i in range(start_run_i, n_runs):
    #     mod2 = run_i % 2  # odd vs. even runs
    #     blockdirs += [glob.glob(pjoin(stimdir, f'set-{sets[mod2]}', "block-*"))]
    blockdirs = glob.glob(pjoin(stimdir, "block-*"))
    conditions = [os.path.basename(blockdir).split("-")[-1] for blockdir in blockdirs]
    n_blocks = len(blockdirs)
    block_inds = list(range(n_blocks))
    return blockdirs, conditions, block_inds, n_blocks


def get_image_paths(blockdirs: list) -> list:
    """
    Retrieves the paths of all images in each block directory.

    Parameters:
    blockdirs (list[str]): A list of paths to block directories.

    Returns:
    list[list[str]]: A nested list containing the paths of images in each block.
    """
    img_paths = [glob.glob(pjoin(blockdir, "*")) for blockdir in blockdirs]
    # assert that all blocks have equal number of images
    assert_equal_lengths(img_paths)
    return img_paths


def preload_image_stimuli(win: visual.Window, img_paths: list, image_degree: float, text_height: float) -> list:
    """
    Preloads all images as psychopy stimuli.

    Parameters:
    win (Visual.Window): The PsychoPy window object where stimuli will be displayed.
    img_paths (list[str]): A list of paths to image files.
    image_degree (float): Size for the image stimuli in degree visual angle.

    Returns:
    list[Visual.ImageStim]: A list of image stimuli.
    """
    wait_text = visual.TextStim(
        win, 
        text="""Gleich beginnt der naechste Teil des Experiments.
        Bitte reagieren Sie immer, wenn das Kreuz in der Mitte des Bildschirms rot wird.
        
        Simuli werden geladen, bitte Warten ... (< 1 min)""", height=text_height
        )
    wait_text.draw()
    win.flip()
    img_stims = [
        [visual.ImageStim(win, image=img_path, size=image_degree) for img_path in sublist] 
        for sublist in tqdm(img_paths)]
    return img_stims


def create_fixation_stimuli(win: visual.Window, white_fix_f: str, red_fix_f: str, fixation_degree: float) -> tuple:
    """
    Initializes the white and red fixation cross stimuli.

    Parameters:
    win (Visual.Window): The PsychoPy window object where stimuli will be displayed.
    white_fix_f (str): File path to the white fixation image.
    red_fix_f (str): File path to the red fixation image.
    fixation_degree (float): Size of the fixation stimuli.

    Returns:
    tuple[Visual.ImageStim, Visual.ImageStim, Visual.ImageStim]: A tuple containing the white fixation,
    red fixation, and generic image stimuli, respectively.
    """
    white_fix_stim = visual.ImageStim(
        win,
        image=white_fix_f,
        size=fixation_degree,
    )
    red_fix_stim = visual.ImageStim(
        win,
        image=red_fix_f,
        size=fixation_degree,
    )
    return white_fix_stim, red_fix_stim


def specify_paths():
    datadir = "./data"
    resultsdir = "./results"
    stimdir = pjoin(datadir, "stimuli")
    white_fix_f = pjoin(stimdir, "fixation_white.png")
    red_fix_f = pjoin(stimdir, "fixation_red.png")
    return datadir, resultsdir, stimdir, white_fix_f, red_fix_f


def main(config: dict) -> None:
    """
    Runs the main experiment loop for a block design experiment in fMRI.

    This function sets up the experiment environment, including display window,
    stimuli, and input devices. It handles the logic for running multiple runs
    of the experiment, each consisting of several blocks. Within each block,
    it presents a sequence of images and fixation tasks, records responses, and
    manages the timing of these elements according to the configuration provided.

    The function organizes and saves the experiment's data, such as the timing
    of stimuli presentations and participant responses, into a structured format.

    Parameters:
    config (dict): A dictionary containing configuration parameters for the experiment.
                   This includes settings for the display, stimuli, timing, and
                   experiment structure, loaded from a YAML configuration file.

    Returns:
    None
    """
    # specify paths
    datadir, resultsdir, stimdir, white_fix_f, red_fix_f = specify_paths()
    
    # Show initial dialog box
    exp_info = show_dlg()
    exp_info.update(config)
    subject = exp_info["subject"]
    config["soa"] = config["stim_dur"] + config["iti"]
    
    print("\nFinding stimuli...")
    blockdirs_odd, conditions_odd, block_inds, n_blocks = infer_block_info(
        pjoin(stimdir, "set-odd"), exp_info["n_runs"], exp_info["start_run_i"]
        )
    blockdirs_even, conditions_even, block_inds, n_blocks = infer_block_info(
        pjoin(stimdir, "set-even"), exp_info["n_runs"], exp_info["start_run_i"]
        )
    img_paths_odd = get_image_paths(blockdirs_odd)
    img_paths_even = get_image_paths(blockdirs_even)

    # path to subject directory
    subject_dir = pjoin(resultsdir, f"sub-{subject}")

    # Setup window, monitor, and keyboard
    mon, win = setup_monitor_and_window(
        config["monitor_distance"],
        config["monitor_width"],
        config["monitor_resolution"],
        config["window_fullscr"],
    )
    kb = Keyboard()

    print(f"\nPreloading stimuli (odd/even separately)..")
    img_stims_odd = preload_image_stimuli(win, img_paths_odd, config["image_degree"], config["text_height"])
    img_stims_even = preload_image_stimuli(win, img_paths_even, config["image_degree"], config["text_height"])
    
    n_imgs_per_block = len(img_paths_odd[0])
    img_inds = list(range(n_imgs_per_block))
    ses_out_dir = pjoin(subject_dir, f'ses-{exp_info["session_id"]}')
    os.makedirs(ses_out_dir, exist_ok=True)
    print(f"created results directory: {ses_out_dir}")
    # initialize clocks and timers
    print("Initializing clocks...")
    run_clock, clock_creation_time = setup_clocks()
    exp_info["clock_creation_time"] = clock_creation_time
    print("initializing timers")
    (
        img_timer,
        trial_timer,
        blockend_timer,
        startblank_timer,
        endblank_timer,
        fix_switch_timer,
    ) = setup_timers()

    # initialize stsimuli
    white_fix_stim, red_fix_stim = create_fixation_stimuli(
        win, white_fix_f, red_fix_f, config["fixation_degree"]
    )

    def fixation_task():
        # defined here in order to have access to objects defined in main()
        global fix_color_switched
        if fix_switch_timer.getTime() <= 0:
            if fix_color_switched:
                fix_color_switched = False
                fix_switch_timer.reset(
                    uniform(
                        config["fix_switchtime_min"],
                        config["fix_switchtime_max"],
                    )
                )
            else:
                fix_color_switched = True
                fix_switch_timer.reset(config["fix_switch_dur"])
                color_switched = run_clock.getTime()
                events.append(
                    {
                        "onset": color_switched,
                        "trial_type": "catch",
                        "duration": config["fix_switch_dur"],
                    }
                )
        if fix_color_switched:
            red_fix_stim.draw()
        else:
            white_fix_stim.draw()

    for run_i in range(exp_info["start_run_i"], config["n_runs"]):
        # store events in list and define tsv file name
        events = []
        run_name = run_i + 1
        events_tsv = pjoin(
            ses_out_dir, f"sub-{subject}_task-blockdesign_run-{run_name:02d}_events.tsv"
        )
        # choose images based on odd or even
        if run_name % 2:
            img_paths = img_paths_odd
            img_stims = img_stims_odd
            conditions = conditions_odd
            imageset = 'odd'
        else:
            img_paths = img_paths_even
            img_stims = img_stims_even
            conditions = conditions_even
            imageset = 'even'
        
        # Randomize block order within runs
        shuffle(block_inds)
        while not block_order_is_valid(conditions, block_inds):
            shuffle(block_inds)

        print('Waiting for experimenter to start next run...')
        wait_for_experimenter_input(win, kb, config["text_height"])

        # wait until first trigger comes
        print(f"Waiting for scanner trigger key: {config['trigger_key']}...")
        run_clock = wait_for_scanner_trigger(
            run_clock,
            kb,
            win,
            config["trigger_key"],
            config["text_height"],
            f"Run {run_name}. Waiting for scanner trigger.",
        )
        print("Trigger received!")
        
        # reset catch trial timer after first trigger (else, there might be a catch immediately)
        fix_switch_timer.reset(uniform(config["fix_switchtime_min"], config["fix_switchtime_max"]))

        # blank screen at the start of each run.
        startblank_timer.reset(config["start_blank"])
        while startblank_timer.getTime() > 0:
            fixation_task()
            win.flip()

        # start block loop
        for iter_i, block_i in enumerate(block_inds):
            # start within-block image loop
            shuffle(img_inds)
            for img_i in img_inds:
                img_stim = img_stims[block_i][img_i]
                img_path = img_paths[block_i][img_i]
                # img_stim = visual.ImageStim(win, image=img_path, size=config["image_degree"])
                trial_timer.reset(config["soa"])
                img_timer.reset(config["stim_dur"])
                # reset timers for trial duration and and image presentation
                # show trial:
                while trial_timer.getTime() > 0:
                    # 1) draw image
                    if img_timer.getTime() > 0:
                        img_onset = run_clock.getTime()
                        img_stim.draw()
                    # 2) draw fixation
                    fixation_task()
                    # show
                    win.flip()

                # exit if somebody hits escape
                if kb.getKeys(keyList=["escape"]):
                    quit()
                    
                # keep track of responses
                if kb.getKeys(keyList=[config["response_key"]]):
                    # add key press event
                    events.append(
                        {
                            "trial_type": "response",
                            "onset": run_clock.getTime(),
                            "duration": 0,
                            "block_number": iter_i+1,
                        }
                    )
                    
                # add trial to events
                events.append(
                    dict(
                        trial_type=conditions[block_i],
                        onset=img_onset,
                        duration=config["stim_dur"],
                        image=os.path.basename(img_path),
                        imageset=imageset,
                    )
                )

            # show blank between blocks, except at the end of the run
            if iter_i < n_blocks - 1:
                blockend_timer.reset(config["inter_block_blank"])
                while blockend_timer.getTime() > 0:
                    fixation_task()
                    win.flip()

        # show blank at the end of this run
        endblank_timer.reset(config["end_blank"])
        while endblank_timer.getTime() > 0:
            fixation_task()
            win.flip()

        # write events to tsv file
        print(f"\nWriting events to file {events_tsv}\n")
        events_df = pd.DataFrame(events)
        events_df['run'] = run_name
        events_df['session'] = exp_info['session_id']
        events_df.to_csv(events_tsv, sep="\t", index=False)

    # close experiment
    win.close()
    core.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Block Design Experiment",
        description="""
            This script runs a block design experiment for fMRI.
            It manages the presentation of visual stimuli in blocks, collects responses, and records timing data. 
            The experiment's configuration, including stimuli presentation, timing, and screen settings, 
            is adjustable via a YAML configuration file.
      """,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="Path to configuration file",
        default="default_config.yml",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    fix_color_switched = False
    main(config)
