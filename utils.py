import math
import os
import cv2
import glob

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Aug.swap: Swapping the order of pose detections between people (50% chance)
# Aug.hflip: Horizontally flipping the rgb image as well as flipping left/right joints
# Aug.crop: Cropping
class Aug:
    hflip = 'hflip'
    crop = 'crop'
    rotate = 'rotate'
    swap = 'swap'
    color = 'color'
    all = {'hflip': hflip, 'crop': crop, 'rotate': rotate, 'swap': swap, 'color': color}


class Options:
    debug = "debug"
    rgb = "rgb"
    bodyparts = "bodyparts"
    depth = "depth"
    jointmaps = "jointmaps"  # detected heatmaps mapped onto cropped image around interacting people
    jointmaps_rgb = "jointmaps_rgb"
    rgb_bodyparts = "rgb_bodyparts"
    rgb_depth = "rgb_depth"
    jointmaps_bodyparts = "jointmaps_bodyparts"
    jointmaps_depth = "jointmaps_depth"
    bodyparts_depth = "bodyparts_depth"
    jointmaps_rgb_bodyparts = "jointmaps_rgb_bodyparts"
    jointmaps_rgb_depth = "jointmaps_rgb_depth"
    rgb_bodyparts_depth = "rgb_bodyparts_depth"
    jointmaps_bodyparts_depth = "jointmaps_bodyparts_depth"

    all = {debug: debug, rgb: rgb, bodyparts: bodyparts, depth: depth, jointmaps: jointmaps,
           jointmaps_rgb: jointmaps_rgb, rgb_bodyparts: rgb_bodyparts, jointmaps_bodyparts: jointmaps_bodyparts,
           jointmaps_rgb_bodyparts: jointmaps_rgb_bodyparts, rgb_depth: rgb_depth, jointmaps_depth: jointmaps_depth,
           bodyparts_depth: bodyparts_depth, jointmaps_rgb_depth: jointmaps_rgb_depth,
           rgb_bodyparts_depth: rgb_bodyparts_depth, jointmaps_bodyparts_depth: jointmaps_bodyparts_depth}


def check_config(cfg):
    print(cfg.AUGMENTATIONS)
    assert cfg.OPTION in Options.all
    for aug in cfg.AUGMENTATIONS:
        assert aug in Aug.all


def parse_config(config_file):
    class Config:
        pass
    with open(config_file, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    cfg_obj = Config()
    for section in cfg:
        print(section, cfg[section])
        setattr(cfg_obj, section, cfg[section])
    setattr(cfg_obj, "ID", config_file.split('/')[-1].split('_')[0])
    check_config(cfg_obj)
    return cfg_obj


def get_experiment_name(cfg, signatures=False):
    experiment_name = f'{"" if not signatures else "sig_"}' \
                      f'{cfg.ID}' \
                      f'_{cfg.OPTION}' \
                      f'_{cfg.BODYPARTS_DIR.split("_")[-1]}' \
                      f'{"_pretr" if cfg.PRETRAINED else ""}{"Copy" if cfg.PRETRAINED and cfg.COPY_RGB_WEIGHTS else ""}' \
                      f'_{cfg.TARGET_SIZE[0]}' \
                      f'{"_Aug-" if len(cfg.AUGMENTATIONS) > 0 else ""}{"-".join(cfg.AUGMENTATIONS)}' \
                      f'{"_multitask" if cfg.MULTITASK else ""}' \
                      f'_lr{cfg.LR}' \
                      f'_b{cfg.BATCH_SIZE}'
    print("Experiment name:")
    print(experiment_name)
    return experiment_name


def find_last_values_tensorboard(log_dir, tag):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    assert len(event_files) > 0, "No event files found in log directory."
    event_files.sort(key=os.path.getmtime)

    event_file = event_files[-1]  # Get the latest event file.
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_events = event_acc.Scalars(tag)
    assert len(scalar_events) > 0, f"No events found for tag '{tag}' in {log_dir}."
    return scalar_events[-1].value


def get_adj_mat():
    A = np.zeros((21, 21))
    A[0, 1] = A[1, 0] = 1
    A[1, 2] = A[2, 1] = 1
    A[0, 2] = A[2, 0] = 1
    A[2, 3] = A[3, 2] = 1
    A[2, 4] = A[4, 2] = 1
    A[3, 5] = A[5, 3] = 1
    A[3, 15] = A[15, 3] = 1
    A[3, 16] = A[16, 3] = 1
    A[4, 6] = A[6, 4] = 1
    A[4, 15] = A[15, 4] = 1
    A[4, 16] = A[16, 4] = 1
    A[5, 6] = A[6, 5] = 1
    A[5, 7] = A[7, 5] = 1
    A[6, 8] = A[8, 6] = 1
    A[7, 8] = A[8, 7] = 1
    A[7, 9] = A[9, 7] = 1
    A[7, 10] = A[10, 7] = 1
    A[8, 9] = A[9, 8] = 1
    A[8, 10] = A[10, 8] = 1
    A[9, 11] = A[11, 9] = 1
    A[10, 12] = A[12, 10] = 1
    A[11, 13] = A[13, 11] = 1
    A[12, 14] = A[14, 12] = 1
    A[15, 17] = A[17, 15] = 1
    A[16, 18] = A[18, 16] = 1
    A[17, 19] = A[19, 17] = 1
    A[18, 20] = A[20, 18] = 1
    assert np.array_equal(A, np.transpose(A)), "Adjacency matrix is not symmetrical!"
    # row - normalized
    A = A / np.sum(A, axis=0)
    assert np.array_equal(np.sum(A, axis=0), np.ones((A.shape[0])))
    return A
