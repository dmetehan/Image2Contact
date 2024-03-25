import math
import os
import cv2
import glob

import torch
import matplotlib.pyplot as plt
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
    jointmaps_bodyparts_opticalflow = "jointmaps_bodyparts_opticalflow"
    jointmaps_rgb_bodyparts_opticalflow = "jointmaps_rgb_bodyparts_opticalflow"

    all = {debug: debug, rgb: rgb, bodyparts: bodyparts, depth: depth, jointmaps: jointmaps,
           jointmaps_rgb: jointmaps_rgb, rgb_bodyparts: rgb_bodyparts, jointmaps_bodyparts: jointmaps_bodyparts,
           jointmaps_rgb_bodyparts: jointmaps_rgb_bodyparts, rgb_depth: rgb_depth, jointmaps_depth: jointmaps_depth,
           bodyparts_depth: bodyparts_depth, jointmaps_rgb_depth: jointmaps_rgb_depth,
           rgb_bodyparts_depth: rgb_bodyparts_depth, jointmaps_bodyparts_depth: jointmaps_bodyparts_depth,
           jointmaps_rgb_bodyparts_opticalflow: jointmaps_rgb_bodyparts_opticalflow,
           jointmaps_bodyparts_opticalflow: jointmaps_bodyparts_opticalflow}


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
                      f'_b{cfg.BATCH_SIZE}' \
                      f'_{cfg.LOSS_WEIGHTS}'
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


def _init_heatmaps():
    people = ['adult', 'child']
    sketches = {pers: cv2.imread('data/rid_base.png') for pers in people}

    def create_mask_index():
        mask_ind = {pers: np.zeros_like(cv2.imread(f'data/rid_base.png', 0)) for pers in people}
        for pers in people:
            for rid in range(21):
                mask = cv2.imread(f'data/masks_coarse_clear/rid_{rid}.png', 0)
                mask_ind[pers][mask < 100] = rid + 1
        return mask_ind

    return people, sketches, create_mask_index()


def vis_pred_errors_heatmap(gts, preds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    gts = torch.Tensor(gts) == 1
    preds = torch.Tensor(preds) == 1
    true_positives = torch.bitwise_and(gts, preds).sum(dim=0)
    xor = torch.bitwise_xor(gts, preds)
    false_positives = torch.bitwise_and(preds, xor).sum(dim=0)
    false_negatives = torch.bitwise_and(gts, xor).sum(dim=0)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)
    # minimum, maximum = ({'precision': min(precision), 'recall': min(recall), 'f1': min(f1)},
    #                     {'precision': max(precision), 'recall': max(recall), 'f1': max(f1)})
    people, sketches, mask_ind = _init_heatmaps()
    precision = {'adult': precision[:21], 'child': precision[21:]}
    recall = {'adult': recall[:21], 'child': recall[21:]}
    f1 = {'adult': f1[:21], 'child': f1[21:]}
    for person in people:
        for _set, name in [(precision, 'precision'), (recall, 'recall'), (f1, 'f1')]:
            img_cpy = sketches[person].copy()
            colormap = plt.get_cmap('RdYlGn')
            for region, value in enumerate(_set[person]):
                print(f'{person} - {name}: {value}')
                # x = (value - minimum[name]) / (maximum[name] - minimum[name])
                r, g, b = colormap(value)[:3]
                img_cpy[mask_ind[person] == (region + 1)] = (255 * b, 255 * g, 255 * r)  # for cv2 the order is bgr
            cv2.imwrite(os.path.join(save_dir, f'{person}_{name}.png'), img_cpy)
    #     cv2.imshow(person, img_cpy)
    # cv2.waitKey(0)


def vis_threshold_eval(gts, scores, eval_func, epoch, save_dir, **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    xs = torch.linspace(0, 1, 25)
    for key in gts:
        eval_results = [eval_func(gts[key], torch.sigmoid(torch.Tensor(scores[key])) > thresh, **kwargs) for thresh in xs]
        plt.plot(xs, eval_results, label=key)
    plt.xticks(torch.linspace(0, 1, 25))
    plt.xticks(rotation=90)
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(save_dir, f'{epoch}.png'))
    plt.clf()

