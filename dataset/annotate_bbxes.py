import os
from json import JSONDecodeError

import cv2
import json
import numpy as np
import pandas as pd


def draw_skeleton(frame, pose, bbxes):
    colors = [(255, 0, 0, 0.5), (0, 255, 0, 0.5)]  # 0: blue, 1: green
    size = [6, 5]
    assert len(pose) == len(bbxes) <= 2, f"{len(pose)} {len(bbxes)}"
    if len(pose) == 0:
        return False
    elif len(pose) == 1:
        print("1 detection!")
    for p in range(len(bbxes)):
        cv2.rectangle(frame, list(map(round, bbxes[p][:2])), list(map(round, bbxes[p][2:4])), colors[p])
    for p in range(len(pose)):
        for k in range(17):
            x, y, _ = pose[p][k]
            cv2.circle(frame, (round(x), round(y)), size[p], colors[p], -1)
    return True


def process_keypress(key, pose_dets, idx):
    if key == ord('s'):
        # swap order
        print("Swap order")
        pose_dets['bbxes_identity'][idx] = 's'
        key = cv2.waitKey(0)
    else:
        pose_dets['bbxes_identity'][idx] = ''
    if key == ord('p'):
        print("Parent incorrect")
        pose_dets['bbxes_identity'][idx] += 'p'
        key = cv2.waitKey(0)
    elif key == ord('c'):
        print("Child incorrect")
        pose_dets['bbxes_identity'][idx] += 'c'
        key = cv2.waitKey(0)
    elif key == ord('b'):
        print("Both incorrect")
        pose_dets['bbxes_identity'][idx] += 'b'
        key = cv2.waitKey(0)
    if key == ord('m'):
        print("Which one is mixed?")
        pose_dets['bbxes_identity'][idx] += 'm'
        key = cv2.waitKey(0)
        if key == ord('p'):
            print("Parent mixed")
            pose_dets['bbxes_identity'][idx] += 'p'
        elif key == ord('c'):
            print("Child mixed")
            pose_dets['bbxes_identity'][idx] += 'c'
        elif key == ord('b'):
            print("Both mixed")
            pose_dets['bbxes_identity'][idx] += 'b'
    if key == ord('q'):
        if idx in pose_dets['bbxes_identity']:
            del pose_dets['bbxes_identity'][idx]
        return False
    print(pose_dets['bbxes_identity'][idx])
    return True


def analyze_annotations(pose_dets):
    swap_count, parent_error_count, child_error_count = 0, 0, 0
    for idx, bbxes_identity in pose_dets['bbxes_identity'].items():
        for character in bbxes_identity:
            if character == 's':
                swap_count += 1
            if character == 'p':
                parent_error_count += 1
            if character == 'c':
                child_error_count += 1
            if character == 'b':
                parent_error_count += 1
                child_error_count += 1
            if character == 'm':
                break  # not counting mixed up ones
    print(f"{len(pose_dets['bbxes_identity'])} total frames with at least 1 person detection.")
    print(f"{swap_count} needs swapping.\n{parent_error_count} incorrect parent keypoints.\n{child_error_count} incorrect parent keypoints")


def fix_order(pose_dets):
    new_pose_dets = {'preds': {}, 'bbxes': {}, 'crop_path': {}, 'contact_type': {}}
    swap_count = 0
    for idx, bbxes_identity in pose_dets['bbxes_identity'].items():
        new_pose_dets['crop_path'][idx] = pose_dets['crop_path'][idx]
        new_pose_dets['contact_type'][idx] = pose_dets['contact_type'][idx]
        if len(bbxes_identity) > 0 and bbxes_identity[0] == 's':
            swap_count += 1
            if len(pose_dets['preds'][idx]) == 1:
                new_pose_dets['preds'][idx] = [np.zeros_like(pose_dets['preds'][idx][0], dtype=int).tolist(), pose_dets['preds'][idx][0]]
                new_pose_dets['bbxes'][idx] = [np.zeros_like(pose_dets['bbxes'][idx][0], dtype=int).tolist(), pose_dets['bbxes'][idx][0]]
            elif len(pose_dets['preds'][idx]) == 2:
                new_pose_dets['preds'][idx] = [pose_dets['preds'][idx][1], pose_dets['preds'][idx][0]]
                new_pose_dets['bbxes'][idx] = [pose_dets['bbxes'][idx][1], pose_dets['bbxes'][idx][0]]
            else:
                raise ValueError(f"{pose_dets['preds'][idx]} for idx={idx} has {len(pose_dets['preds'][idx])} elements!")
        else:
            if len(pose_dets['preds'][idx]) == 1:
                new_pose_dets['preds'][idx] = [pose_dets['preds'][idx][0], np.zeros_like(pose_dets['preds'][idx][0], dtype=int).tolist()]
                new_pose_dets['bbxes'][idx] = [pose_dets['bbxes'][idx][0], np.zeros_like(pose_dets['bbxes'][idx][0], dtype=int).tolist()]
            elif len(pose_dets['preds'][idx]) == 2:
                new_pose_dets['preds'][idx] = pose_dets['preds'][idx]
                new_pose_dets['bbxes'][idx] = pose_dets['bbxes'][idx]
            else:
                raise ValueError(f"{pose_dets['preds'][idx]} for idx={idx} has {len(pose_dets['preds'][idx])} elements!")
    print(f"{swap_count} poses swapped!")
    return new_pose_dets


def main():
    root_dir = r"C:\Pose2Contact\data\youth\signature\all"
    dets_annots_file = os.path.join(root_dir, "pose_detections_bbxes_identity.json")
    dets_out_file = os.path.join(root_dir, "pose_detections_identity_fixed.json")
    with open(dets_annots_file, 'r') as f:
        pose_dets = json.load(f)
    pose_dets['bbxes_identity_info'] = ('the default order of bbxes is 0:parent, 1:child!\n'
                                        'For the bbxes_identity markers:\n'
                                        'p: parent incorrect\n'
                                        'c: child incorrect\n'
                                        'b: both incorrect\n'
                                        'mp: parent keypoints mixed to the child\n'
                                        'mc: child keypoints mixed to the parent\n'
                                        'mb: both parent and child keypoints mixed into each other\n'
                                        's: swap parent/child order (0:child 1:parent)\n')
    if 'bbxes_identity' not in pose_dets:
        pose_dets['bbxes_identity'] = {}
    # del pose_dets['bbxes_identity']['2732']
    full_annotated = True
    ambiguous_count = 0
    for idx, label in pose_dets['contact_type'].items():
        if label == 'ambiguous':
            ambiguous_count += 1
            continue  # skip ambiguous frames
        elif idx in pose_dets['bbxes_identity']:
            print(f"Skipping {idx}, already annotated! {pose_dets['bbxes_identity'][idx]}")
            continue  # skip already annotated bits
        print(f"{idx}/{len(pose_dets['bbxes'])}")
        if os.path.exists(pose_dets['crop_path'][idx]):
            path = pose_dets['crop_path'][idx]
        else:
            path = os.path.join(r"D:\Datasets\YOUth\signature\crops", "/".join(pose_dets['crop_path'][idx].split('/')[-3:]))
        print(path)
        img, window_name = cv2.imread(path), "_".join(pose_dets['crop_path'][idx].split('/')[-2:])
        draw_response = draw_skeleton(img, pose_dets['preds'][idx], pose_dets['bbxes'][idx])
        if not draw_response:
            continue  # no skeletons to draw
        full_annotated = False
        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, 600, 300)  # Move it to (40,30)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        response = process_keypress(key, pose_dets, idx)
        cv2.destroyWindow(window_name)
        if not response:
            break
    if full_annotated:
        print(f"{ambiguous_count} ambiguous frames.")
        analyze_annotations(pose_dets)
        pose_dets_fixed = fix_order(pose_dets)
        with open(dets_out_file, 'w') as f:
            json.dump(pose_dets_fixed, f)
    else:
        with open(dets_annots_file, 'w') as f:
            json.dump(pose_dets, f)


if __name__ == '__main__':
    main()
