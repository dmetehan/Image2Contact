import os
from json import JSONDecodeError

import cv2
import json
import numpy as np
import pandas as pd


def draw_skeleton(frame, pose, bbxes):
    colors = [(255, 0, 0), (0, 255, 0)]  # 0: blue, 1: green
    assert len(pose) == len(bbxes) <= 2, f"{len(pose)} {len(bbxes)}"
    if len(pose) == 0:
        return False
    for p in range(len(bbxes)):
        cv2.rectangle(frame, list(map(round, bbxes[p][:2])), list(map(round, bbxes[p][2:4])), colors[p])
    for p in range(len(pose)):
        for k in range(17):
            x, y, _ = pose[p][k]
            cv2.circle(frame, (round(x), round(y)), 5, colors[p], -1)
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


def main():
    dets_file = "/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/all/pose_detections.json"
    pose_dets = json.load(open(dets_file))
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
    for idx, label in pose_dets['contact_type'].items():
        if label == 'ambiguous':
            continue  # skip ambiguous frames
        elif idx in pose_dets['bbxes_identity']:
            print(f"Skipping {idx}, already annotated! {pose_dets['bbxes_identity'][idx]}")
            continue  # skip already annotated bits
        print(f"{idx}/{len(pose_dets['bbxes'])}")
        img, window_name = cv2.imread(pose_dets['crop_path'][idx]), "_".join(pose_dets['crop_path'][idx].split('/')[-2:])
        draw_response = draw_skeleton(img, pose_dets['preds'][idx], pose_dets['bbxes'][idx])
        if not draw_response:
            continue  # no skeletons to draw
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        response = process_keypress(key, pose_dets, idx)
        cv2.destroyWindow(window_name)
        if not response:
            break
    with open(dets_file, 'w') as f:
        json.dump(pose_dets, f)


if __name__ == '__main__':
    main()
