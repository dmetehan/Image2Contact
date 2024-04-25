import itertools
import json
import os
from collections import defaultdict

import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.stats import multivariate_normal
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.sampler import WeightedRandomSampler

import sys
sys.path.append("/mnt/hdd1/GithubRepos/DyadicSignatureDetection")
from utils import Aug, Options, parse_config


# Images should be cropped around interacting people pairs before using this class.
class YOUth10mSignatureTestVideo(Dataset):
    def __init__(self, root_dir, camera='cam1', transform=None, target_transform=None, option=Options.jointmaps,
                 target_size=(224, 224), augment=(), recalc_joint_hmaps=False, bodyparts_dir=None, depthmaps_dir=None):
        # Using folds from Pose2Contact
        self.camera = camera
        self.option = option
        if Aug.crop in augment:
            self.resize = (int(round((1.14285714286 * target_size[0]))),
                           int(round((1.14285714286 * target_size[1]))))  # 224 to 256, 112 to 128 etc.
        else:
            self.resize = tuple(target_size)
        self.target_size = target_size
        self.heatmaps_dir = os.path.join(root_dir, "heatmaps")
        self.gauss_hmaps_dir = os.path.join(root_dir, "gauss_hmaps")
        self.joint_hmaps_dir = os.path.join(root_dir, "joint_hmaps")
        self.opticalflow_dir = os.path.join(root_dir, "optical_flow")
        self.crops_dir = os.path.join(root_dir, "crops")
        if bodyparts_dir:
            self.bodyparts_dir = os.path.join(root_dir, bodyparts_dir)
        if depthmaps_dir:
            self.depthmaps_dir = os.path.join(root_dir, depthmaps_dir)
        os.makedirs(self.gauss_hmaps_dir, exist_ok=True)
        os.makedirs(self.joint_hmaps_dir, exist_ok=True)
        labels_dets_file = os.path.join(root_dir, "pose_detections.json")
        self.img_labels_dets = pd.read_json(labels_dets_file)
        self.img_labels_dets['frame'] = self.img_labels_dets['crop_path'].apply(lambda x: x.split('/')[-1])
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment
        self.hflip = transforms.RandomHorizontalFlip(1)  # the probability of flipping is defined in the function
        self.rand_rotate = transforms.RandomRotation(10)
        self.flip_pairs_pose = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.flip_pairs_bodyparts = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
        self.debug_printed = False
        self.color_aug = transforms.Compose(
            [transforms.RandomPhotometricDistort(), transforms.GaussianBlur(3, [0.01, 1.0])])
        self.recalc_joint_hmaps = recalc_joint_hmaps
        self.reg_mapper = self.comb_regs('dataset/combined_regions_6.txt')

    @staticmethod
    def convert_folds_to_sets(folds_path):
        with open(folds_path) as f:
            folds = json.load(f)
        fold_sets = []
        for f in range(len(folds)):
            fold_sets.append({'test': folds[f],
                              'val': folds[f],
                              'train': list(itertools.chain.from_iterable([folds[i]
                                                                           for i in range(len(folds)) if i != f]))
                              })
        return fold_sets

    def __len__(self):
        return len(self.img_labels_dets)

    def get_rgb(self, idx):
        crop_path = self.img_labels_dets.loc[idx, "crop_path"]
        label = self.get_label(idx)
        crop = Image.open(crop_path)
        # noinspection PyTypeChecker
        rgb = np.transpose(np.array(crop.resize(self.resize), dtype=np.float32) / 255, (2, 0, 1))
        return rgb, label

    @staticmethod
    def bbox_xyxy2cs(x1, y1, x2, y2, aspect_ratio, padding=1.):
        """ Converts xyxy bounding box format to center and scale with the added padding.
        """
        bbox_xyxy = np.array([x1, y1, x2, y2])
        bbox_xywh = bbox_xyxy.copy()
        bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0]
        bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1]
        x, y, w, h = bbox_xywh[:4]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        pixel_std = 200
        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h], dtype=np.float32) / pixel_std
        scale = scale * padding
        return center, scale

    @staticmethod
    def bbox_cs2xyxy(center, scale, padding=1., pixel_std=200.):
        wh = scale / padding * pixel_std
        xy = center - 0.5 * wh
        bbox_xywh = np.r_[xy, wh]
        bbox_xyxy = bbox_xywh.copy()
        bbox_xyxy[2] = bbox_xyxy[2] + bbox_xyxy[0]
        bbox_xyxy[3] = bbox_xyxy[3] + bbox_xyxy[1]
        x1, y1, x2, y2 = bbox_xyxy
        return x1, y1, x2, y2

    @staticmethod
    def scale_bbox(x1, y1, x2, y2, heatmap_size, padding=1.25):
        aspect_ratio = 0.75  # fixed input size to the network: w=288, h=384
        center, scale = YOUth10mSignatureTestVideo.bbox_xyxy2cs(x1, y1, x2, y2, aspect_ratio, padding=padding)

        hmap_aspect_ratio = heatmap_size[0] / heatmap_size[1]
        hmap_center, hmap_scale = YOUth10mSignatureTestVideo.bbox_xyxy2cs(0, 0, heatmap_size[0], heatmap_size[1],
                                                                 hmap_aspect_ratio)

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0
        scale_x = scale[0] / heatmap_size[0]
        scale_y = scale[1] / heatmap_size[1]

        hmap_scale[0] *= scale_x
        hmap_scale[1] *= scale_y
        hmap_center[0] = center[0]
        hmap_center[1] = center[1]
        x1, y1, x2, y2 = YOUth10mSignatureTestVideo.bbox_cs2xyxy(hmap_center, hmap_scale)
        return x1, y1, x2, y2

    def get_joint_hmaps(self, idx, rgb=False):
        label = self.get_label(idx)
        crop_path = self.img_labels_dets.loc[idx, "crop_path"]
        frame = crop_path.split('/')[-1]
        joint_hmap_path = f'{os.path.join(self.joint_hmaps_dir, frame)}.npy'
        if os.path.exists(joint_hmap_path):
            joint_hmaps = np.array(np.load(joint_hmap_path), dtype=np.float32)
            if joint_hmaps.shape[1:] != self.resize:
                heatmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
                if len(self.img_labels_dets.loc[idx, 'bbxes']) > 0:
                    for p in range(len(self.img_labels_dets.loc[idx, 'bbxes'])):
                        for k in range(17):
                            heatmaps[p * 17 + k, :, :] = transforms.Resize((self.resize[0], self.resize[1]))(
                                Image.fromarray(joint_hmaps[p * 17 + k, :, :]))
                    joint_hmaps = (heatmaps - np.min(heatmaps)) / (np.max(heatmaps) - np.min(heatmaps))
                else:
                    print(f"{self.img_labels_dets.loc[idx, 'crop_path']} - joint_hmaps all zeros")
                    joint_hmaps = np.zeros((34, self.resize[0], self.resize[1]), dtype=np.float32)
            else:
                joint_hmaps = (joint_hmaps - np.min(joint_hmaps)) / (np.max(joint_hmaps) - np.min(joint_hmaps))
            if rgb:
                crop = Image.open(crop_path)
                # noinspection PyTypeChecker
                joint_hmaps_rgb = np.concatenate((joint_hmaps,
                                                  np.transpose(
                                                      np.array(crop.resize(self.resize), dtype=np.float32) / 255,
                                                      (2, 0, 1))), axis=0)
                return joint_hmaps_rgb, label
            else:
                return joint_hmaps, label
        if not self.recalc_joint_hmaps:
            print(f"WARNING! {joint_hmap_path} does not exist!")
            return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label

        crop = Image.open(crop_path)
        heatmap_path = f'{os.path.join(self.heatmaps_dir, frame)}.npy'
        print(heatmap_path)
        if not os.path.exists(heatmap_path):
            # no detections
            print("no detections")
            return np.zeros((34 if not rgb else 37, self.resize[0], self.resize[1]), dtype=np.float32), label
        hmap = np.load(heatmap_path)
        if hmap.shape[0] == 1:
            # only 1 detection
            hmap = np.concatenate(
                (hmap, np.zeros((1, hmap.shape[1] + (0 if not rgb else 3),
                                 hmap.shape[2], hmap.shape[3]), dtype=np.float32)))
        try:
            heatmap_size = hmap.shape[3], hmap.shape[2]
        except:
            print(hmap.shape)
        width, height = crop.size
        heatmaps = np.zeros((34 if not rgb else 37, height, width), dtype=np.float32)
        for p in range(len(self.img_labels_dets.loc[idx, "bbxes"])):
            x1, y1, x2, y2 = self.img_labels_dets.loc[idx, "bbxes"][p][:4]
            x1, y1, x2, y2 = self.scale_bbox(x1, y1, x2, y2, heatmap_size)
            x1, y1, x2, y2 = list(map(int, map(round, [x1, y1, x2, y2])))
            w, h = x2 - x1, y2 - y1

            rx1 = max(-x1, 0)
            ry1 = max(-y1, 0)
            rx2 = min(x2, width) - x1
            ry2 = min(y2, height) - y1
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width)
            y2 = min(y2, height)
            # person_hmaps = np.zeros((34 if not rgb else 37, h, w, 3), dtype=np.float32)
            # person_crop = np.zeros((h, w, 3), dtype=np.float32)
            for k in range(17):
                heatmaps[p * 17 + k, y1:y2, x1:x2] = np.array(transforms.Resize((h, w), antialias=True)
                                                              (Image.fromarray(hmap[p, k, :, :])))[ry1:ry2, rx1:rx2]
                # person_hmaps[p*17+k, :, :, p] = np.array(transforms.Resize((h, w), antialias=True)
                #                                                  (Image.fromarray(heatmap[p, k, :, :])))
            # heatmap_img = Image.fromarray((person_hmaps.max(axis=0)*255).astype(np.uint8))
            # person_crop[ry1:ry2, rx1:rx2, :] = np.array(crop)[y1:y2, x1:x2, :]
            # super_imposed_img = Image.blend(Image.fromarray(person_crop.astype(np.uint8)).convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
            # plt.imshow(super_imposed_img)
            # plt.show()
        # noinspection PyArgumentList
        # heatmap_img = Image.fromarray((heatmaps.max(axis=0)*255).astype(np.uint8))
        # super_imposed_img = Image.blend(crop.convert('RGBA'), heatmap_img.convert('RGBA'), alpha=0.5)
        # draw = ImageDraw.Draw(super_imposed_img)
        # for p in range(len(self.pose_dets[idx]['bbxes'])):
        #     draw.rectangle(self.pose_dets[idx]['bbxes'][p][:4], outline="red" if p == 0 else "green")
        # for p in range(len(self.pose_dets[idx]['preds'])):
        #     for k in range(17):
        #         x, y, _ = self.pose_dets[idx]['preds'][p][k]
        #         draw.ellipse((x-5, y-5, x+5, y+5), outline="yellow" if p == 0 else "white")
        # plt.imshow(super_imposed_img)
        # plt.savefig(f'{idx:0d}.png')
        # plt.show()
        np.save(joint_hmap_path, heatmaps)
        return heatmaps, label

    def get_bodyparts(self, idx):
        part_ids = [0, 13, 18, 24, 21, 20, 11, 8, 12, 6, 2, 16, 5, 25, 22]
        bodyparts_path = f"{os.path.join(self.bodyparts_dir, self.img_labels_dets.loc[idx, 'crop_path'].split('/')[-1])}"
        bodyparts_base_path = f'{bodyparts_path.split(".")[0]}'
        if os.path.exists(bodyparts_path):
            # convert colors into boolean maps per body part channel (+background)
            bodyparts_img = np.asarray(
                transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(Image.open(bodyparts_path)),
                dtype=np.uint32)
            x = bodyparts_img // 127
            x = x * np.array([9, 3, 1])
            x = np.add.reduce(x, 2)
            bodyparts = [(x == i) for i in part_ids]
            bodyparts = np.stack(bodyparts, axis=0).astype(np.float32)
            if self.option == 0:  # debug option
                crop = Image.open(
                    f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels_dets.loc[idx, "crop_path"]))}')
                crop = np.array(crop.resize(self.resize))
                plt.imshow(crop)
                plt.imshow(bodyparts_img, alpha=0.5)
                plt.show()
                for i in range(15):
                    plt.imshow(crop)
                    plt.imshow(bodyparts[i, :, :], alpha=0.5)
                    plt.show()
            return bodyparts
        elif os.path.exists(bodyparts_base_path + '_0.png'):
            bodyparts = np.zeros(self.resize + (15,), dtype=np.float32)
            for i in range(5):
                cur_part_path = f"{bodyparts_base_path}_{i}.png"
                # convert colors into boolean maps per body part channel (+background)
                bodyparts_img = np.asarray(
                    transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(Image.open(cur_part_path)),
                    dtype=np.uint32)
                bodyparts[:, :, (3 * i):(3 * i + 3)] = bodyparts_img / 255.0  # normalization
                # if self.option == Options.debug:  # debug option
                # crop = Image.open(f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels_dets.loc[idx, "crop_path"]))}')
                # crop = np.array(crop.resize(self.resize))
                # plt.imshow(crop)
                # plt.imshow(bodyparts_img, alpha=0.5)
                # plt.show()
            return np.transpose(bodyparts, (2, 0, 1))  # reorder dimensions
        else:
            print(f"WARNING: {bodyparts_path} doesn't exist!")
            return np.zeros((15, self.resize[0], self.resize[1]), dtype=np.float32)

    def get_depthmaps(self, idx):
        depthmaps_path = f"{os.path.join(self.depthmaps_dir, self.camera, '/'.join(self.img_labels_dets.loc[idx, 'crop_path'].split('/')[-2:])).replace('.jpg', '.png')}"
        if os.path.exists(depthmaps_path):
            depthmaps_img = np.asarray(
                transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(Image.open(depthmaps_path)),
                dtype=np.uint32)
            depthmaps = depthmaps_img.astype(np.float32) / 256  # normalization for depthmaps
            if self.option == 0:  # debug option
                crop = Image.open(
                    f'{os.path.join(self.crops_dir, os.path.basename(self.img_labels_dets.loc[idx, "crop_path"]))}')
                crop = np.array(crop.resize(self.resize))
                plt.imshow(crop)
                plt.imshow(depthmaps_img, alpha=0.5)
                plt.show()
            return depthmaps.reshape((1, depthmaps.shape[0], depthmaps.shape[1]))
        else:
            print(f"WARNING: {depthmaps_path} doesn't exist!")
            return np.zeros((1, self.resize[0], self.resize[1]), dtype=np.float32)

    def get_optical_flow(self, idx):
        subject, frame = self.img_labels_dets.loc[idx, 'crop_path'].split('/')[-2:]
        optical_flow_path = os.path.join(self.opticalflow_dir, subject, frame.replace('.jpg', '.png'))
        if not os.path.exists(optical_flow_path):
            print(f"WARNING: {optical_flow_path} doesn't exist!")
            return np.zeros((3, self.resize[0], self.resize[1]), dtype=np.float32)
        optical_flow = np.asarray(
            transforms.Resize(self.resize, interpolation=InterpolationMode.NEAREST)(Image.open(optical_flow_path)),
            dtype=np.uint32)
        return np.transpose(np.asarray(optical_flow, dtype=np.float32), (2, 0, 1))  # reorder dimensions

    def get_label(self, idx, swap=False):
        #onehot = {'42': self.onehot_segmentation(self.img_labels_dets.loc[idx, "seg21_adult"], self.img_labels_dets.loc[idx, "seg21_child"], res=21, swap=swap),
        #          '12': self.onehot_segmentation(self.img_labels_dets.loc[idx, "seg6_adult"], self.img_labels_dets.loc[idx, "seg6_child"], res=6, swap=swap),
        #          '21*21': self.onehot_sig(self.img_labels_dets.loc[idx, "signature21x21"], res=21, swap=swap),
        #          '6*6': self.onehot_sig(self.img_labels_dets.loc[idx, "signature6x6"], res=6, swap=swap)}
        #return onehot
        return 0

    @staticmethod
    def onehot_segmentation(adult_seg, child_seg, res=21, swap=False):
        to_be_swapped = {21: {9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 17: 18, 18: 17, 19: 20, 20: 19},
                         6: {2: 3, 3: 2, 4: 5, 5: 4}}
        mat = torch.zeros(res + res, dtype=torch.int8)
        for adult in adult_seg:
            if swap:
                if adult in to_be_swapped[res]:
                    adult = to_be_swapped[res][adult]
            mat[adult] = 1
        for child in child_seg:
            if swap:
                if child in to_be_swapped[res]:
                    child = to_be_swapped[res][child]
            mat[res + child] = 1
        return mat.flatten()

    @staticmethod
    def onehot_sig(signature, res=21, swap=False):
        to_be_swapped = {21: {9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 17: 18, 18: 17, 19: 20, 20: 19},
                         6: {2: 3, 3: 2, 4: 5, 5: 4}}
        mat = torch.zeros(res, res, dtype=torch.int8)
        for adult, child in signature:
            if swap:
                if adult in to_be_swapped[res]:
                    adult = to_be_swapped[res][adult]
                if child in to_be_swapped[res]:
                    child = to_be_swapped[res][child]
            mat[adult, child] = 1
        return mat.flatten()

    @staticmethod
    def comb_regs(path):
        mapping = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                for reg in list(map(int, map(str.strip, line.strip().split(',')))):
                    mapping[reg] = i
        return lambda x: mapping[x]

    def __getitem__(self, idx):
        augment = ()
        if idx >= len(self):
            raise IndexError()
        if self.option == Options.debug:
            # for debugging
            if not self.debug_printed:
                print("DEBUG: ON")
                self.debug_printed = True
            data = np.zeros((52, self.resize[0], self.resize[1]), dtype=np.float32)
            label = self.get_label(idx)
        elif self.option == Options.rgb:
            data, label = self.get_rgb(idx)
        elif self.option == Options.jointmaps:
            data, label = self.get_joint_hmaps(idx)
        elif self.option == Options.bodyparts:
            label = self.get_label(idx)
            data = self.get_bodyparts(idx)
        elif self.option == Options.depth:
            label = self.get_label(idx)
            data = self.get_depthmaps(idx)
        elif self.option == Options.jointmaps_rgb:
            data, label = self.get_joint_hmaps(idx, rgb=True)
        elif self.option == Options.jointmaps_rgb_bodyparts:
            data, label = self.get_joint_hmaps(idx, rgb=True)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.rgb_bodyparts:
            data, label = self.get_rgb(idx)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.jointmaps_bodyparts:
            data, label = self.get_joint_hmaps(idx, rgb=False)
            bodyparts = self.get_bodyparts(idx)
            data = np.vstack((data, bodyparts))
        elif self.option == Options.jointmaps_bodyparts_depth:
            data, label = self.get_joint_hmaps(idx, rgb=False)
            bodyparts = self.get_bodyparts(idx)
            depthmaps = self.get_depthmaps(idx)
            data = np.vstack((data, bodyparts, depthmaps))
        elif self.option == Options.jointmaps_rgb_bodyparts_opticalflow:
            data, label = self.get_joint_hmaps(idx, rgb=True)
            bodyparts = self.get_bodyparts(idx)
            optical_flow = self.get_optical_flow(idx)
            data = np.vstack((data, bodyparts, optical_flow))
        elif self.option == Options.jointmaps_bodyparts_opticalflow:
            data, label = self.get_joint_hmaps(idx, rgb=False)
            bodyparts = self.get_bodyparts(idx)
            optical_flow = self.get_optical_flow(idx)
            data = np.vstack((data, bodyparts, optical_flow))
        else:
            raise NotImplementedError()

        data, label = self.do_augmentations(data, label, idx, augment)
        metadata = self.get_subj_frame(idx)
        return idx, data, label, metadata

    def get_subj_frame(self, idx):
        frame = self.img_labels_dets.loc[idx, "crop_path"].split('/')[-1]
        return "B50284", frame

    def do_augmentations(self, data, label, idx, augment):
        for aug in augment:
            if aug == Aug.hflip:
                if np.random.randint(2) == 0:  # 50% chance to flip
                    # swap channels of left/right pairs of pose channels
                    if self.option in [Options.jointmaps_rgb, Options.jointmaps, Options.jointmaps_rgb_bodyparts,
                                       Options.jointmaps_bodyparts, Options.jointmaps_bodyparts_depth]:
                        for i, j in self.flip_pairs_pose:
                            data[i, :, :], data[j, :, :] = data[j, :, :], data[i, :, :]
                            data[i + 17, :, :], data[j + 17, :, :] = data[j + 17, :, :], data[i + 17, :, :]
                    # swap channels of left/right pairs of body-part channels
                    if self.option in [Options.jointmaps_rgb_bodyparts]:
                        for i, j in self.flip_pairs_bodyparts:
                            data[i + 37, :, :], data[j + 37, :, :] = data[j + 37, :, :], data[i + 37, :, :]
                    elif self.option in [Options.rgb_bodyparts]:
                        for i, j in self.flip_pairs_bodyparts:
                            data[i + 3, :, :], data[j + 3, :, :] = data[j + 3, :, :], data[i + 3, :, :]
                    elif self.option in [Options.jointmaps_bodyparts, Options.jointmaps_bodyparts_depth]:
                        for i, j in self.flip_pairs_bodyparts:
                            data[i + 34, :, :], data[j + 34, :, :] = data[j + 34, :, :], data[i + 34, :, :]
                    elif self.option in [Options.bodyparts]:
                        for i, j in self.flip_pairs_bodyparts:
                            data[i, :, :], data[j, :, :] = data[j, :, :], data[i, :, :]
                    data[:, :, :] = data[:, :, ::-1]  # flip everything horizontally
                    label = self.get_label(idx, swap=True)  # Updates the label with hflip as well!
            elif aug == Aug.crop:
                i = torch.randint(0, self.resize[0] - self.target_size[0] + 1, size=(1,)).item()
                j = torch.randint(0, self.resize[1] - self.target_size[1] + 1, size=(1,)).item()
                data = data[:, i:i + self.target_size[0], j:j + self.target_size[1]]
            elif aug == Aug.color:
                # random color-based augmentations to the rgb channels
                if self.option in [Options.jointmaps_rgb, Options.jointmaps_rgb_bodyparts]:
                    data[34:37, :, :] = np.transpose(
                        self.color_aug(Image.fromarray(np.transpose(255 * data[34:37, :, :],
                                                                    (1, 2, 0)).astype(np.uint8))),
                        (2, 0, 1)).astype(np.float32) / 255
                elif self.option in [Options.rgb, Options.rgb_bodyparts]:
                    data[:3, :, :] = np.transpose(self.color_aug(Image.fromarray(np.transpose(255 * data[:3, :, :],
                                                                                              (1, 2, 0)).astype(
                        np.uint8))),
                                                  (2, 0, 1)).astype(np.float32) / 255
        return data, label


def init_datasets_with_cfg(root_dir, _, cfg, fold=0):
    return init_datasets(root_dir, root_dir, option=cfg.OPTION,
                         target_size=cfg.TARGET_SIZE, num_workers=8,
                         augment=cfg.AUGMENTATIONS, bodyparts_dir=cfg.BODYPARTS_DIR, depthmaps_dir=cfg.DEPTHMAPS_DIR)


def init_datasets_with_cfg_dict(root_dir, _, config_dict):
    return init_datasets(root_dir, root_dir, option=config_dict["OPTION"],
                         target_size=config_dict["TARGET_SIZE"], num_workers=8, augment=config_dict["AUGMENTATIONS"],
                         bodyparts_dir=config_dict["BODYPARTS_DIR"], depthmaps_dir=config_dict["DEPTHMAPS_DIR"])


def init_datasets(root_dir, _, option=Options.jointmaps, target_size=(224, 224), num_workers=2, augment=(),
                  bodyparts_dir=None, depthmaps_dir=None):
    test_dataset = YOUth10mSignatureTestVideo(root_dir, option=option, target_size=target_size, augment=augment,
                                              bodyparts_dir=bodyparts_dir, depthmaps_dir=depthmaps_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                              num_workers=num_workers)
    return test_loader


def test_class():
    option = Options.jointmaps_rgb_bodyparts
    root_dir = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/test_video/B50284/features'
    train_loader, validation_loader, test_loader = init_datasets(root_dir, root_dir, option=option,
                                                                 num_workers=1, bodyparts_dir='bodyparts_split')
    # print(len(train_loader))
    dataiter = iter(train_loader)
    # for heatmap, label in dataiter:
    #     # continue
    #     print(np.count_nonzero(label), len(label))
    #     cv2.imshow("image", np.array(transforms.ToPILImage()(heatmap[0, 0])))
    #     cv2.waitKey()
    for idx, heatmap, label in validation_loader:
        print(np.count_nonzero(label), len(label))
    # for heatmap, label in test_loader:
    #     continue


def test_get_joint_hmaps():
    option = Options.jointmaps
    root_dir = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mSignatures/test_video/B50284/features'
    dataset = YOUth10mSignatureTestVideo(root_dir, option=option, recalc_joint_hmaps=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dataiter = iter(data_loader)
    count = 0
    for idx, data, label, metadata in dataiter:
        count += len(label)
        if count % 100 == 0:
            print(count)


if __name__ == '__main__':
    # test_class()
    test_get_joint_hmaps()
