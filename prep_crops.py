# This code crops the images around the bounding boxes of each interacting people couple.
import csv
import os
import sys
import cv2
import json

import numpy as np
import pandas as pd


# check_names = ['boys_21220', 'boys_21644', 'boys_53117']


def crop(img, bbxes, person_ids):
    try:
        height, width, _ = img.shape
    except AttributeError:
        width, height = img.size
    p = [{'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}, {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}]
    p[0]['x1'], p[0]['y1'], p[0]['x2'], p[0]['y2'] = bbxes[person_ids[0]]
    p[1]['x1'], p[1]['y1'], p[1]['x2'], p[1]['y2'] = bbxes[person_ids[1]]
    # get bounding box including two interacting people's bounding boxes tightly
    x1, y1, x2, y2 = min(p[0]['x1'], p[1]['x1']), min(p[0]['y1'], p[1]['y1']), max(p[0]['x2'], p[1]['x2']), max(
        p[0]['y2'], p[1]['y2'])

    # calculate offsets to add to the tight bounding box including two interacting people
    scale_left, scale_right, scale_top, scale_bottom = 0.11, 0.11, 0.11, 0.11
    left = 1 if p[1]['x1'] < p[0]['x1'] else 0
    right = 1 if p[1]['x2'] > p[0]['x2'] else 0
    top = 1 if p[1]['y1'] < p[0]['y1'] else 0
    bottom = 1 if p[1]['y2'] > p[0]['y2'] else 0
    dx_left = int(round((p[left]['x2'] - p[left]['x1']) * scale_left))
    dx_right = int(round((p[right]['x2'] - p[right]['x1']) * scale_right))
    dy_top = int(round((p[top]['y2'] - p[top]['y1']) * scale_top))
    dy_bottom = int(round((p[bottom]['y2'] - p[bottom]['y1']) * scale_bottom))

    # actual cropping
    x1, y1, x2, y2 = max(0, int(round(x1)) - dx_left), max(0, int(round(y1)) - dy_top), min(width, int(round(
        x2)) + dx_right), min(height, int(round(y2)) + dy_bottom)

    try:
        return img[y1:y2, x1:x2], (x1, y1)
    except TypeError:
        return img.crop((x1, y1, x2, y2)), (x1, y1)


def prep(set_dir, regmap):
    img_dir = os.path.join(set_dir, 'images')
    crop_dir = os.path.join(set_dir, 'crops')
    annotations_file = os.path.join(set_dir, 'interaction_contact_signature.json')
    ann_data = json.load(open(annotations_file))

    labels_file = os.path.join(set_dir, 'crop_contact_signatures.csv')
    labels = []

    for img_name in ann_data:
        print(img_name)
        img_path = os.path.join(img_dir, f'{img_name}.png')
        bbxes = ann_data[img_name]['bbxes']
        img = cv2.imread(img_path)
        count = 0
        for ci in ann_data[img_name]['ci_sign']:
            print(ci)
            reg_ids = ci['ghum']['region_id']
            # Mapping region ids [0,74) to combined region ids [0,21)
            reg_ids = map_reg_ids(reg_ids, regmap)
            crop_img, offset = crop(img, bbxes, ci['person_ids'])
            crop_path = os.path.join(crop_dir, f'{img_name}_{count:02d}.png')
            print(ci['person_ids'])
            vis_bbox(crop_img, img, bbxes[ci['person_ids'][0]], bbxes[ci['person_ids'][1]], offset, reg_ids)
            # Uncomment the following for saving the crops!
            cv2.imwrite(crop_path, crop_img)
            print(crop_path)
            labels.append([crop_path, reg_ids, offset])
            count += 1
            assert count < 100, f"count ({count}) became 3 digits!"
    pd.DataFrame(labels, columns=['crop_path', 'reg_ids', 'offset']).to_csv(labels_file, index=False)


def vis_bbox(crop_img, img, bbxes1, bbxes2, offset, reg_ids):
    copy_img = img.copy()
    copy_crop = crop_img.copy()
    s1, e1 = ((int(round(bbxes1[0] - offset[0])), int(round(bbxes1[1] - offset[1]))),
              (int(round(bbxes1[2] - offset[0])), int(round(bbxes1[3] - offset[1]))))
    s2, e2 = ((int(round(bbxes2[0] - offset[0])), int(round(bbxes2[1] - offset[1]))),
              (int(round(bbxes2[2] - offset[0])), int(round(bbxes2[3] - offset[1]))))
    cv2.rectangle(copy_crop, s1, e1, (255, 0, 0))
    cv2.rectangle(copy_crop, s2, e2, (0, 255, 0))
    cv2.imshow("crop", copy_crop)
    cv2.rectangle(copy_img, (int(round(bbxes1[0])), int(round(bbxes1[1]))),
                  (int(round(bbxes1[2])), int(round(bbxes1[3]))), (255, 0, 0))
    cv2.rectangle(copy_img, (int(round(bbxes2[0])), int(round(bbxes2[1]))),
                  (int(round(bbxes2[2])), int(round(bbxes2[3]))), (0, 255, 0))
    cv2.imshow("img", copy_img)
    vis_labels(get_label(reg_ids))
    cv2.waitKey(0)


def get_label(reg_ids):
    label = reg_ids
    label = list(set([tuple(elem) for elem in label]))
    label = [[elem[0] for elem in label], [elem[1] for elem in label]]
    label[0] = list(set(label[0]))
    label[1] = list(set(label[1]))
    onehot = [0] * 42
    for l in label[0]:
        onehot[l] = 1
    for l in label[1]:
        onehot[21+l] = 1
    return onehot


def vis_labels(label):
    PERSON = ['blue', 'green']
    N_COARSE = 21
    MASK_COARSE_DIR = '/mnt/hdd1/GithubRepos/contact-signature-annotator/masks_coarse'
    sketches = {pers: cv2.imread('/mnt/hdd1/GithubRepos/contact-signature-annotator/rid_base.png') for pers in PERSON}

    def create_mask_index():
        mask_ind = {pers: np.zeros_like(cv2.imread(f'/mnt/hdd1/GithubRepos/contact-signature-annotator/rid_base.png', 0)) for pers in PERSON}
        for pers in PERSON:
            for rid in range(N_COARSE):
                mask = cv2.imread(f'{MASK_COARSE_DIR}/rid_{rid}.png', 0)
                mask_ind[pers][mask < 100] = rid + 1
        return mask_ind

    MASK_INDEX = create_mask_index()

    def light_up(person, lbl):
        img_cpy = sketches[person].copy()
        for rid in range(len(lbl)):
            if lbl[rid] > 0:
                img_cpy[MASK_INDEX[person] == (rid + 1)] = (0, 0, 255)
        return img_cpy
    labels = {'blue': label[:21], 'green': label[21:]}
    for person in PERSON:
        img = light_up(person, labels[person])
        cv2.imshow(person, img)
    # cv2.waitKey(0)


def map_reg_ids(reg_ids, regmap):
    mapped_ids = []
    for pair in reg_ids:
        mapped_ids.append([regmap[pair[0]], regmap[pair[1]]])
    return mapped_ids


def read_region_map(filepath):
    reader = csv.reader(open(filepath))
    regmap = {}
    for l, line in enumerate(reader):
        for region in line:
            regmap[int(region)] = l
    return regmap


def main():
    # root_dir should include train and test folders for Flickr Signature dataset.
    # root_dir = '/mnt/hdd1/Datasets/CI3D/FlickrCI3D-Signatures'
    root_dir = sys.argv[1]
    print(root_dir)
    regmap = read_region_map("combined_regions.txt")
    prep(os.path.join(root_dir, 'test'), regmap)
    prep(os.path.join(root_dir, 'train'), regmap)


if __name__ == '__main__':
    main()
