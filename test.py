import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, jaccard_score, accuracy_score, f1_score
import seaborn as sn
import pandas as pd
from tqdm import tqdm

from ContactSignatureModel import ContactSignatureModel, initialize_model
from utils import parse_config, get_experiment_name

# this is important for FLickrCI3DClassification. It doesn't allow importint v2 after initializing the network.
import torchvision.transforms.v2 as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
_ = ContactSignatureModel()
# net should be initialized here before importing torchvision

from dataset.FlickrCI3D_Signature import init_datasets_with_cfg
# from dataset.YOUth10mClassification import init_datasets

train_dir = '~/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/train'
# train_dir = ''
test_dir = '~/GithubRepos/ContactClassification-ssd/FlickrCI3DClassification/test'
# test_dir = '~/GithubRepos/ContactClassification-ssd/YOUth10mClassification/test'
classes = ("no touch", "touch")


def test_model(model, model_name, experiment_name, exp_dir, data_loader, test_set, device, segmentation):
    model.eval()
    model = model.to(device)
    print(f'Testing on {"test set" if test_set else "validation set"} using {model_name}')
    all_preds, all_labels = (np.zeros((len(data_loader), (21+21) if segmentation else (21*21))),
                             np.zeros((len(data_loader), (21+21) if segmentation else (21*21))))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        i = 0
        for data in tqdm(data_loader):
            _, images, labels = data
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            preds = outputs.T.detach().cpu()
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            all_preds[i, :] = preds[:, 0]

            labels = labels.data.cpu().numpy()
            all_labels[i, :] = labels.T[:, 0]
            i += 1

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="micro")
    jaccard = jaccard_score(all_labels, all_preds, average="micro")
    print(f'Jaccard Score of the network on the test images ({len(all_labels)}): {jaccard:.4f}')
    print(f'Accuracy of the network on the test images ({len(all_labels)}): {acc:.4f}')
    print(f'F1 Score of the network on the test images ({len(all_labels)}): {f1:.4f}')

    # dir_name = [file_name for file_name in sorted(os.listdir(exp_dir)) if experiment_name in file_name and os.path.isdir(f'{exp_dir}/{file_name}')][-1]
    # cf_matrix_norm = confusion_matrix(all_labels, all_preds, normalize='true')
    # df_cm = pd.DataFrame(cf_matrix_norm, index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True, cmap='Blues')
    # plt.savefig(f'{exp_dir}/{dir_name}/{"TEST" if test_set else "VAL"}{f"_YOUth" if "YOUth" in test_dir+train_dir else ""}'
    #             f'_acc{100*acc:.2f}_f1{100*f1:.2f}_norm.png')
    #
    # cf_matrix = confusion_matrix(all_labels, all_preds, normalize=None)
    # df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))
    # sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    # plt.savefig(f'{exp_dir}/{dir_name}/{"TEST" if test_set else "VAL"}{f"_YOUth" if "YOUth" in test_dir+train_dir else ""}'
    #             f'_acc{100*acc:.2f}_f1{100*f1:.2f}.png')
    # acc_no_contact, acc_contact = cf_matrix.diagonal()/cf_matrix.sum(axis=1)
    return f'{100*acc:.2f}', f'{100*f1:.2f}'


def load_model_weights(model, exp_dir, exp_name):
    models = [(file_name, int(file_name.split('_')[-1])) for file_name in sorted(os.listdir(exp_dir))
              if (exp_name in file_name) and os.path.isfile(os.path.join(exp_dir, file_name))]
    models.sort(key=lambda x: x[1])
    print(models)
    model_name = models[-1][0]
    model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))
    return model_name


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    parser.add_argument('--test_set', action='store_true', default=False, help='False: validation set, True: test set')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)
    train_dir_ssd = '~/GithubRepos/ContactClassification-ssd/FlickrCI3DSignatures/train'
    test_dir_ssd = '~/GithubRepos/ContactClassification-ssd/FlickrCI3DSignatures/test'
    exp_name = get_experiment_name(cfg)
    exp_dir = "exp/Flickr"
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(train_dir_ssd, test_dir_ssd, cfg)
    cfg.BATCH_SIZE = 1  # for accurate results
    model, _, _ = initialize_model(cfg, device)
    model_name = load_model_weights(model, exp_dir, exp_name)
    test_model(model, model_name, exp_name, exp_dir, test_loader if args.test_set else validation_loader,
               args.test_set, device, segmentation=cfg.SEGMENTATION)


if __name__ == '__main__':
    main()
