import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, precision_score, recall_score
import seaborn as sn
import pandas as pd
from tqdm import tqdm
from scipy import stats

from ContactSignatureModel import ContactSignatureModel, initialize_model
from utils import parse_config, get_experiment_name, vis_pred_errors_heatmap, vis_threshold_eval

# this is important for FLickrCI3DClassification. It doesn't allow importint v2 after initializing the network.
import torchvision.transforms.v2 as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
ContactSignatureModel(backbone="resnet50", weights="DEFAULT")
# net should be initialized here before importing torchvision
from dataset.YOUth_Signature import init_datasets_with_cfg


def test_model(model, model_name, save_dir, data_loader, test_set, device):
    model.eval()
    model = model.to(device)
    print(f'Testing on {"test set" if test_set else "validation set"} using {model_name}')
    pred_scores = {key: [] for key in model.output_keys}
    all_labels = {key: [] for key in model.output_keys}
    all_preds = {key: [] for key in model.output_keys}
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, vdata in enumerate(data_loader):
            _, inputs, labels = vdata
            for key in all_labels:
                labels[key] = labels[key].to(device)
                all_labels[key] += labels[key].detach().cpu().tolist()
            inputs = inputs.to(device)

            outputs_list = model(inputs)
            for k, key in enumerate(labels):
                pred_scores[key] += outputs_list[k].detach().cpu().tolist()
                all_preds[key] += (torch.sigmoid(outputs_list[k].detach().cpu()) > model.thresholds[key]).long().tolist()

    vis_pred_errors_heatmap(all_labels['42'], all_preds['42'], save_dir)

    jaccard = {}
    for key in all_labels:
        jaccard[key] = jaccard_score(all_labels[key], all_preds[key], average='micro', zero_division=0)
    kwargs = {'epoch': 0, 'save_dir': os.path.join(save_dir, 'thresholding', 'test' if test_set else 'val'), 'average': 'micro'}
    vis_threshold_eval(all_labels, pred_scores, jaccard_score, **kwargs)
    print('Jaccard {} {}'.format('test' if test_set else 'val',
                                 ','.join([f'{key}: {jaccard[key].item():.2%}' for key in model.output_keys])))


def load_model_weights(model, exp_dir, exp_name):
    print([file_name for file_name in sorted(os.listdir(exp_dir))])
    print(exp_name)
    timestamps = ['_'.join(folder_name.split('_')[-2:]) for folder_name in sorted(os.listdir(exp_dir))
                  if (exp_name in folder_name) and os.path.isdir(os.path.join(exp_dir, folder_name))]
    timestamp = sorted(timestamps)[-1]
    print(timestamp)
    models = [(file_name, int(file_name.split('_')[-1]), '_'.join(file_name.split('_')[-3:-1])) for file_name in sorted(os.listdir(exp_dir))
              if (exp_name in file_name) and os.path.isfile(os.path.join(exp_dir, file_name))]
    models = [model_tuple for model_tuple in models if model_tuple[-1] == timestamp]
    models.sort(key=lambda x: x[1])
    model_name = models[-1][0]
    model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))
    return model_name, os.path.join(exp_dir, f'{exp_name}_{timestamp}')


def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset directory path')
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('--test_set', action='store_true', default=False, help='False: validation set, True: test set')
    parser.add_argument('--finetune', action='store_true', default=False, help='False: no finetuning'
                                                                               'True: finetune on YOUth')

    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"{args.dataset_dir} could not be found!")
    cfg = parse_config(args.config_file)
    data_dir = args.dataset_dir
    root_dir_ssd = os.path.join(data_dir, "all")
    model, _, _ = initialize_model(cfg, device)
    if args.finetune:
        cfg.LR = cfg.LR / 5
    exp_name = get_experiment_name(cfg)
    if args.finetune:
        exp_name = f'finetune_{exp_name}'
    cfg.BATCH_SIZE = 1  # to get accurate results
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(root_dir_ssd, root_dir_ssd, cfg)
    model_name, save_dir = load_model_weights(model, args.exp_dir, exp_name)
    test_model(model, model_name, save_dir, test_loader if args.test_set else validation_loader,
               args.test_set, device)


if __name__ == '__main__':
    main()
