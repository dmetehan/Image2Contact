import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

from argparse import ArgumentParser
from ContactSignatureModel import ContactSignatureModel, initialize_model
from utils import parse_config, get_experiment_name, find_last_values_tensorboard, vis_pred_errors_heatmap, vis_threshold_eval

# this is important for FLickrCI3DClassification. It doesn't allow importing v2 after initializing the network.
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on {device}")
ContactSignatureModel(backbone="resnet50", weights="DEFAULT")
# net should be initialized here before importing torchvision
from dataset.YOUth_Signature_test_video import init_datasets_with_cfg


def get_predictions(model, test_loader, cfg, exp_dir="YOUth"):
    save_dir = os.path.join(exp_dir, "test_video")
    model = model.to(device)
    # We don't need gradients on to do reporting
    model.train(False)
    pred_scores = {key: [] for key in model.output_keys}
    all_preds = {key: [] for key in model.output_keys}
    all_meta = []
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            _, vinputs, vlabels, vmeta = vdata
            all_meta.append(vmeta)
            vinputs = vinputs.to(device)
            voutputs_list = model(vinputs)
            for k, key in enumerate(vlabels):
                pred_scores[key] += voutputs_list[k].detach().cpu().tolist()
                all_preds[key] += (torch.sigmoid(voutputs_list[k].detach().cpu()) > model.thresholds[key]).long().tolist()

    save_preds = {'preds': all_preds,
                  'scores': {key: torch.sigmoid(torch.Tensor(pred_scores[key])).tolist() for key in pred_scores},
                  'metadata': all_meta}
    json.dump(save_preds, open(os.path.join(save_dir, "test_video_preds.json"), 'w'))


def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset directory path')
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('model_path', help='experiment directory')

    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"{args.dataset_dir} could not be found!")
    cfg = parse_config(args.config_file)
    exp_dir = args.exp_dir
    root_dir_ssd = args.dataset_dir

    random_seed = 1525
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model, optimizer, loss_fn = initialize_model(cfg, device)
    model.load_state_dict(torch.load(args.model_path))
    test_loader = init_datasets_with_cfg(root_dir_ssd, root_dir_ssd, cfg)
    print(f'Test size: {len(test_loader.dataset)}')
    get_predictions(model, test_loader, cfg, exp_dir=exp_dir)


if __name__ == '__main__':
    main()
