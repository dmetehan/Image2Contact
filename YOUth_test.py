import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sn
import pandas as pd
from tqdm import tqdm
from scipy import stats

from ContactClassifier import ContactClassifier, initialize_model
from utils import parse_config, get_experiment_name

# this is important for FLickrCI3DClassification. It doesn't allow importint v2 after initializing the network.
import torchvision.transforms.v2 as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
_ = ContactClassifier()
# net should be initialized here before importing torchvision

from dataset.YOUth10mClassification import init_datasets_with_cfg

CLASSES = ("no touch", "touch")


def test_model(model, model_name, exp_name, exp_dir, data_loader, test_set, device):
    model.eval()
    model = model.to(device)
    print(f'Testing on {"test set" if test_set else "validation set"} using {model_name}')
    y_pred = []
    y_true = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(data_loader):
            _, images, labels = data
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            y_pred.extend(predicted.cpu())  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    acc = accuracy_score(y_true, y_pred)
    acc_blncd = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f'Accuracy of the network on the test images ({len(y_true)}): {acc:.4f}')
    print(f'Balanced accuracy of the network on the test images ({len(y_true)}): {acc_blncd:.4f}')
    print(f'F1 Score of the network on the test images ({len(y_true)}): {f1:.4f}')
    print(f'Precision of the network on the test images ({len(y_true)}): {precision:.4f}')
    print(f'Recall of the network on the test images ({len(y_true)}): {recall:.4f}')

    print(f"Paired t-test with all-no-contact: {stats.ttest_rel(y_pred, np.zeros_like(y_pred))}")
    np.random.seed(1)
    print(f"Paired t-test with random: {stats.ttest_rel(y_pred, np.random.randint(0, 2, len(y_pred)))}")
    print(f"Paired t-test with informed guess: "
          f"{stats.ttest_rel(y_pred, [np.random.choice(np.arange(0, 2), p=[0.5510818953711312, 0.44891810462886883]) for _ in range(len(y_pred))])}")

    print(f"Paired t-test with all-no-contact: {stats.ttest_rel(np.zeros_like(y_pred), y_pred)}")
    np.random.seed(1)
    print(f"Paired t-test with random: {stats.ttest_rel(np.random.randint(0, 2, len(y_pred)), y_pred)}")
    print(f"Paired t-test with informed guess: "
          f"{stats.ttest_rel([np.random.choice(np.arange(0, 2), p=[0.5510818953711312, 0.44891810462886883]) for _ in range(len(y_pred))], y_pred)}")

    dir_name = [file_name for file_name in sorted(os.listdir(exp_dir)) if exp_name in file_name and os.path.isdir(f'{exp_dir}/{file_name}')][-1]
    cf_matrix_norm = confusion_matrix(y_true, y_pred, normalize='true')
    df_cm = pd.DataFrame(cf_matrix_norm, index=[i for i in CLASSES],
                         columns=[i for i in CLASSES])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.savefig(f'{exp_dir}/{dir_name}/{"TEST" if test_set else "VAL"}_YOUth'
                f'_acc{100*acc:.2f}_accblncd{100*acc_blncd:.2f}_f1{100*f1:.2f}_norm.png')

    cf_matrix = confusion_matrix(y_true, y_pred, normalize=None)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in CLASSES],
                         columns=[i for i in CLASSES])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
    plt.savefig(f'{exp_dir}/{dir_name}/{"TEST" if test_set else "VAL"}_YOUth'
                f'_acc{100*acc:.2f}_accblncd{100*acc_blncd:.2f}_f1{100*f1:.2f}.png')
    return f'{100*acc:.2f}', f'{100*acc_blncd:.2f}', f'{100*f1:.2f}'


def load_model_weights(model, exp_dir, exp_name):
    print([file_name for file_name in sorted(os.listdir(exp_dir))])
    print(exp_name)
    models = [(file_name, int(file_name.split('_')[-1])) for file_name in sorted(os.listdir(exp_dir))
              if (exp_name in file_name) and os.path.isfile(os.path.join(exp_dir, file_name))]
    models.sort(key=lambda x: x[1])
    model_name = models[-1][0]
    model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))
    return model_name


def main():
    parser = ArgumentParser()
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('--test_set', action='store_true', default=False, help='False: validation set, True: test set')
    parser.add_argument('--finetune', action='store_true', default=False, help='False: no finetuning'
                                                                               'True: finetune on YOUth')
    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    cfg = parse_config(args.config_file)
    root_dir_ssd = '/home/sac/GithubRepos/ContactClassification-ssd/YOUth10mClassification/all'
    model, _, _ = initialize_model(cfg, device)
    if args.finetune:
        cfg.LR = cfg.LR / 5
    exp_name = get_experiment_name(cfg)
    if args.finetune:
        exp_name = f'finetune_{exp_name}'
    cfg.BATCH_SIZE = 1  # to get accurate results
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(root_dir_ssd, root_dir_ssd, cfg)
    model_name = load_model_weights(model, args.exp_dir, exp_name)
    test_model(model, model_name, exp_name, args.exp_dir, test_loader if args.test_set else validation_loader,
               args.test_set, device)


if __name__ == '__main__':
    main()
