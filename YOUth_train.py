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
from utils import parse_config, get_experiment_name, find_last_values_tensorboard

# this is important for FLickrCI3DClassification. It doesn't allow importing v2 after initializing the network.
import torchvision.transforms.v2 as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Working on {device}")
ContactSignatureModel(backbone="resnet50", weights="DEFAULT")
# net should be initialized here before importing torchvision
from dataset.YOUth_Signature import init_datasets_with_cfg


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def train_one_epoch(model, optimizer, loss_fn, loss_weights, train_loader, epoch_index, tb_writer, batch_size, multitask=True):
    overall_loss = 0.
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    i = 0
    all_preds = {key: np.zeros((len(train_loader.dataset), eval(key), batch_size)) for key in model.output_keys}
    all_labels = {key: np.zeros((len(train_loader.dataset), eval(key), batch_size)) for key in model.output_keys}
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        _, inputs, labels = data
        for key in all_labels:
            all_labels[key][i, :, :len(labels[key])] = labels[key].T
            labels[key] = labels[key].to(device)
        inputs = inputs.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs_list = model(inputs)
        loss = torch.zeros(1).to(device)
        for k, key in enumerate(labels):
            preds = outputs_list[k].T.detach().cpu()
            preds[preds >= model.thresholds[key]] = 1
            preds[preds < model.thresholds[key]] = 0
            all_preds[key][i, :, :len(labels[key])] = preds
            # Compute the loss and its gradients
            loss += loss_weights[key] * loss_fn(outputs_list[k], labels[key].float())
            #print(key, loss)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        cur_loss = loss.detach().item()
        running_loss += cur_loss
        tb_x = epoch_index * len(train_loader) + i + 1
        tb_writer.add_scalar('Loss/train', cur_loss, tb_x)
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('\tbatch {} loss: {}'.format(i + 1, last_loss))
            overall_loss += running_loss
            running_loss = 0.
    jaccard = {}
    for key in all_labels:
        all_preds[key] = np.swapaxes(all_preds[key], 1, 2).reshape(-1, eval(key))[:-(batch_size - len(labels[key])), :]
        all_labels[key] = np.swapaxes(all_labels[key], 1, 2).reshape(-1, eval(key))[:-(batch_size - len(labels[key])), :]
        jaccard[key] = jaccard_score(all_labels[key], all_preds[key], average='micro', zero_division=0)
    # acc = accuracy_score(all_labels, all_preds)
    # f1 = f1_score(all_labels, all_preds, average="micro")
    if i % 1000 != 999:
        last_loss = running_loss / (i % 1000 + 1)  # loss per batch
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        overall_loss += running_loss
    overall_loss = overall_loss / len(train_loader)
    return overall_loss, jaccard


def train_model(model, optimizer, scheduler, loss_fn_train, experiment_name, cfg, train_loader, val_loader,
                exp_dir="YOUth", start_epoch=0, resume=False):
    multitask = cfg.MULTITASK
    loss_weights = {key: cfg.LOSS_WEIGHTS[k]  for k, key in enumerate(model.output_keys)}
    #loss_fn_valid = nn.BCEWithLogitsLoss()  # using IOU as loss for validation doesn't work
    loss_fn_valid = loss_fn_train
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)
    best_model_path = ''
    if resume:
        timestamps = ['_'.join(folder_name.split('_')[-2:]) for folder_name in sorted(os.listdir(exp_dir))
                      if (experiment_name in folder_name) and os.path.isdir(os.path.join(exp_dir, folder_name))]
        timestamp = sorted(timestamps)[-1]
        log_dir = f'{exp_dir}/{experiment_name}_{timestamp}/Training vs. Validation Loss_Validation'
        best_vloss = find_last_values_tensorboard(log_dir, 'Training vs. Validation Loss')
        print("Best validation loss so far:", best_vloss)
        log_dir = f'{exp_dir}/{experiment_name}_{timestamp}/Training vs. Validation Balanced Accuracy_Validation'
        best_vacc_blncd = find_last_values_tensorboard(log_dir, 'Training vs. Validation Balanced Accuracy')
        print("Best validation balanced accuracy so far:", best_vacc_blncd)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_vloss = 1_000_000.
        best_vacc_blncd = 0.0
    writer = SummaryWriter('{}/{}_{}'.format(exp_dir, experiment_name, timestamp))

    model = model.to(device)
    for epoch in range(start_epoch, cfg.EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, jaccard = train_one_epoch(model, optimizer, loss_fn_train, loss_weights, train_loader, epoch, writer, cfg.BATCH_SIZE)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        i = 0
        all_preds = {key: np.zeros((len(val_loader.dataset), eval(key), cfg.BATCH_SIZE)) for key in model.output_keys}
        all_labels = {key: np.zeros((len(val_loader.dataset), eval(key), cfg.BATCH_SIZE)) for key in model.output_keys}
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                _, vinputs, vlabels = vdata
                for key in all_labels:
                    all_labels[key][i, :, :len(vlabels[key])] = vlabels[key].T
                    vlabels[key] = vlabels[key].to(device)
                vinputs = vinputs.to(device)

                vloss = torch.zeros(1).to(device)
                voutputs_list = model(vinputs)
                for k, key in enumerate(vlabels):
                    vpreds = voutputs_list[k].T.detach().cpu()
                    vpreds[vpreds >= model.thresholds[key]] = 1
                    vpreds[vpreds < model.thresholds[key]] = 0
                    all_preds[key][i, :, :len(vlabels[key])] = vpreds
                    vloss += loss_weights[key] * loss_fn_valid(voutputs_list[k], vlabels[key].float())
                running_vloss += vloss.detach().item()

        vjaccard = {}
        for key in all_labels:
            all_preds[key] = np.swapaxes(all_preds[key], 1, 2).reshape(-1, eval(key))[:-(cfg.BATCH_SIZE - len(vlabels[key])), :]
            all_labels[key] = np.swapaxes(all_labels[key], 1, 2).reshape(-1, eval(key))[:-(cfg.BATCH_SIZE - len(vlabels[key])), :]
            vjaccard[key] = jaccard_score(all_labels[key], all_preds[key], average='micro', zero_division=0)
        avg_vloss = running_vloss / (i + 1)
        scheduler.step(avg_vloss)
        print('LOSS train {:.4f} valid {:.4f} - Jaccard train {} valid {}'.format(avg_loss,
                                                                                  avg_vloss,
                                                                                  ','.join([f'{key}: {jaccard[key].item():.2%}' for key in model.output_keys]),
                                                                                  ','.join([f'{key}: {vjaccard[key].item():.2%}' for key in model.output_keys])))
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        for key in model.output_keys:
            writer.add_scalars('Training vs. Validation Jaccard',
                               {'Training'+key: jaccard[key], 'Validation'+key: vjaccard[key]},
                               epoch + 1)
        writer.flush()

        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = '{}/{}_{}_{}'.format(exp_dir, experiment_name, timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
        # if vacc_blncd > best_vacc_blncd:
        #     best_vacc_blncd = vacc_blncd
        #     model_path = '{}/{}_{}_{}'.format(exp_dir, experiment_name, timestamp, epoch + 1)
        #     torch.save(model.state_dict(), model_path)
        #     best_model_path = model_path
        # early stopping
        early_stopping(avg_loss, avg_vloss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:", i)
            break

    print('Finished Training')
    print(f'Best model is saved at: {best_model_path}')
    return best_model_path


def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset directory path')
    parser.add_argument('config_file', help='config file')
    parser.add_argument('exp_dir', help='experiment directory')
    parser.add_argument('--resume', action='store_true', default=False, help='False: start from scratch, ')
    parser.add_argument('--test', action='store_true', default=False, help='False: no testing'
                                                                           'True: testing on the test set at the end')
    parser.add_argument('--log_test_results', action='store_true', default=False, help='False: no logging'
                                                                                       'True: logging test results')

    args = parser.parse_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"{args.config_file} could not be found!")
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"{args.dataset_dir} could not be found!")
    cfg = parse_config(args.config_file)
    exp_dir = args.exp_dir
    data_dir = args.dataset_dir
    root_dir_ssd = os.path.join(data_dir, "all")

    random_seed = 1525
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model, optimizer, loss_fn = initialize_model(cfg, device)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    experiment_name = get_experiment_name(cfg)
    start_epoch = 0
    if args.resume:
        models = [(file_name, int(file_name.split('_')[-1])) for file_name in sorted(os.listdir(exp_dir))
                              if (experiment_name in file_name) and os.path.isfile(os.path.join(exp_dir, file_name))]
        models.sort(key=lambda x: x[1])
        model_name, start_epoch = models[-1]
        model.load_state_dict(torch.load(f"{exp_dir}/{model_name}"))
    train_loader, validation_loader, test_loader = init_datasets_with_cfg(root_dir_ssd, root_dir_ssd, cfg)
    print(f'Training size: {len(train_loader.dataset)}, Validation size: {len(validation_loader.dataset)}, '
          f'Test size: {len(test_loader.dataset)}')
    best_model_path = train_model(model, optimizer, scheduler, loss_fn, experiment_name, cfg, train_loader,
                                  validation_loader, exp_dir=exp_dir, start_epoch=start_epoch, resume=args.resume)
    if args.test:
        from test import test_model
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        model = model.to(device)
        acc, f1 = test_model(model, best_model_path, experiment_name, exp_dir, test_loader, True, device,
                             cfg.MULTITASK)
        if args.log_test_results:
            with open("test_results.txt", 'a+') as file1:
                file1.write(f"{best_model_path}, {acc}, {f1}\n")


if __name__ == '__main__':
    main()
