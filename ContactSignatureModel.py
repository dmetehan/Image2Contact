import torch
from torch import nn, optim

from models.custom_loss import IoUBCELoss
from utils import Options


class ContactSignatureModel(nn.Module):
    def __init__(self, backbone="resnet50", weights="IMAGENET1K_V2", option=Options.debug, copy_rgb_weights=False,
                 finetune=False):
        super(ContactSignatureModel, self).__init__()
        resnet50 = torch.hub.load("pytorch/vision", backbone, weights=weights)
        conv1_pretrained = list(resnet50.children())[0]
        if option == Options.rgb:
            self.conv1 = conv1_pretrained
        else:
            input_size = -1
            if option in [Options.jointmaps]:
                input_size = 34
            elif option in [Options.jointmaps_rgb]:
                input_size = 37
            elif option in [Options.jointmaps_rgb_bodyparts]:
                input_size = 52
            elif option in [Options.bodyparts]:
                input_size = 15
            elif option in [Options.rgb_bodyparts]:
                input_size = 18
            elif option in [Options.jointmaps_bodyparts]:
                input_size = 49
            elif option in [Options.jointmaps_bodyparts_depth]:
                input_size = 50
            elif option in [Options.debug, Options.depth]:
                input_size = 1
            self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if copy_rgb_weights:
            if option in [Options.jointmaps_rgb, Options.jointmaps_rgb_bodyparts]:
                self.conv1.weight.data[:, 34:37, :, :] = conv1_pretrained.weight  # copy the weights to the rgb channels
            elif option in [Options.rgb, Options.rgb_bodyparts]:
                self.conv1.weight.data[:, :3, :, :] = conv1_pretrained.weight  # copies the weights to the rgb channels
        modules = list(resnet50.children())[1:-1]
        resnet50 = nn.Sequential(*modules)
        if finetune:
            print('Freezing the first convolutional layer!')
            for name, param in resnet50.named_parameters():
                if param.requires_grad and ('0.weight' == name or '0.bias' == name
                                            or '3.0.bn1' in name or '3.0.conv1' in name):
                    print(name)
                    param.requires_grad = False

        # for name, param in resnet50.named_parameters():
        #     if 'bn' in name:
        #         print(name)
        # print(list(resnet50.named_parameters()))
        # print(resnet50)
        self.feat_extractor = resnet50
        self.output_keys = ['42', '12', '21*21', '6*6']
        self.fc = [nn.Linear(in_features=2048, out_features=eval(key), bias=True) for key in self.output_keys]

    def forward(self, x):
        x = self.conv1(x)
        x = self.feat_extractor(x)
        x = torch.flatten(x, 1)
        x_list = [self.fc[k](x) for k in range(len(self.output_keys))]

        return x_list


def initialize_model(cfg, device, finetune=False):
    model = ContactSignatureModel(backbone="resnet50", weights="IMAGENET1K_V2" if cfg.PRETRAINED else None,
                                  option=cfg.OPTION, copy_rgb_weights=cfg.COPY_RGB_WEIGHTS, finetune=finetune)
    loss_fn = IoUBCELoss()  # MultiLabelSoftMargin loss is also good but needs to be adjusted for dictionary output from the model
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    return model, optimizer, loss_fn


def init_model(option=Options.jointmaps):
    """
    deprecated
    """
    model = ContactSignatureModel(backbone="resnet50", weights="IMAGENET1K_V2", option=option)
    return model


def main():
    model = init_model(option=Options.jointmaps_rgb)
    random_data = torch.rand((1, 37, 512, 256))
    result = model(random_data)
    print(result)
    print(model)
    loss_fn = nn.BCEWithLogitsLoss()
    # Compute the loss and its gradients
    loss = loss_fn(result, torch.zeros(len(result), dtype=int))
    print(list(list(list(model.children())[0].children())[4].children()))
    loss.backward()
    for bneck in list(list(model.children())[0].children())[4].children():
        for name, layer in bneck.named_children():
            try:
                if layer.weight.grad is not None:
                    print(name, layer)
            except AttributeError:
                continue


if __name__ == '__main__':
    main()
