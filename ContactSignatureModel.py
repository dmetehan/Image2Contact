import torch
from torch import nn, optim

from torch.nn import MultiLabelSoftMarginLoss
from models.custom_loss import IoUBCELoss
from utils import Options


class ContactSignatureModel(nn.Module):
    def __init__(self, backbone, weights, option=Options.debug, copy_rgb_weights=False,
                 finetune=False):
        super(ContactSignatureModel, self).__init__()
        print(backbone)
        resnet = torch.hub.load("pytorch/vision", backbone, weights=weights)
        conv1_pretrained = list(resnet.children())[0]
        if option == Options.rgb:
            self.conv1 = conv1_pretrained
        else:
            input_size = -1
            if option in [Options.jointmaps]:
                input_size = 34
            elif option in [Options.jointmaps_rgb]:
                input_size = 37
            elif option in [Options.jointmaps_rgb_bodyparts, Options.jointmaps_bodyparts_opticalflow]:
                input_size = 52
            elif option in [Options.jointmaps_rgb_bodyparts_opticalflow]:
                input_size = 55
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
            if option in [Options.jointmaps_rgb, Options.jointmaps_rgb_bodyparts, Options.jointmaps_rgb_bodyparts_opticalflow]:
                self.conv1.weight.data[:, 34:37, :, :] = conv1_pretrained.weight  # copy the weights to the rgb channels
            elif option in [Options.rgb, Options.rgb_bodyparts]:
                self.conv1.weight.data[:, :3, :, :] = conv1_pretrained.weight  # copies the weights to the rgb channels
        modules = list(resnet.children())[1:-1]
        resnet = nn.Sequential(*modules)

        if finetune:
            print('Freezing the first convolutional layer!')
            for name, param in resnet.named_parameters():
                if param.requires_grad and ('0.weight' == name or '0.bias' == name
                                            or '3.0.bn1' in name or '3.0.conv1' in name):
                    print(name)
                    param.requires_grad = False

        # for name, param in resnet50.named_parameters():
        #     if 'bn' in name:
        #         print(name)
        # print(list(resnet50.named_parameters()))
        # print(resnet50)
        #self.feat_extractor = resnet50
        self.feat_extractor = resnet
        self.thresholds = {'42': 0.3, '12': 0.5, '21*21': 0.1, '6*6': 0.2}
        self.output_keys = list(self.thresholds.keys())
        in_features = 2048 if backbone == 'resnet50' else 512
        self.fc42 = nn.Linear(in_features=in_features, out_features=42, bias=True)
        self.fc12 = nn.Linear(in_features=in_features, out_features=12, bias=True)
        # self.fc21adult = nn.Linear(in_features=2048, out_features=21, bias=True)
        # self.fc21child = nn.Linear(in_features=2048, out_features=21, bias=True)
        # self.fc6adult = nn.Linear(in_features=2048, out_features=6, bias=True)
        # self.fc6child = nn.Linear(in_features=2048, out_features=6, bias=True)

        self.fc21x21 = nn.Linear(in_features=in_features, out_features=21*21, bias=True)
        self.fc6x6 = nn.Linear(in_features=in_features, out_features=6*6, bias=True)

        # self.fc21adult = nn.Linear(in_features=42, out_features=21, bias=True)
        # self.fc21child = nn.Linear(in_features=42, out_features=21, bias=True)
        # self.fc6adult = nn.Linear(in_features=12, out_features=6, bias=True)
        # self.fc6child = nn.Linear(in_features=12, out_features=6, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.feat_extractor(x)
        x = torch.flatten(x, 1)
        x42 = self.fc42(x)
        x12 = self.fc12(x)

        # x21adult = self.fc21adult(torch.relu(x42))
        # x6adult = self.fc6adult(torch.relu(x12))
        # x21child = self.fc21child(torch.relu(x42))
        # x6child = self.fc6child(torch.relu(x12))

        # x21adult = self.fc21adult(x)
        # x6adult = self.fc6adult(x)
        # x21child = self.fc21child(x)
        # x6child = self.fc6child(x)

        # # xadult * xchild.T
        # x21x21 = torch.bmm(x21adult.reshape(-1, 21, 1), x21child.reshape(-1, 1, 21)).reshape(-1, 21*21)
        # x6x6 = torch.bmm(x6adult.reshape(-1, 6, 1), x6child.reshape(-1, 1, 6)).reshape(-1, 6*6)

        x21x21 = self.fc21x21(x)
        x6x6 = self.fc6x6(x)

        return x42, x12, x21x21, x6x6


def initialize_model(cfg, device, finetune=False):
    model = ContactSignatureModel(backbone=cfg.BACKBONE, weights="DEFAULT", option=cfg.OPTION, copy_rgb_weights=cfg.COPY_RGB_WEIGHTS, finetune=finetune)
    # MultiLabelSoftMargin loss is also good but needs to be adjusted for dictionary output from the model
    # loss_fn = IoUBCELoss()
    loss_fn = MultiLabelSoftMarginLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
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
