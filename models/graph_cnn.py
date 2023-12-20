"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
from torch import nn, optim

from .graph_layers import GraphResBlock, GraphLinear
from .resnet import resnet50
from utils import get_adj_mat


class GraphCNN(nn.Module):

    def __init__(self, A, num_layers=5, num_channels=512, output_size=2):
        super(GraphCNN, self).__init__()
        self.A = A
        self.resnet = resnet50(pretrained=True)
        layers = [GraphLinear(2048, 2 * num_channels),
                  GraphResBlock(2 * num_channels, num_channels, A)]
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 1))
        self.gc = nn.Sequential(*layers)
        self.fc = nn.Linear(21, output_size, bias=True)

    def forward(self, image):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 21, 3)
        """
        batch_size = image.shape[0]
        image_enc = self.resnet(image)
        x = image_enc.reshape((batch_size, 2048, 1)).expand(-1, -1, 21)  # copy it N region times
        x = self.gc(x)
        shape = self.shape(x)
        y = self.fc(shape)
        return y.reshape((-1, 2))


def initialize_gcn_model(cfg, device, segmentation=False, finetune=False):
    adjmat = get_adj_mat()
    adjmat = torch.Tensor(adjmat).to(device)
    model = GraphCNN(adjmat, num_channels=256, num_layers=5, output_size=(21+21) if segmentation else (21*21)).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    return model, optimizer, loss_fn

