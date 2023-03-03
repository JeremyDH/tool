import math

import torch
import torch.nn as nn
from base_block import BaseConv
from losses import IOUloss

class Head(nn.Module):
    def __init__(self, num_class, width=1.0, strides=[8, 16, 32],
                 in_channels=[256, 512, 1024], act="silu"):
        super().__init__()
        self.num_classes = num_class
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_convs = nn.ModuleList()
        self.stem = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stem.append(
                BaseConv(
                    in_channels = int(in_channels[i] * width),
                    out_channels = int(256 * width),
                    kernel=1,
                    stride=1,
                    act=act
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(
                            in_channels = int(256 * width),
                            out_channel= int(256 * width),
                            kernel=1,
                            stride=1,
                            act=act

                        ),
                        BaseConv(
                            in_channels = int(256 * width),
                            out_channel= int(256 * width),
                            kernel=1,
                            stride=1,
                            act = act
                        )
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        BaseConv(
                            in_channels=int(256 * width),
                            out_channel=int(256 * width),
                            kernel=1,
                            stride=1,
                            act=act

                        ),
                        BaseConv(
                            in_channels=int(256 * width),
                            out_channel=int(256 * width),
                            kernel=1,
                            stride=1,
                            act=act
                        )
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256*width),
                          out_channels=self.num_classes,
                          kernel_size=1,
                          stride=1,
                          padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256*width),
                          out_channels= 4,
                          kernel_size= 1,
                          stride= 1,
                          padding=0)
            )
            self.obj_convs.append(
                nn.Conv2d(in_channels=int(256*width),
                          out_channels= 1,
                          kernel_size= 1,
                          stride= 1,
                          padding=0)
            )
        self.use_l1 = False
        self.L1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
    def initialize_biases(self, prior_prob):
        for conv in self.cls_convs:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_convs:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
