import torch
import torch.nn as nn
import base_block as bb

class DarkNet53(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DarkNet53, self).__init__()
        self.block1 = bb.BaseConv()
        self.block2 = bb.ResBlock()

        self.block3 = nn.ModuleList([bb.ResBlock() for i in range(2)])

        self.block4 = nn.ModuleList([bb.ResBlock() for i in range(8)])
        self.block5 = nn.ModuleList([bb.ResBlock() for i in range(8)])
        self.block6 = nn.ModuleList([bb.ResBlock() for i in range(4)])
        self.block7_1 = bb.BaseConv()
        self.block7_2 = bb.BaseConv()

        self.block8 = bb.SPPConv()
        self.block9_1 = bb.BaseConv()
        self.block9_2 = bb.BaseConv()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        feature_1 = x
        x = self.block5(x)
        feature_2 = x
        x = self.block6(x)
        x = self.block7_1(x)
        x = self.block7_2(x)
        x = self.block8(x)
        x = self.block9_1(x)
        x = self.block9_2(x)

        return x, feature_1, feature_2


class Neck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Neck, self).__init__()
        self.neck1 = bb.BaseConv()
        #上采样
        self.up1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

        self.neck2 = nn.ModuleList([bb.BaseConv() for i in range(5)])

        self.neck3 = bb.BaseConv()
        #上采样
        self.up2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)

        self.neck4 = nn.ModuleList([bb.BaseConv() for i in range(5)])

    def forward(self, ):
        x, feature1, feature2 = DarkNet53()
        x1 = self.neck1(x)
        x1 = self.up1(x1)
        x1 = torch.cat([feature1, x1], dim=1)
        x1 = self.neck2(x1)

        x2 = self.neck3(x1)
        x2 = self.up2(x2)
        x2 = torch.cat([feature2, x2], dim=1)
        x2 = self.neck4(x2)

        return x, x1, x2







