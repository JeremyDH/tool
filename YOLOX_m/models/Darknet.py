import torch
import torch.nn as nn
from base_block import BaseConv, Bottleneck, ResBlock, CSPLayer, Focus,SPPBottleneck


class CSPDarnet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_feature=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__()
        assert out_feature
        self.out_feature = out_feature

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        #focus
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        #dark2
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels*2, 3, 2, act=act),
            CSPLayer(base_channels * 2,
                     base_channels * 2,
                     n = base_depth,
                     act = act)
        )
        # dark3
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
    def forward(self, x):
        output = {}
        x = self.stem(x)
        output["stem"] = x
        x = self.dark2(x)
        output["dark2"] = x
        x = self.dark3(x)
        output["dark3"] = x
        x = self.dark4(x)
        output["dark4"] = x
        x = self.dark5(x)
        output["dark5"] = x

        return {k : v for k, v in output.items() if k in self.out_feature}