import torch
import torch.nn as nn


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(act="silu", inplace=True):
    if act == "silu":
        active = nn.SiLU(inplace=True)
    elif act == "relu" :
        active = nn.ReLU(inplace=True)
    elif act == "lrelu":
        active = nn.LeakyReLU(0.1, inplace=True)
    else:
        raise AttributeError("Unsuppored act type:{} ".format(act))
    return active


# 模型中的CBL块
class BaseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, group, bias=False, act="relu"):
        super(BaseConv, self).__init__()

        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding= pad,
                              groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = get_activation(act, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


#模型中的Resunit模块
class ResBlock(nn.Module):
    "Residual layer with 'in_channels' inputs"

    def __init__(self, in_channel:int):
        mid_channel = in_channel // 2
        self.block1 = BaseConv(in_channel, mid_channel, kernel=3, stride=1, group=1, act="lrelu")
        self.block2 = BaseConv(mid_channel,in_channel, kernel=3, stride=1, group=1, act="lrelu")
    def forward(self, input_x):
        out = self.block2(self.block1(input_x))
        return input_x + out


#SPP模块
class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
            self, in_channel, out_channel, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super(SPPBottleneck, self).__init__()
        hidden_channels = in_channel // 2
        # pad = (kernel - 1) // 2
        self.conv1 = nn.Conv2d(in_channel, hidden_channels,  1, stride=1, act = activation)

        self.pooling = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding= ks//2)
            for ks in kernel_sizes
        ])
        conv_channel = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv_channel,  out_channel, kernel_sizes=1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.pooling], dim=1)
        x = self.conv2(x)
        return x

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat( (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

#模型中的CSP层
class CSPLayer(nn.Module):
    def __init__(self, in_channel, out_channel, n=1, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden_channels = int(out_channel * expansion)
        self.conv1 = BaseConv(in_channel, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channel, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2*hidden_channels, out_channel, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)



#采样头
class Pre_Head(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.block1 = BaseConv()
        self.block2_1 = BaseConv()
        self.block2_2 = BaseConv()
        self.block3_1 = BaseConv()
        self.block3_2 = BaseConv()
        self.conv1 = nn.Conv2d(in_channel, 80)
        self.conv2 = nn.Conv2d(in_channel, 1)
        self.conv3 = nn.Conv2d( in_channel, 4)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x1 = self.block2_1(x)
        x1 = self.block2_2(x1)

        x2 = self.block3_1(x1)
        x2 = self.block3_2(x2)

        head1 = self.conv1(x1)
        head1 = self.act1(head1)
        head2 = self.conv2(x2)
        head2 = self.act2(head2)
        head3 = self.conv3(x2)

        headx = torch.cat([head1, head2, head3], dim=1)

        return headx






