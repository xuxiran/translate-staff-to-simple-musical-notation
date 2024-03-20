import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class DenseBlock(nn.Module):
    def conv_block(self,in_channels, out_channels):
        blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                            nn.ReLU(),
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        return blk

    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(self.conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # get the out channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # concat the input dim and output dim
        return X


class DenseNet_key(nn.Module):
    def transition_block(self, in_channels, out_channels):
        blk = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        return blk

    def densenet(self,channel_num=16):
        net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels, growth_rate = 64, 32  # num_channels is the currenct channels
        num_convs_in_dense_blocks = [4, 4, 4, 4]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            net.add_module("DenseBlosk_%d" % i, DB)
            # last channel
            num_channels = DB.out_channels
            # reduce the output channel
            if i != len(num_convs_in_dense_blocks) - 1:
                net.add_module("transition_block_%d" % i, self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        net.add_module("BN", nn.BatchNorm2d(num_channels))
        net.add_module("relu", nn.ReLU())
        return net

    def __init__(self,channel_num=16):
        super(DenseNet_key, self).__init__()
        self.feature = self.densenet(channel_num)
        self.linear = nn.Linear(248, 8)

    def forward(self, x):
        # Layer 1
        x = x.unsqueeze(dim=1)
        x = self.feature(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x
