import torch
import torch.nn as nn
import torch.nn.functional as F


class Convnet(nn.Module):
    def __init__(self, img_size):
        self.img_size = img_size
        super().__init__()
        base_depth = 32
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = base_depth, kernel_size = 5, stride = 2, padding=1)  #1st in_channel: color channels
        self.conv2 = nn.Conv2d(in_channels = base_depth, out_channels = base_depth * 2, kernel_size = 3, stride = 2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=base_depth * 2, out_channels= base_depth * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=base_depth * 4, out_channels=base_depth * 8, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=base_depth * 8, out_channels=base_depth * 8, kernel_size=4, stride=1, padding=0)

    def forward(self, t):
        # reshape vector into B x H x W x C
        h, w, c = self.img_size
        t = t.view(-1, h, w, c)
        # put channels first
        t = t.permute(0, 3, 1, 2)

        t = self.conv1(t)
        t = F.relu(t)

        t = self.conv2(t)
        t = F.relu(t)

        t = self.conv3(t)
        t = F.relu(t)

        t = self.conv4(t)
        t = F.relu(t)

        t = self.conv5(t)
        t = F.relu(t)

        return t
