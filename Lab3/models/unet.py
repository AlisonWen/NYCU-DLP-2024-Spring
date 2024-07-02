import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_layer(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)


class conv_layer_down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_down_layers = nn.Sequential(
            nn.MaxPool2d(2),
            conv_layer(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv_down_layers(x)


class conv_layer_up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_layer(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_layer(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.init_layer = (conv_layer(n_channels, 64))
        self.down1 = (conv_layer_down(64, 128))
        self.down2 = (conv_layer_down(128, 256))
        self.down3 = (conv_layer_down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (conv_layer_down(512, 1024 // factor))
        self.up1 = (conv_layer_up(1024, 512 // factor, bilinear))
        self.up2 = (conv_layer_up(512, 256 // factor, bilinear))
        self.up3 = (conv_layer_up(256, 128 // factor, bilinear))
        self.up4 = (conv_layer_up(128, 64, bilinear))
        self.output_layer = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.init_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.output_layer(x)
        return logits
    def use_checkpointing(self):
        self.init_layer = torch.utils.checkpoint(self.init_layer)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
