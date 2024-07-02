import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)  # Xavier initialization
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv2.weight)  # Xavier initialization
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        
        # residual shortcut
        if self.stride != 1 or self.in_channels != self.out_channels:
            conv_layer = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False).to(residual.device)
            residual = conv_layer(residual)

            bn_layer = nn.BatchNorm2d(self.out_channels).to(residual.device)
            residual = bn_layer(residual)

        output += residual
        output = self.relu2(output)

        return output


class RN18(nn.Module):
    def __init__(self):
        super(RN18, self).__init__()
        self.in_channels = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)  # Apply Xavier initialization
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.layer_init(BasicBlock, 64, stride=1)
        self.layer2 = self.layer_init(BasicBlock, 128, stride=2)
        self.layer3 = self.layer_init(BasicBlock, 256, stride=2)
        self.layer4 = self.layer_init(BasicBlock, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def layer_init(self, block, out_channels, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        # 2 blocks
        layers.append(block(out_channels, out_channels, stride=1))
        layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output