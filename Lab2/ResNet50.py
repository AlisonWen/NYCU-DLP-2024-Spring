import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.residual = None
        if stride != 1 or in_channels != out_channels * BottleneckBlock.expansion:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleneckBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleneckBlock.expansion),
            )

        self.stride = stride

    def forward(self, x):
        identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu1(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.residual is not None:
            identity = self.residual(x)

        output += identity
        output = self.relu3(output)

        return output

class RN50(nn.Module):
    def __init__(self, num_classes=1000):
        super(RN50, self).__init__()
        self.in_channels = 64
        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self.layer_init(out_channels=64, blocks=3, stride=1)
        self.layer2 = self.layer_init(out_channels=128, blocks=4, stride=2)
        self.layer3 = self.layer_init(out_channels=256, blocks=6, stride=2)
        self.layer4 = self.layer_init(out_channels=512, blocks=3, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

    def layer_init(self, out_channels, blocks, stride):
        layers = []
        layers.append(BottleneckBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * BottleneckBlock.expansion
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
