import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(conv_layer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG19, self).__init__()
        self.stage1 = self.layer_init(3, 64, kernel_size=3, stride=1, padding=1, num_blocks=2)
        self.stage2 = self.layer_init(64, 128, kernel_size=3, stride=1, padding=1, num_blocks=2)
        self.stage3 = self.layer_init(128, 256, kernel_size=3, stride=1, padding=1, num_blocks=4)
        self.stage4 = self.layer_init(256, 512, kernel_size=3, stride=1, padding=1, num_blocks=4)
        self.stage5 = self.layer_init(512, 512, kernel_size=3, stride=1, padding=1, num_blocks=4)
        
        self.tail = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        ])
        
        
        
    def layer_init(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, num_blocks=1):
        layers = [conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        for _ in range(1, num_blocks):
            layers.append(conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.tail(x)
        return x
        