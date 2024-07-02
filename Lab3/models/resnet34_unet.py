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


class RN34(nn.Module):
    def __init__(self, num_classes=1000):
        super(RN34, self).__init__()
        # Initial convolution
        # self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.in_channels)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.init_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # ResNet layers
        self.layer1 = self.layer_init(in_channels=64, out_channels=64, blocks=3, stride=1)
        self.layer2 = self.layer_init(in_channels=64, out_channels=128, blocks=4, stride=2)
        self.layer3 = self.layer_init(in_channels=128, out_channels=256, blocks=6, stride=2)
        self.layer4 = self.layer_init(in_channels=256, out_channels=512, blocks=3, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def layer_init(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride))

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
    
class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv_layers(x)


class conv_layer_up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_layer(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_layer(in_channels, out_channels)

    def forward(self, x1, x2=None):
        if x2 is None:
            x = self.up(x1)
            x = self.conv(x)
            return x
        # print('In Conv Layer Forward:', x1.shape, x2.shape)
        x2 = self.up(x2)
        # print('scaled x2:', x2.shape)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # print('padded x2:', x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('x =', x.shape)
        # print('-'*50)
        return self.conv(x)

class resnet34_unet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.resnet34_encoder = RN34(num_classes)
        
        # self.up1 = conv_layer_up(768, 512)
        # self.up2 = conv_layer_up(640, 256)
        # self.up3 = conv_layer_up(320, 128)
        # self.up4 = conv_layer(128, 64)
        self.up1 = conv_layer_up(768, 512)
        self.up2 = conv_layer_up(640, 256)
        self.up3 = conv_layer_up(320, 128)
        self.up4 = conv_layer_up(192, 64)
        self.up5 = conv_layer_up(64, 32)
        self.up6 = conv_layer_up(32, 32)
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        # print('x shape:', x.shape)
        x1 = self.resnet34_encoder.init_layers(x)
        # print('init_layers output shape:', x1.shape)
        x2 = self.resnet34_encoder.layer1(x1)
        # print('layer1 output shape:', x2.shape)
        x3 = self.resnet34_encoder.layer2(x2)
        # print('layer2 output shape:', x3.shape)
        x4 = self.resnet34_encoder.layer3(x3)
        x5 = self.resnet34_encoder.layer4(x4)
        
        x6 = self.up1(x4, x5)
        x7 = self.up2(x3, x6)
        x8 = self.up3(x2, x7)
        x9 = self.up4(x1, x8)
        x10 = self.up5(x9)
        x11 = self.up6(x10)
        
        return self.output_layer(x11)
        
