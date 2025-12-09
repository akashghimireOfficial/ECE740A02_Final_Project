import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


## conv block with SiLU + dropout
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),

            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.conv(x)


## encoder block
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c, dropout=0.1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


## decoder block
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c, dropout=0.1)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


## small UNet from scratch
class SmallUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b = conv_block(512, 1024, dropout=0.2)

        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b = self.b(p4)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        return self.out(d4)


## UNet with pretrained ResNet encoder
class ResNetUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        ## encoder
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool0  = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        ## decoder
        self.up4 = decoder_block(512, 256)
        self.up3 = decoder_block(256, 128)
        self.up2 = decoder_block(128, 64)
        self.up1 = decoder_block(64, 64)

        ## classifier
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        ## encoder forward
        s0 = self.layer0(x)
        s1 = self.layer1(self.pool0(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)

        ## decoder forward
        d4 = self.up4(s4, s3)
        d3 = self.up3(d4, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)

        ## raw output is 1/2 resolution → fix
        out = self.out(d1)

        ## FIX → restore exact input resolution
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


## model factory
def build_model(name, num_classes):
    name = name.lower()

    if name == "small":
        return SmallUNet(num_classes)

    if name == "resnet":
        return ResNetUNet(num_classes)

    raise ValueError(f"unknown model type: {name}")
