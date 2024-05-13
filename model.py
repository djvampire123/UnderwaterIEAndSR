import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedEnhanceSuperResNet(nn.Module):
    def __init__(self, upscale_factor=2):
        super(UnifiedEnhanceSuperResNet, self).__init__()
        
        # Enhancement layers
        self.backscatter_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.backscatter_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.backscatter_pool = nn.AdaptiveAvgPool2d(1)
        self.backscatter_fc1 = nn.Conv2d(64, 32, kernel_size=1)
        self.backscatter_fc2 = nn.Conv2d(32, 3, kernel_size=1)

        self.direct_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.direct_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)
        self.direct_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=2)
        self.direct_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=5)
        self.direct_conv5 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        # Super-resolution layers
        self.residual_layer = self._make_layer(ResidualBlock, 5)
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def _make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # Enhancement part
        b1 = F.relu(self.backscatter_conv1(x))
        b2 = F.relu(self.backscatter_conv2(b1))
        b_pool = self.backscatter_pool(b2)
        b_fc1 = F.relu(self.backscatter_fc1(b_pool))
        backscatter = torch.sigmoid(self.backscatter_fc2(b_fc1))

        d1 = F.relu(self.direct_conv1(x))
        d2 = F.relu(self.direct_conv2(d1))
        d3 = F.relu(self.direct_conv3(d2))
        d4 = F.relu(self.direct_conv4(d3))
        direct_transmission = torch.sigmoid(self.direct_conv5(d4))

        enhanced = (x - backscatter) * direct_transmission + backscatter

        # Super-resolution part
        res = self.residual_layer(enhanced)
        upsampled = self.upsample(res)
        output = self.output_conv(upsampled)
        
        return output

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        return x + residual
