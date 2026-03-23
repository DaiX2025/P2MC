import torch
import torch.nn as nn

from einops import rearrange


class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c h w d -> b h w d c')
        x = self.norm(x)
        
        return rearrange(x, 'b h w d c -> b c h w d')


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_type='reflect', relufactor=0.2):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=pad_type, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.act = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.InstanceNorm3d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        residual = x.clone()

        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = self.act(x)

        return x


class FusionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_type='reflect', relufactor=0.2):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm3d(out_channels)

        self.act = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))

        return x


class UpBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_type='reflect', relufactor=0.2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=pad_type)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm3d(out_channels)

        self.act = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x, skip):
        x = self.up(x)

        x = self.act(self.norm1(self.conv1(x)))

        x = torch.cat((x, skip), dim=1)

        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))

        return x


class OutBlock(nn.Module):
    
    def __init__(self, in_channels, hidden_channels=None, num_classes=4, kernel_size=1, stride=1, padding=0, pad_type='reflect', relufactor=0.2):
        super().__init__()

        hidden_channels = hidden_channels or in_channels//2

        self.conv = nn.Conv3d(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.norm = nn.InstanceNorm3d(hidden_channels)
        self.act = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

        self.out = nn.Conv3d(hidden_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        x = self.softmax(self.out(x))
        
        return x
