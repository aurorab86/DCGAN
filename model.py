import torch
import torch.nn as nn


z_size = 100


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=z_size,
                out_channels=512,
                kernel_size=(4, 4),
                stride=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=512,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.ReLU(inplace=True)
        )
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.ReLU(inplace=True)
        )
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=3,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x) 

        return x
    


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=128,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=256,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False
                ),
            nn.BatchNorm2d(
                num_features=512,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
                ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=(4, 4),
                stride=(1, 1), bias=False
                ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x