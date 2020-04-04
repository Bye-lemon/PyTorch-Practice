import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.dis(x)
        return x


class SimpleGenerator(nn.Module):
    def __init__(self, in_features):
        super(SimpleGenerator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=3 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=128*4*4, out_features=1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.view(-1, 128*4*4)
        return self.fc(x)


class Generator(nn.Module):
    def __init__(self, in_features, num_features):
        super(Generator, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=num_features)
        self.bn = nn.BatchNorm2d(num_features=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 64, 64)
        return self.conv3(self.conv2(self.conv1(self.bn(x))))


