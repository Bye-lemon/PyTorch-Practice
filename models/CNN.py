import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.dropout1 = nn.Dropout()
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.dropout2 = nn.Dropout()
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=384)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=384)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.dropout3 = nn.Dropout()
        self.fc1 = nn.Linear(in_features=6 * 6 * 256, out_features=4096)
        self.bn6 = nn.BatchNorm1d(num_features=4096)
        self.dropout4 = nn.Dropout()
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.bn7 = nn.BatchNorm1d(num_features=4096)
        self.dropout5 = nn.Dropout()
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        x = F.max_pool2d(self.bn1(torch.relu(self.conv1(x))), kernel_size=3, stride=2)
        x = self.dropout1(x)
        x = F.max_pool2d(self.bn2(torch.relu(self.conv2(x))), kernel_size=3, stride=2)
        x = self.dropout2(x)
        x = self.bn3(torch.relu(self.conv3(x)))
        x = self.bn4(torch.relu(self.conv4(x)))
        x = self.bn5(torch.relu(self.conv5(x)))
        x = self.dropout3(F.max_pool2d(x, kernel_size=3, stride=2))
        x = x.view(-1, 6 * 6 * 256)
        x = self.dropout4(self.bn6(torch.relu(self.fc1(x))))
        x = self.dropout5(self.bn7(torch.relu(self.fc2(x))))
        x = F.softmax(self.fc3(x))
        return x


class _VGGBLOCK(nn.Sequential):
    def __init__(self, in_channels, out_channels, conv_num, block_name):
        super(_VGGBLOCK, self).__init__()
        self.add_module(f"{block_name}_conv0",
                        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1))
        self.add_module(f"{block_name}_relu0", nn.ReLU())
        for i in range(conv_num):
            self.add_module(f"{block_name}_conv{i + 1}",
                            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))
            self.add_module(f"{block_name}_relu{i + 1}", nn.ReLU())
        self.add_module(f"{block_name}_max_pool", nn.MaxPool2d(kernel_size=2, stride=2))


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            _VGGBLOCK(in_channels=3, out_channels=64, conv_num=2, block_name="block1"),
            _VGGBLOCK(in_channels=64, out_channels=128, conv_num=2, block_name="block2"),
            _VGGBLOCK(in_channels=128, out_channels=256, conv_num=3, block_name="block3"),
            _VGGBLOCK(in_channels=256, out_channels=512, conv_num=3, block_name="block4"),
            _VGGBLOCK(in_channels=512, out_channels=512, conv_num=3, block_name="block5")
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.classifier(x)
        x = F.softmax(x, dim=1000)
        return x


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            _VGGBLOCK(in_channels=3, out_channels=64, conv_num=2, block_name="block1"),
            _VGGBLOCK(in_channels=64, out_channels=128, conv_num=2, block_name="block2"),
            _VGGBLOCK(in_channels=128, out_channels=256, conv_num=4, block_name="block3"),
            _VGGBLOCK(in_channels=256, out_channels=512, conv_num=4, block_name="block4"),
            _VGGBLOCK(in_channels=512, out_channels=512, conv_num=4, block_name="block5")
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.classifier(x)
        x = F.softmax(x, dim=1000)
        return x
