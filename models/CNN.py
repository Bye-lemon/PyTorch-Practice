import torch
import torch.nn as nn
import torch.nn.functional as F

WATCH = lambda x: print(x.shape)


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
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
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
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1000),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 7 * 7 * 512)
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
        return x


class _BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(_BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class _InceptionV1(nn.Module):
    def __init__(self, in_channels, out_ch1x1, out_ch3x3re, out_ch3x3, out_ch5x5re,
                 out_ch5x5, pool_proj):
        super(_InceptionV1, self).__init__()
        self.branch1x1 = _BasicConv2d(in_channels=in_channels, out_channels=out_ch1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            _BasicConv2d(in_channels=in_channels, out_channels=out_ch3x3re, kernel_size=1),
            _BasicConv2d(in_channels=out_ch3x3re, out_channels=out_ch3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            _BasicConv2d(in_channels=in_channels, out_channels=out_ch5x5re, kernel_size=1),
            _BasicConv2d(in_channels=out_ch5x5re, out_channels=out_ch5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            _BasicConv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch3x3(x)
        out3 = self.branch5x5(x)
        out4 = self.branch_pool(x)
        x = torch.cat([out1, out2, out3, out4], dim=1)
        return x


class _Inception_Auxiliary(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(_Inception_Auxiliary, self).__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            _BasicConv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=num_classes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 2048)
        x = self.classifier(x)
        return F.softmax(x, dim=0)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.step1 = nn.Sequential(
            _BasicConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            _BasicConv2d(in_channels=64, out_channels=64, kernel_size=1),
            _BasicConv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.inception3 = nn.Sequential(
            _InceptionV1(in_channels=192, out_ch1x1=64, out_ch3x3re=96, out_ch3x3=128,
                         out_ch5x5re=16, out_ch5x5=32, pool_proj=32),
            _InceptionV1(in_channels=256, out_ch1x1=128, out_ch3x3re=128, out_ch3x3=192,
                         out_ch5x5re=32, out_ch5x5=96, pool_proj=64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.inception4a = _InceptionV1(in_channels=480, out_ch1x1=192, out_ch3x3re=96, out_ch3x3=208,
                                        out_ch5x5re=16, out_ch5x5=48, pool_proj=64)
        self.aux1 = _Inception_Auxiliary(in_channels=480, num_classes=1000)
        self.inception4b = _InceptionV1(in_channels=512, out_ch1x1=160, out_ch3x3re=112, out_ch3x3=224,
                                        out_ch5x5re=24, out_ch5x5=64, pool_proj=64)
        self.inception4c = _InceptionV1(in_channels=512, out_ch1x1=128, out_ch3x3re=128, out_ch3x3=256,
                                        out_ch5x5re=24, out_ch5x5=64, pool_proj=64)
        self.inception4d = _InceptionV1(in_channels=512, out_ch1x1=112, out_ch3x3re=144, out_ch3x3=288,
                                        out_ch5x5re=32, out_ch5x5=64, pool_proj=64)
        self.aux2 = _Inception_Auxiliary(in_channels=528, num_classes=1000)
        self.inception4e = _InceptionV1(in_channels=528, out_ch1x1=256, out_ch3x3re=160, out_ch3x3=320,
                                        out_ch5x5re=32, out_ch5x5=128, pool_proj=128)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception5 = nn.Sequential(
            _InceptionV1(in_channels=832, out_ch1x1=256, out_ch3x3re=160, out_ch3x3=320,
                         out_ch5x5re=32, out_ch5x5=128, pool_proj=128),
            _InceptionV1(in_channels=832, out_ch1x1=384, out_ch3x3re=192, out_ch3x3=384,
                         out_ch5x5re=48, out_ch5x5=128, pool_proj=128),
            nn.AvgPool2d(kernel_size=7, ceil_mode=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=1000)
        )

    def forward(self, x):
        x = self.step1(x)
        x = self.inception3(x)
        aux1 = self.aux1(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.max_pool(x)
        x = self.inception5(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        x = F.softmax(x, dim=0)
        return aux1, aux2, x


class _ResUnit2L(nn.Module):
    def __init__(self, in_channels, out_channels, aux=False, **kwargs):
        super(_ResUnit2L, self).__init__()
        self.conv1 = _BasicConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                                  **kwargs)
        self.conv2 = _BasicConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.aux_conv = _BasicConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                     **kwargs) if aux else None

    def forward(self, x):
        oral = x
        x = self.conv2(self.conv1(x))
        if self.aux_conv is not None:
            oral = self.aux_conv(oral)
        x += oral
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        self.conv2 = nn.Sequential(
            _ResUnit2L(in_channels=64, out_channels=64),
            _ResUnit2L(in_channels=64, out_channels=64),
            _ResUnit2L(in_channels=64, out_channels=64)
        )
        self.conv3 = nn.Sequential(
            _ResUnit2L(in_channels=64, out_channels=128, aux=True, stride=2),
            _ResUnit2L(in_channels=128, out_channels=128),
            _ResUnit2L(in_channels=128, out_channels=128),
            _ResUnit2L(in_channels=128, out_channels=128)
        )
        self.conv4 = nn.Sequential(
            _ResUnit2L(in_channels=128, out_channels=256, aux=True, stride=2),
            _ResUnit2L(in_channels=256, out_channels=256),
            _ResUnit2L(in_channels=256, out_channels=256),
            _ResUnit2L(in_channels=256, out_channels=256),
            _ResUnit2L(in_channels=256, out_channels=256),
            _ResUnit2L(in_channels=256, out_channels=256)
        )
        self.conv5 = nn.Sequential(
            _ResUnit2L(in_channels=256, out_channels=512, aux=True, stride=2),
            _ResUnit2L(in_channels=512, out_channels=512),
            _ResUnit2L(in_channels=512, out_channels=512)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(in_features=512, out_features=1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return F.softmax(x, dim=0)


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, batch_size):
        super(_DenseLayer, self).__init__()
        self.add_module(f"bn1", nn.BatchNorm2d(num_features=in_channels))
        self.add_module(f"relu1", nn.ReLU(inplace=True))
        self.add_module("conv1",
                        nn.Conv2d(in_channels=in_channels, out_channels=growth_rate * batch_size, kernel_size=1))
        self.add_module(f"bn2", nn.BatchNorm2d(num_features=growth_rate * batch_size))
        self.add_module(f"relu2", nn.ReLU(inplace=True))
        self.add_module(f"conv2",
                        nn.Conv2d(in_channels=growth_rate * batch_size, out_channels=growth_rate, kernel_size=3,
                                  padding=1))

    def forward(self, x):
        out = super(_DenseLayer, self).forward(x)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, batch_size, num_layers):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module(f"Layer{i}",
                            _DenseLayer(in_channels=in_channels + growth_rate * i,
                                        growth_rate=growth_rate, batch_size=batch_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels):
        super(_Transition, self).__init__()
        self.add_module(f"bn", nn.BatchNorm2d(num_features=in_channels))
        self.add_module(f"relu", nn.ReLU(inplace=True))
        self.add_module(f"conv", nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1))
        self.add_module(f"avg_pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet121(nn.Module):
    def __init__(self, batch_size=4, growth_rate=12):
        super(DenseNet121, self).__init__()
        self.growth_rate = growth_rate
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense1 = _DenseBlock(in_channels=64, growth_rate=growth_rate, batch_size=batch_size, num_layers=6)
        self.trans1 = _Transition(in_channels=64 + 6 * growth_rate)
        self.dense2 = _DenseBlock(in_channels=64 + 6 * growth_rate, growth_rate=growth_rate, batch_size=batch_size,
                                  num_layers=12)
        self.trans2 = _Transition(in_channels=64 + 18 * growth_rate)
        self.dense3 = _DenseBlock(in_channels=64 + 18 * growth_rate, growth_rate=growth_rate, batch_size=batch_size,
                                  num_layers=24)
        self.trans3 = _Transition(in_channels=64 + 42 * growth_rate)
        self.dense4 = _DenseBlock(in_channels=64 + 42 * growth_rate, growth_rate=growth_rate, batch_size=batch_size,
                                  num_layers=16)
        self.dense = nn.Sequential(
            self.dense1, self.trans1, self.dense2, self.trans2, self.dense3, self.trans3, self.dense4
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(in_features=64+58*growth_rate, out_features=1000)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.dense(x)
        x = self.avg_pool(x)
        x = x.view(-1, 64+58*self.growth_rate)
        x = self.classifier(x)
        return F.softmax(x, dim=0)
