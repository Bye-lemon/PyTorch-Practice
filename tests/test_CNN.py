from unittest import TestCase
from models.CNN import *

import torch


class TestAlexNet(TestCase):
    def test_forward(self):
        try:
            self.net = ResNet34()
            x = torch.rand(4, 3, 227, 227)
            x = torch.autograd.Variable(x)
            y = self.net(x)
        except:
            self.fail()


class TestVGG16(TestCase):
    def test_forward(self):
        try:
            self.net = VGG16()
            x = torch.rand(4, 3, 227, 227)
            x = torch.autograd.Variable(x)
            y = self.net(x)
        except:
            self.fail()


class TestVGG19(TestCase):
    def test_forward(self):
        try:
            self.net = VGG19()
            x = torch.rand(4, 3, 227, 227)
            x = torch.autograd.Variable(x)
            y = self.net(x)
        except:
            self.fail()


class TestGoogLeNet(TestCase):
    def test_forward(self):
        try:
            self.net = GoogLeNet()
            x = torch.rand(4, 3, 224, 224)
            x = torch.autograd.Variable(x)
            y = self.net(x)
        except:
            self.fail()


class TestResNet34(TestCase):
    def test_forward(self):
        try:
            self.net = ResNet34()
            x = torch.rand(4, 3, 224, 224)
            x = torch.autograd.Variable(x)
            y = self.net(x)
        except:
            self.fail()
