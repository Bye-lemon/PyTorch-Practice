from unittest import TestCase
from models.Attention import *

import torch


class TestCBAMConv2d(TestCase):
    def test_forward(self):
        try:
            x = torch.rand(4, 3, 224, 224)
            x = torch.autograd.Variable(x)
            self.att = CBAMConv2d(in_channels=3, out_channels=24, ratio=4, kernel_size=7, padding=3)
            y = self.att(x)
        except:
            self.fail()


class TestBAMConv2d(TestCase):
    def test_forward(self):
        try:
            x = torch.rand(4, 3, 224, 224)
            x = torch.autograd.Variable(x)
            self.att = BAMConv2d(in_channels=3, out_channels=24, ratio=4, kernel_size=7, padding=3)
            y = self.att(x)
        except:
            self.fail()


class TestAttentionAugmentedConv2d(TestCase):
    def test_forward(self):
        try:
            x = torch.rand(16, 3, 32, 32)
            self.att = AttentionAugmentedConv2d(in_channels=3, out_channels=20, dim_k=40, dim_v=4, num_h=4,
                                                kernel_size=7, padding=3, stride=2)
            y = self.att(x)
        except:
            self.fail()
