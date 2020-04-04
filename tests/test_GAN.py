from unittest import TestCase

from models.GAN import *

import torch


class TestSimpleDiscriminator(TestCase):
    def test_forward(self):
        try:
            dis = SimpleDiscriminator()
            input_ = torch.randn(16, 3*32*32)
            output = dis(input_)
            assert output.shape == (16, 1)
        except Exception as e:
            print(e)
            self.fail()


class TestSimpleGenerator(TestCase):
    def test_forward(self):
        try:
            gen = SimpleGenerator(in_features=10)
            input_ = torch.randn(16, 10)
            output = gen(input_)
            assert output.shape == (16, 3*32*32)
        except Exception as e:
            print(e)
            self.fail()


class TestDiscriminator(TestCase):
    def test_forward(self):
        try:
            dis = Discriminator()
            input_ = torch.randn(16, 3, 32, 32)
            output = dis(input_)
            assert output.shape == (16, 1)
        except Exception as e:
            print(e)
            self.fail()


class TestGenerator(TestCase):
    def test_forward(self):
        try:
            gen = Generator(in_features=10*10*10, num_features=64*64)
            input_ = torch.randn(16, 10*10*10)
            output = gen(input_)
            assert output.shape == (16, 3, 32, 32)
        except Exception as e:
            print(e)
            self.fail()
