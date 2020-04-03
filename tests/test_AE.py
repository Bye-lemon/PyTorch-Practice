import torch
import torch.nn as nn

from unittest import TestCase

from models.AE import *


class TestAutoEncoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ae = AutoEncoder()
        cls.encoder = nn.Sequential(
            nn.Linear(3*32*32, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
            nn.Tanh()
        )
        cls.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 3*32*32),
            nn.Sigmoid()
        )
        cls.input = torch.randn(16, 3, 32, 32)

    def test__get_default_encoder_decoder(self):
        try:
            assert self.ae.encoder is not None
            assert self.ae.decoder is not None
        except AssertionError as e:
            print(e)
            self.fail()

    def test_forward(self):
        try:
            assert self.input.shape == self.ae(self.input).shape
            self.ae = AutoEncoder(self.encoder, self.decoder)
            input_ = self.input.view(self.input.shape[0], -1)
            assert input_.shape == self.ae(input_).shape
        except AssertionError as e:
            print(e)
            self.fail()
