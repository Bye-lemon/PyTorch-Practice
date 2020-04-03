import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, encoder=None, decoder=None):
        super(AutoEncoder, self).__init__()
        if encoder is None or decoder is None:
            self.encoder, self.decoder = self._get_default_encoder_decoder()
        else:
            self.encoder, self.decoder = encoder, decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @staticmethod
    def _get_default_encoder_decoder():
        encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.Tanh()
        )
        decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2),
            nn.Sigmoid()
        )
        return encoder, decoder