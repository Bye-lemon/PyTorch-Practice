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


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.fc1 = nn.Linear(in_features=64, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=64, out_features=latent_dim)
        self.fc3 = nn.Linear(in_features=latent_dim, out_features=64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    @staticmethod
    def reparameterize(mu, logvar):
        epsilon = torch.autograd.Variable(torch.randn(mu.size(0), mu.size(1))).to(mu.device)
        return mu + torch.exp(logvar / 2) * epsilon

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        x = self.reparameterize(mu, logvar)
        x = self.fc3(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.decoder(x)
        return x, mu, logvar
