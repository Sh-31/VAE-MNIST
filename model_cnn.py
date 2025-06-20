import torch
import torch.nn as nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation="sigmoid"):
        super(VariationalAutoEncoder, self).__init__()
        self.activation = activation
        
        self.en_fc = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same"),  # b, 32, 28, 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # b, 64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), # b, 64, 7, 7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )                   
        
        self.lin_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.lin_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.de_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, output_padding=1, stride=2), # b, 64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, output_padding=1, stride=2), # b, 32, 28, 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1) # b, 1, 28, 28
        )

    def encoder(self, x):
        x1 = self.en_fc(x)
        mu = self.lin_mu(x1)
        logvar = self.lin_logvar(x1)
        return mu, logvar   
    
    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)
        return mu + std * epsilon
        
    def decoder(self, z, activation):
        if activation == "sigmoid":
            return F.sigmoid(self.de_fc(z))
        return self.de_fc(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparamterization(mu, logvar)
        return self.decoder(z, self.activation), mu, logvar    

if __name__ == "__main__":

    x = torch.randn(2, 1, 28, 28)
    model = VariationalAutoEncoder(input_dim=None, latent_dim=256)

    print(model(x))
