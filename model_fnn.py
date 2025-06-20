import torch
import torch.nn as nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation="sigmoid"):
        super(VariationalAutoEncoder, self).__init__()
        self.activation = activation
        
        self.en_fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )                   
        
        self.lin_mu = nn.Linear(256, latent_dim)
        self.lin_logvar = nn.Linear(256, latent_dim)

        self.de_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
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
        b, c, h, w = x.shape
        x = x.view(b, h*w)
        mu, logvar = self.encoder(x)
        z = self.reparamterization(mu, logvar)
        return self.decoder(z, self.activation).view(b, c, h, w), mu, logvar    

if __name__ == "__main__":

    x = torch.randn(2, 1, 28, 28)
    model = VariationalAutoEncoder(28 * 28, 64, "sigmoid")

    # Test with flattened input
    x_flat = x.view(2, 28 * 28)
    print(model(x))
