# pytorch_implementations/autoencoders.py

import torch
from torch import nn

class VanillaAE(nn.Module):
    """
    A simple vanilla Autoencoder with fully-connected layers.
    This model is great for understanding the basic principles of encoding
    and decoding.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: Compresses the input into the latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )

        # Decoder: Reconstructs the input from the latent space representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Use Sigmoid to ensure output values are between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input image for the dense layers
        x = x.view(-1, self.input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ConvAE(nn.Module):
    """
    A Convolutional Autoencoder. This architecture is much better suited for
    image data as it preserves spatial hierarchies through convolutional layers.
    """
    def __init__(self):
        super().__init__()

        # Encoder: Uses Conv2d layers to extract features and downsample
        self.encoder = nn.Sequential(
            # Input: (B, 1, 28, 28)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (B, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)                       # -> (B, 64, 1, 1)
        )

        # Decoder: Uses ConvTranspose2d to upsample and reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),             # -> (B, 32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (B, 1, 28, 28)
            nn.Sigmoid() # To ensure pixel values are between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VAE(nn.Module):
    """
    A Variational Autoencoder. This is a generative model that learns a
    probability distribution for the latent space, allowing us to sample from it
    to generate new data.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 400)
        # Two output layers for mean (mu) and log-variance (log_var)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_log_var = nn.Linear(400, latent_dim)
        
        # Decoder layers
        self.fc_decode = nn.Linear(latent_dim, 400)
        self.fc_out = nn.Linear(400, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes the input into the parameters of a latent distribution.
        
        Returns:
            mu (torch.Tensor): The mean of the latent distribution.
            log_var (torch.Tensor): The log-variance of the latent distribution.
        """
        h1 = nn.functional.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        log_var = self.fc_log_var(h1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        The reparameterization trick. Instead of sampling from q(z|x), we
        sample from a standard normal distribution and scale by the learned
        standard deviation and shift by the learned mean. This allows the
        gradient to flow through the sampling process.
        
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * log_var)  # std = exp(log(std)) = exp(0.5 * log(std^2))
        epsilon = torch.randn_like(std) # sample from N(0, 1)
        return mu + std * epsilon

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent vector z back into a reconstruction.
        """
        h3 = nn.functional.relu(self.fc_decode(z))
        return torch.sigmoid(self.fc_out(h3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flatten the input
        x = x.view(-1, self.input_dim)
        
        # Encode to get latent distribution parameters
        mu, log_var = self.encode(x)
        
        # Sample from the latent distribution using the reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode the latent vector to get the reconstruction
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var