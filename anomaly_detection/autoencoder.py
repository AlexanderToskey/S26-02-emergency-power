#This file defines a simple autoencoder architecture for anomaly detection

import torch
import torch.nn as nn
import numpy as np

#Train the model on normal weather data and if it
#can't reconstruct a new sample well that tells us the sample is probably unusual
class Autoencoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        #This part shrinks the input down to a compressed representation
        #The model learns to capture the most important patterns in the data here
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        #---Decoder---
        #This part tries to reconstruct the original input from the compressed representation
        #If reconstruction is poor it suggests the input is unusual compared to the training data
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    #Pass data through the encoder and then the decoder to get the reconstruction
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def detect(self, X_np, _ae_threshold):
        #Runs anomaly detection on a batch of samples
        #Compares original vs reconstructed inputs
        with torch.no_grad():
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
            recon = self.forward(X_tensor).numpy()
            # Higher error means the model struggled to reconstruct this sample suggesting it's unusual
            errors = np.mean((X_np - recon) ** 2, axis=1)
            anomaly_mask = errors > _ae_threshold
        return errors, anomaly_mask


#A thin wrapper that bundles the model and its threshold together
#so callers don't have to pass the threshold around separately.
class AutoencoderWrapper:

    def __init__(self, model: Autoencoder, threshold: float):
        self.model = model
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(X, dtype=torch.float32)
        recon = self.model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1)
        return error.numpy()

    def detect(self, X: np.ndarray):
        errors = self.reconstruction_error(X)
        mask = errors > self.threshold
        return errors, mask
