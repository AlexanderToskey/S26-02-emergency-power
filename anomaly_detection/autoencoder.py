import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def detect(self, X_np, _ae_threshold):
        """
        X_np: numpy array of normalized features (n_samples x n_features)
        Returns:
            errors: np.array of reconstruction MSE per row
            anomaly_mask: boolean np.array where True = anomaly
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
            recon = self.forward(X_tensor).numpy()
            errors = np.mean((X_np - recon) ** 2, axis=1)
            anomaly_mask = errors > _ae_threshold
        return errors, anomaly_mask


class AutoencoderWrapper:
    """
    Lightweight wrapper for inference-time anomaly detection.
    """

    def __init__(self, model: Autoencoder, threshold: float):
        self.model = model
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Returns per-row reconstruction error (MSE).
        """
        x_tensor = torch.tensor(X, dtype=torch.float32)
        recon = self.model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1)
        return error.numpy()

    def detect(self, X: np.ndarray):
        """
        Returns:
            errors: np.ndarray
            anomaly_mask: np.ndarray (bool)
        """
        errors = self.reconstruction_error(X)
        mask = errors > self.threshold
        return errors, mask