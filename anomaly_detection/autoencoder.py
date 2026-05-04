"""
autoencoder.py - PyTorch autoencoder for weather anomaly detection.

Learns to reconstruct normal county-day weather feature vectors during
training. At inference, counties whose weather is unusual produce high
reconstruction error and are flagged as anomalous (Tier 2 detection).

Architecture: input → 64 → 32 → latent_dim (encoder)
              latent_dim → 32 → 64 → input (decoder)
"""

import torch
import torch.nn as nn
import numpy as np


class Autoencoder(nn.Module):
    """
    Symmetric encoder-decoder network for reconstruction-based anomaly detection.

    Trained only on normal (non-outage) weather samples so that unusual
    conditions produce a high mean-squared reconstruction error.
    """

    def __init__(self, input_dim: int, latent_dim: int = 16):
        """
        Args:
            input_dim: Number of input weather features.
            latent_dim: Size of the compressed bottleneck representation.
        """
        super().__init__()

        # ── Encoder: compress input to latent representation ──────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # ── Decoder: reconstruct input from latent representation ─────────────
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

    def detect(self, X_np: np.ndarray, _ae_threshold: float):
        """
        Flag counties whose weather deviates from the learned normal distribution.

        Args:
            X_np: Normalized feature array of shape (n_samples, n_features).
            _ae_threshold: Reconstruction MSE above which a sample is anomalous.

        Returns:
            errors: Per-row reconstruction MSE as a numpy array.
            anomaly_mask: Boolean array where True indicates an anomaly.
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

    Holds a trained Autoencoder and its threshold so callers only need
    to pass raw feature arrays — no threshold management required.
    """

    def __init__(self, model: Autoencoder, threshold: float):
        """
        Args:
            model: Trained Autoencoder instance.
            threshold: MSE cutoff above which a sample is flagged as anomalous.
        """
        self.model = model
        self.model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-row reconstruction MSE for a batch of samples.

        Args:
            X: Normalized feature array of shape (n_samples, n_features).

        Returns:
            Per-row MSE as a numpy array of shape (n_samples,).
        """
        x_tensor = torch.tensor(X, dtype=torch.float32)
        recon = self.model(x_tensor)
        error = torch.mean((x_tensor - recon) ** 2, dim=1)
        return error.numpy()

    def detect(self, X: np.ndarray):
        """
        Run anomaly detection on a batch of normalized weather features.

        Args:
            X: Normalized feature array of shape (n_samples, n_features).

        Returns:
            errors: Per-row reconstruction MSE as a numpy array.
            anomaly_mask: Boolean array where True indicates an anomaly.
        """
        errors = self.reconstruction_error(X)
        mask = errors > self.threshold
        return errors, mask