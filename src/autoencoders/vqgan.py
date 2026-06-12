"""
AutoEncoder backed by the **vq-f4** VQGAN used by Bui et al. (RoSteALS).

vq-f4 is the frozen first-stage autoencoder from CompVis' latent-diffusion repo:
a downsampling factor of 4, 3 latent channels, and an 8192-entry codebook. We load
the pre-converted ``xvjiarui/ldm-vq-f4`` checkpoint through ``diffusers.VQModel``,
which provides the matching architecture so we only ever deal with the weights.

Images are ``np.ndarray`` of shape ``(H, W, C)`` with float values in ``[0, 1]``.
Latents are ``np.ndarray`` of shape ``(3, H // 4, W // 4)``.
"""
import numpy as np
import torch
from diffusers import VQModel

from src.autoencoders.autoencoder import AutoEncoder


class VQGAN(AutoEncoder):
    """Frozen vq-f4 autoencoder (see Bui et al., https://arxiv.org/pdf/2304.03400)."""

    MODEL_ID = "xvjiarui/ldm-vq-f4"
    SHRINK_FACTOR = 4
    LATENT_CHANNELS = 3

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or self._default_device())
        self.model = VQModel.from_pretrained(self.MODEL_ID).eval().to(self.device)

    @staticmethod
    def _default_device() -> str:
        """Picks the best available torch device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Returns the (continuous, pre-quantization) latent of the image.

        Quantization is deferred to :meth:`decode` so downstream watermarking can
        operate on the continuous latent space, following Bui et al.
        """
        tensor = self._image_to_tensor(image)
        with torch.no_grad():
            latents = self.model.encode(tensor).latents
        return latents.squeeze(0).cpu().numpy()

    def decode(self, latent_variable: np.ndarray) -> np.ndarray:
        """
        Returns the image reconstructed from a latent produced by :meth:`encode`.

        The latent is vector-quantized to the nearest codebook entries before being
        decoded back into pixel space.
        """
        tensor = torch.from_numpy(latent_variable).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            image = self.model.decode(tensor).sample
        return self._tensor_to_image(image)

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Converts an ``(H, W, C)`` image in ``[0, 1]`` to a ``(1, C, H, W)`` tensor in ``[-1, 1]``."""
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        return (tensor * 2.0 - 1.0).to(self.device)

    @staticmethod
    def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """Converts a ``(1, C, H, W)`` tensor in ``[-1, 1]`` back to an ``(H, W, C)`` image in ``[0, 1]``."""
        tensor = (tensor.squeeze(0) / 2.0 + 0.5).clamp(0.0, 1.0)
        return tensor.permute(1, 2, 0).cpu().numpy()
