"""
AutoEncoder backed by the **vq-f4** VQGAN used by Bui et al. (RoSteALS).

vq-f4 is the frozen first-stage autoencoder from CompVis' latent-diffusion repo:
a downsampling factor of 4, 3 latent channels, and an 8192-entry codebook. We load
the pre-converted ``xvjiarui/ldm-vq-f4`` checkpoint through ``diffusers.VQModel``,
which provides the matching architecture so we only ever deal with the weights.

This is a torch-native interface: images are ``(B, C, H, W)`` tensors with float
values in ``[0, 1]`` and latents are ``(B, 3, H // 4, W // 4)`` tensors, both on
the model's device. ``encode``/``decode`` are differentiable so gradients flow
through the (frozen) autoencoder to the watermarker during training.
"""
from pathlib import Path

import torch
from diffusers import VQModel

from src.autoencoders.autoencoder import AutoEncoder


class VQGAN(AutoEncoder):
    """Frozen vq-f4 autoencoder (see Bui et al., https://arxiv.org/pdf/2304.03400)."""

    MODEL_ID = "xvjiarui/ldm-vq-f4"
    SHRINK_FACTOR = 4
    LATENT_CHANNELS = 3

    # Where the converted checkpoint is saved on first use so later runs load it
    # straight from disk instead of hitting the Hugging Face Hub.
    LOCAL_DIR = Path("models/ldm-vq-f4")

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or self._default_device())
        # Frozen in RoSteALS: eval mode + no parameter grads. Gradients still flow
        # *through* the autoencoder to the message encoder during training; the
        # autoencoder's own weights just never update.
        self.model = self._load_model().eval().requires_grad_(False)
        self.model = self.model.to(self.device)

    @classmethod
    def _load_model(cls) -> VQModel:
        """
        Loads the vq-f4 weights, preferring a one-time local copy.

        On the first run the checkpoint is fetched from the Hub and saved to
        ``LOCAL_DIR``; every later run loads from that directory and skips the
        Hub entirely.
        """
        if cls.LOCAL_DIR.exists():
            return VQModel.from_pretrained(cls.LOCAL_DIR)

        model = VQModel.from_pretrained(cls.MODEL_ID)
        cls.LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(cls.LOCAL_DIR)
        return model

    @staticmethod
    def _default_device() -> str:
        """Picks the best available torch device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns the (continuous, pre-quantization) latents of a batch of images.

        Quantization is deferred to :meth:`decode` so downstream watermarking can
        operate on the continuous latent space, following Bui et al. Expects a
        ``(B, C, H, W)`` tensor in ``[0, 1]`` on the model's device.
        """
        images = images * 2.0 - 1.0  # the model works in [-1, 1]
        return self.model.encode(images).latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Returns the images reconstructed from latents.
        """
        images = self.model.decode(latents).sample
        return images / 2.0 + 0.5  # back to [0, 1]

    @classmethod
    def get_latent_dim(cls, h_image: int, w_image: int) -> tuple:
        return (
            cls.LATENT_CHANNELS,
            h_image / cls.SHRINK_FACTOR,
            w_image / cls.SHRINK_FACTOR
        )
