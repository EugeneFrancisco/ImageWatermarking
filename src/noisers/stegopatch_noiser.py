"""
This file defines a noising class for use with the stegopatch watermarker. It is identical to the
RoSteALSNoiser except that it adds a cropping noise: when cropping is sampled, the whole batch is
cropped to a single random rectangular window whose height and width are sampled independently and
uniformly, each at least crop_size and at most the corresponding image dimension.
"""
from typing import Union, Callable
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.noisers.rosteals_noiser import RoSteALSNoiser


# Canonical noise-type names. These are exactly the keys of
# ``named_noise_functions`` and the names accepted by ``noise_function_at_severity``
# and ``evaluate_noise_robustness``; use these constants instead of string literals.

# Structural noise types defined by the noiser classes themselves.
NOISE_IDENTITY = "identity"
NOISE_DIFFERENTIABLE = "differentiable"
NOISE_IMAGENET = "imagenet"
NOISE_CROP = "crop"
NOISE_ROTATE = "rotate"

# Individual imagenet-c corruptions (the only noise types that carry a severity).
# Each name matches the corresponding function name in imagenet_corruptions.py.
NOISE_GAUSSIAN_NOISE = "gaussian_noise"
NOISE_SHOT_NOISE = "shot_noise"
NOISE_IMPULSE_NOISE = "impulse_noise"
NOISE_DEFOCUS_BLUR = "defocus_blur"
NOISE_FOG = "fog"
NOISE_BRIGHTNESS = "brightness"
NOISE_CONTRAST = "contrast"
NOISE_ELASTIC_TRANSFORM = "elastic_transform"
NOISE_PIXELATE = "pixelate"
NOISE_JPEG_COMPRESSION = "jpeg_compression"
NOISE_SPECKLE_NOISE = "speckle_noise"
NOISE_GAUSSIAN_BLUR = "gaussian_blur"
NOISE_SPATTER = "spatter"
NOISE_SATURATE = "saturate"


class StegoPatchNoiser(RoSteALSNoiser):
    CROP = 3
    ROTATE = 4
    _NAME_TO_INT = {
        **RoSteALSNoiser._NAME_TO_INT,
        NOISE_CROP: CROP,
        NOISE_ROTATE: ROTATE,
    }

    def __init__(self, configs: dict):
        super().__init__(configs)
        c = self.configs

        # Require every branch probability to be set explicitly (fail loudly otherwise).
        self.set_probabilities(
            p_identity=c["p_identity"],
            p_differentiable=c["p_differentiable"],
            p_imagenet=c["p_imagenet"],
            p_crop=c["p_crop"],
            p_rotate=c["p_rotate"],
        )

        # The (square) side length of the random crop window.
        self.crop_size = int(c["crop_size"])

        # Inclusive bounds (in degrees) for the uniformly sampled rotation angle.
        self.rotation_lower_bound = float(c["rotation_lower_bound"])
        self.rotation_upper_bound = float(c["rotation_upper_bound"])

    # -- probability control -------------------------------------------------
    def set_probabilities(
        self,
        p_identity: float,
        p_differentiable: float,
        p_imagenet: float,
        p_crop: float,
        p_rotate: float,
    ) -> None:
        """Overwrite the branch sampling probabilities. They must sum to 1."""
        assert abs(p_identity + p_differentiable + p_imagenet + p_crop + p_rotate - 1.0) < 1e-6
        self.p_identity = p_identity
        self.p_diff = p_differentiable
        self.p_imagenet = p_imagenet
        self.p_crop = p_crop
        self.p_rotate = p_rotate

    # -- type normalisation --------------------------------------------------
    def _normalize_type(self, noise_type: Union[str, int]) -> int:
        if noise_type == self.CROP or (
            isinstance(noise_type, str) and noise_type.lower() == NOISE_CROP
        ):
            return self.CROP
        if noise_type == self.ROTATE or (
            isinstance(noise_type, str) and noise_type.lower() == NOISE_ROTATE
        ):
            return self.ROTATE
        return super()._normalize_type(noise_type)

    # -- dispatch ------------------------------------------------------------
    def get_noise_function(
        self, noise_type: Union[str, int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        t = self._normalize_type(noise_type)
        if t == self.CROP:
            return self._crop_noise
        if t == self.ROTATE:
            return self._rotate_noise
        return super().get_noise_function(noise_type)

    def sample_noise_type(self) -> int:
        types = [self.DIFFERENTIABLE, self.IMAGENET, self.IDENTITY, self.CROP, self.ROTATE]
        probs = [self.p_diff, self.p_imagenet, self.p_identity, self.p_crop, self.p_rotate]
        return int(np.random.choice(types, p=probs))

    # -- crop ----------------------------------------------------------------
    def _crop_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Crop the whole (B, C, H, W) batch to a single random rectangular window. The crop
        height and width are sampled independently and uniformly, each at least crop_size and at
        most the corresponding image dimension.
        Slicing is differentiable, so gradients flow through to the kept region."""
        _, _, h, w = x.shape
        ch = int(np.random.randint(self.crop_size, h + 1))
        cw = int(np.random.randint(self.crop_size, w + 1))
        top = int(np.random.randint(0, h - ch + 1))
        left = int(np.random.randint(0, w - cw + 1))
        return x[:, :, top:top + ch, left:left + cw]

    # -- rotate --------------------------------------------------------------
    @staticmethod
    def _inscribed_hw(h: int, w: int, angle_deg: float) -> tuple:
        """Return the (height, width) of the largest axis-aligned rectangle that fits entirely
        inside an h x w image rotated by angle_deg degrees, i.e. a rectangle containing only real
        image pixels and no black corners. Standard largest-inscribed-rectangle solution."""
        if h <= 0 or w <= 0:
            return h, w
        angle = math.radians(angle_deg)
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a

        # Inset by 1px so the outermost ring (faintly darkened by bilinear sampling at the
        # rotated frame edge) is excluded, leaving only clean interior pixels.
        return max(1, int(math.floor(hr)) - 1), max(1, int(math.floor(wr)) - 1)

    def _rotate_noise(self, x: torch.Tensor, angle: float | None = None) -> torch.Tensor:
        """Rotate the whole (B, C, H, W) batch by a single angle (counter-clockwise), then
        centre-crop to the largest rectangle of real image pixels. This "zooms in" so no black
        corners remain; the spatial dimensions shrink as the rotation angle grows. Bilinear
        interpolation and the slice-based crop are both differentiable, so gradients flow back to
        the input pixels.

        When ``angle`` is None the angle is sampled uniformly from
        [rotation_lower_bound, rotation_upper_bound] degrees; otherwise the given fixed angle (in
        degrees) is used."""
        if angle is None:
            angle = float(np.random.uniform(self.rotation_lower_bound, self.rotation_upper_bound))
        rotated = TF.rotate(
            x,
            angle=angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=False,
        )
        _, _, h, w = x.shape
        hr, wr = self._inscribed_hw(h, w, angle)
        return TF.center_crop(rotated, [hr, wr])

    def rotate_function_at_angle(
        self, angle: float
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a (B, C, H, W) -> (B, C, H, W) function rotating by the fixed ``angle``
        (in degrees, counter-clockwise) rather than a randomly sampled one."""
        return lambda x: self._rotate_noise(x, angle=angle)

    # -- per-noise evaluation ------------------------------------------------
    def named_noise_functions(self) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Extend the parent's per-noise-type map with the StegoPatch-specific crop and rotate
        branches so bit accuracy can be measured against each independently. Each call still samples
        fresh parameters (a random crop window, or an angle from
        [rotation_lower_bound, rotation_upper_bound])."""
        funcs = super().named_noise_functions()
        funcs[NOISE_CROP] = self._crop_noise
        funcs[NOISE_ROTATE] = self._rotate_noise
        return funcs

    # -- numpy entry point ---------------------------------------------------
    def apply_noise_np(self, image: np.ndarray, noise_type: Union[str, int]) -> np.ndarray:
        t = self._normalize_type(noise_type)
        if t == self.CROP:
            h, w = image.shape[:2]
            ch = int(np.random.randint(self.crop_size, h + 1))
            cw = int(np.random.randint(self.crop_size, w + 1))
            top = int(np.random.randint(0, h - ch + 1))
            left = int(np.random.randint(0, w - cw + 1))
            return image[top:top + ch, left:left + cw].copy()
        if t == self.ROTATE:
            was_float = np.issubdtype(image.dtype, np.floating)
            x = image.astype(np.float32) if was_float else image.astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                x = self._rotate_noise(x)
            out = x.squeeze(0).permute(1, 2, 0).numpy()
            return out if was_float else (out * 255).astype(np.uint8)
        return super().apply_noise_np(image, noise_type)
