"""
Utils file for odds and ends
"""
import torch

def rgb_to_yuv(images: torch.Tensor) -> torch.Tensor:
    """
    Converts the passed in images from RGB into YUV space.
    Args:
        images: a (B, C, H, W) tensor of images where the C dimension is in RGB.
    Returns:
        A (B, C, H, W) tensor of images where the C dimension is in YUV.
    """
    # BT.601 RGB -> YUV conversion matrix (rows map RGB to Y, U, V).
    weight = torch.tensor(
        [
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001],
        ],
        dtype=images.dtype,
        device=images.device,
    )
    # einsum keeps the op differentiable: gradients flow back through `images`.
    return torch.einsum("oc,bchw->bohw", weight, images)
