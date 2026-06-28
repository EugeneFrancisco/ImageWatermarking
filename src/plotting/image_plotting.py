"""
Plot watermarked images from different checkpoints of the model.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# pylint: disable=import-error  # Modules resolve when run from the repo root.

def load_image(path: Path, height: int, width: int) -> np.ndarray:
    """
    Loads the image in path as an (H, W, C) float array in [0, 1] at height x width.

    The image is first center-cropped to the aspect ratio of width:height (so no
    content is stretched), then resampled to height x width with a box filter,
    which averages the pixels in each block rather than cropping or subsampling
    them.
    """
    image = Image.open(path).convert("RGB")
    # Crop to the target aspect ratio about the center using the larger possible box.
    target_ratio = width / height
    if image.width / image.height > target_ratio:
        # Image is too wide: crop the width.
        crop_width = round(image.height * target_ratio)
        crop_height = image.height
    else:
        # Image is too tall: crop the height.
        crop_width = image.width
        crop_height = round(image.width / target_ratio)
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height), Image.BOX)  # pylint: disable=no-member
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image_plot(image: np.ndarray, title: str, save_path: Path) -> None:
    """Saves a single (H, W, C) [0, 1] image as a titled plot to save_path (a .png file)."""
    assert str(save_path).endswith(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def plot_side_by_side(image_1: np.ndarray, image_2: np.ndarray, save_path: Path) -> None:
    """
    Plots a single png of image_1 on the left and image_2 on the right side by side and saves the
    image to the save_path folder.
    """
    assert image_1.shape == image_2.shape
    save_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2)
    for ax, image in zip(axes, (image_1, image_2)):
        ax.imshow(np.clip(image, 0.0, 1.0))
        ax.axis("off")
    fig.savefig(save_path / "side_by_side.png", bbox_inches="tight")
    plt.close(fig)
