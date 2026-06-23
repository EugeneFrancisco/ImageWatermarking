"""
Plot watermarked images from different checkpoints of the model.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# pylint: disable=import-error  # Modules resolve when run from the repo root.

# Directory where the cover/stego image plots are saved.
PLOT_SAVE_DIR = Path("results/experiment_1/plots")

# The cover image to watermark and plot.
IMAGE_PATH = Path("data/menu.png")


def load_face_image(path: Path, size: int) -> np.ndarray:
    """
    Loads the face image as an (H, W, C) float array in [0, 1] at size x size.

    The image is first center-cropped to a square using the smaller side, then
    downsampled to size x size with a box filter, which averages the pixels in
    each block rather than cropping or subsampling them.
    """
    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.BOX)  # pylint: disable=no-member
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image_plot(image: np.ndarray, title: str, filename: str) -> None:
    """Saves a single (H, W, C) [0, 1] image as a titled plot in PLOT_SAVE_DIR."""
    PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(PLOT_SAVE_DIR / filename, bbox_inches="tight")
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
