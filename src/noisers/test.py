"""
Just some code to test the noisers
"""
# pylint: skip-file

import matplotlib.pyplot as plt
import torch
import numpy as np

import src.utils as utils
from src.noisers.rosteals_noiser import RoSteALSNoiser
DATA_DIR = "./data/train2017_numpy_256.npy"
IMAGE_SIZE = 256

def main():
    image = utils.load_random_image(DATA_DIR, 256)
    image = torch.from_numpy(image)
    import ipdb; ipdb.set_trace()
    configs = {
        "p_differentiable": 0,
        "p_imagenet": 0,
        "p_identity": 1
    }
    noiser = RoSteALSNoiser(configs)
    image_prime = noiser.apply_noise(image)

    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(image_prime)
    axes[1].set_title("image_prime")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
