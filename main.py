# pylint: skip-file
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from PIL import Image

from src.autoencoders.vqgan import VQGAN
import src.utils as utils
from src.watermarkers.rosteals import RoSteALS

DEVICE = "mps"

DATA_DIR = Path("data/train2017")
# vq-f4 was trained on 256x256 crops, so we work at that resolution.
IMAGE_SIZE = 256
MESSAGE_LENGTH = 50
BATCH_SIZE = 4

C_IMAGE = 3
H_IMAGE = IMAGE_SIZE
W_IMAGE = IMAGE_SIZE
H_LITTLE = IMAGE_SIZE / 8
W_LITTLE = IMAGE_SIZE / 8
C_LITTLE = 3
ALPHA = 1.5
BETA_MIN = 0.1
BETA_MAX = 10
BETA_DELTA = 1
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
TRAINING_SUBSET_SIZE = 8


def main():
    configs = {
        "device": "mps",
        "autoencoder_type": "VQGAN",
        "message_length": MESSAGE_LENGTH,
        "c_image": C_IMAGE,
        "h_image": H_IMAGE,
        "w_image": W_IMAGE,
        "h_little": H_LITTLE,
        "w_little": W_LITTLE,
        "c_little": C_LITTLE,
        "alpha": ALPHA,
        "beta_min": BETA_MIN,
        "beta_max": BETA_MAX,
        "beta_delta": BETA_DELTA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_subset_size": TRAINING_SUBSET_SIZE
    }
    rosteals = RoSteALS(configs)

    # One image stuff
    # image = load_random_image(DATA_DIR, IMAGE_SIZE)
    # message = np.random.randint(0, 2, (MESSAGE_LENGTH, 1))
    # watermarked = rosteals.encode_image(image, message)
    # recovered = rosteals.decode_image(watermarked)
    # # Save the original and reconstruction side by side so the round-trip is visible.
    # side_by_side = np.concatenate([image, watermarked], axis=1)
    # out = Image.fromarray((side_by_side * 255).round().astype(np.uint8))
    # out.save("results/roundtrip.png")
    # print("Wrote roundtrip.png (original | reconstruction)")

    dataset = utils.NpyImageDataset(Path("data/train2017_numpy_256.npy"))
    dataset = Subset(dataset, range(15000))
    rosteals.train(dataset)

    # images = utils.load_random_images(DATA_DIR, IMAGE_SIZE, 1000)
    # rosteals.train(images)
    


if __name__ == "__main__":
    main()
