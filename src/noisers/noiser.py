"""
A file that defines the noiser base class, which is a class that stores different image noising
functions.
"""
from abc import ABC, abstractmethod
from typing import Union, Callable
import numpy as np
import torch

class Noiser(ABC):
    """
    Stores a bunch of ways to noise an image and exports methods to apply noise to images.
    """
    def __init__(self, configs: dict):
        self.configs = configs
        assert self.configs["w_image"] == self.configs["h_image"]
        self.image_size: int = self.configs["w_image"]

    @abstractmethod
    def get_noise_function(
        self,
        noise_type: Union[str, int]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns the noise function associated with noise_type noise. If noise_type noise is not
        valid, this will raise an error.
        Args:
            noise_type: the type of noise to apply.
        Returns:
            A function which takes in tensors of shape (B, C, H, W) and returns tensors of the same
            shape. This function should be fully differentiable and should apply the transformation
            referred to by noise_type to each image in the batch.
        """

    @abstractmethod
    def sample_noise_type(self) -> int:
        """
        Samples a valid noise type to use.
        """

    def apply_noise(self, images: torch.Tensor) -> torch.Tensor:
        """
        Applies a randomly sampled noise transformation to the passed in images. All images will get
        the same noise sample applied to them.
        """
        noise_func = self.get_noise_function(self.sample_noise_type())
        return noise_func(images)

    @abstractmethod
    def apply_noise_np(self, image: np.ndarray, noise_type: Union[str, int]) -> np.ndarray:
        """
        Apply noise_type noise to the image. If noise_type is not a valid noise type, this should
        raise an error. Args:
            image: an image_size by image_size image on which we will apply noise_type.
            noise_type_str: a string or int describing the noise type which we will apply.
        """
