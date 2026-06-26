"""
This file defines the StegoPatchLegacy watermarker.

StegoPatchLegacy is identical to :class:`StegoPatch` except that it restores the
*legacy* :meth:`encode_batch`, the one used right before watermarking was moved
into latent space. In the legacy version the autoencoder sits *between* the patch
and stitch steps: each pixel patch is encoded independently, the delta is added,
each patch is decoded back to pixels independently, and only then are the pixel
patches stitched into the final image. (The current :class:`StegoPatch` instead
encodes the whole image once, patches/adds-delta/stitches in latent space, and
decodes once.)

Use this class to load and run checkpoints that were trained against the legacy
per-patch pixel-space encoding.
"""
import torch

from src.watermarkers.stegopatch import StegoPatch


class StegoPatchLegacy(StegoPatch):
    """StegoPatch with the legacy per-patch (pixel-space) ``encode_batch``."""

    def encode_batch(self, covers: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        Args:
            covers: a (B, C, H, W) tensor of images to watermark.
            messages: a (B, message_length) tensor of messages to use.
        Returns:
            A (B, C, H, W) tensor of images that have been watermarked.
        """

        B, C, H, W = covers.shape

        assert H == W
        assert H % self.patch_size == 0

        # Number of patches along each spatial axis (images are square so these are equal).
        nh = H // self.patch_size
        nw = W // self.patch_size

        # First, "unravel" each cover into its patches and stack every image's patches
        # together, giving a tensor of shape (B * nh * nw, C, patch_size, patch_size).
        #
        # Split each spatial axis into (tile_index, within_tile_offset), move the tile
        # indices next to the batch dim, then flatten (B, nh, nw) into one leading axis.
        # The (nh, nw) ordering is row-major (patch (i, j) lands at flat index
        # i * nw + j within each image), which we rely on to stitch patches back together.
        patches = covers.reshape(B, C, nh, self.patch_size, nw, self.patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(B * nh * nw, C, self.patch_size, self.patch_size)

        # Now placed in the latent space.
        patches_latent = self.image_autoencoder.encode(patches)

        # Number of patches per image.
        num_patches = nh * nw

        deltas = self.message_encoder(messages)
        # Track how large the offset is getting during training: if bit accuracy is
        # stuck, this tells us whether the encoder is actually learning to embed.
        if self.log_tensorboard and self.message_encoder.training:
            delta_l2 = deltas.flatten(1).norm(dim=1).mean().item()
            self.tensorboard.add_scalar("delta_l2", delta_l2, self.step)

        # Repeat each image's delta across all of its patches so the deltas line up with
        # the patch ordering above. Because image b's patches occupy the contiguous block
        # [b*num_patches, (b+1)*num_patches), we repeat *consecutively* (repeat_interleave),
        # giving shape (B * num_patches, *latent_dims) where row b*num_patches + k is image
        # b's delta for every patch k.
        deltas = deltas.repeat_interleave(num_patches, dim=0)

        assert deltas.shape == patches_latent.shape

        watermarked_patches_latent = patches_latent + deltas
        watermarked_patches = self.image_autoencoder.decode(watermarked_patches_latent)

        # Stitch the watermarked patches back into full images. This inverts the unravel
        # above: split the flat patch axis back into (B, nh, nw), move the spatial tile
        # indices back next to their within-patch offsets, then merge each (nh, patch_size)
        # and (nw, patch_size) pair into the full H and W.
        watermarked = watermarked_patches.reshape(B, nh, nw, C, self.patch_size, self.patch_size)
        watermarked = watermarked.permute(0, 3, 1, 4, 2, 5)
        watermarked = watermarked.reshape(B, C, H, W)

        return watermarked
