"""Descriptors: compute_descriptor(img, config, mask) for colour histograms (src/colour.py), texture
(src/texture.py) and fused colour + texture, with an optional painting mask and noise filtering (src/denoise.py)."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.colour import colour_descriptor
from src.denoise import denoise, noise_type, remove_noise
from src.texture import describe_texture


def _as_bool_mask(mask: NDArray | None, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
    """Boolean mask of the image size (accepts 0/255 masks); None stays None."""
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} differs from image shape {shape}")
    mask = mask > 0 if mask.dtype != np.bool_ else mask
    if not mask.any():
        raise ValueError("mask must contain foreground pixels")
    return mask


def crop_to_mask(img: NDArray[np.uint8], mask: NDArray[np.bool_] | None) -> tuple[NDArray[np.uint8], NDArray[np.bool_] | None]:
    """Image and mask cropped to the rectangle around the True pixels; unchanged without a mask or if it is empty."""
    if mask is None or not mask.any():
        return img, mask
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _needs_denoising(config: dict[str, Any]) -> bool:
    """Whether a config (or one of its parts) asks for noise filtering."""
    return bool(config.get("denoise_method")) or any(_needs_denoising(p) for p in config.get("parts", []))


def prepare_image(img: NDArray[np.uint8], config: dict[str, Any], mask: NDArray | None = None, *,
                  noisy: bool | str | None = None) -> tuple[NDArray[np.uint8], NDArray[np.bool_] | None, bool | str]:
    """The image every descriptor starts from: cropped to the mask's bounding box, then filtered as the config asks.

    noisy: None = detect, False = clean, True = impulse noise, or a noise type from src.denoise.noise_type.
    Returns the image, its boolean mask and the noise found, which the parts of a combined descriptor reuse
    (the detector runs once per image).
    """
    img, bool_mask = crop_to_mask(img, _as_bool_mask(mask, img.shape[:2]))
    if noisy is None and _needs_denoising(config):
        noisy = noise_type(img) or False
    noise = "impulse" if noisy is True else (noisy or None)
    method = config.get("denoise_method")
    if method == "auto":  # the filter that suits the detected noise; clean images untouched
        img = denoise(img, noise)
    elif method and noise == "impulse":  # fixed filter (Week 3 branch presets): impulse-noisy images only
        img = remove_noise(img, method)
    return img, bool_mask, noisy


def compute_descriptor(img: NDArray[np.uint8], config: dict[str, Any], mask: NDArray | None = None, *,
                       noisy: bool | str | None = None) -> NDArray[np.float32]:
    """Descriptor of an image for a colour ("1d", "2d", "3d"), "texture" or "combined" config (src/config.py).

    With a mask, the image and the mask are first cropped to the bounding box of the mask, so the grid / pyramid
    blocks are laid over the painting. Without a mask nothing is cropped.
    """
    img, bool_mask, noisy = prepare_image(img, config, mask, noisy=noisy)
    if config["kind"] == "combined":
        # Weighted concatenation of parts, each divided by its scale (the mean L1 distance between two
        # database descriptors), so the L1 distance of the result is the weighted sum of the parts' L1
        # distances in comparable units. Each part denoises with the filter that suits it.
        parts = [compute_descriptor(img, part, bool_mask, noisy=noisy) for part in config["parts"]]
        return np.concatenate([weight * part / scale for part, weight, scale
                               in zip(parts, config["weights"], config["scales"])]).astype(np.float32)
    if config["kind"] == "texture":
        return describe_texture(img, config["texture"], mask=bool_mask, **config.get("params", {}))
    return colour_descriptor(img, config, bool_mask)
