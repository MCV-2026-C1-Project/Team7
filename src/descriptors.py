"""Compute global color histogram descriptors indexed by image ID."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.data_loading import get_image_id, load_image

# OpenCV conversion from BGR to each colour space it supports directly.
CONVERSIONS: dict[str, int] = {
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "HLS": cv2.COLOR_BGR2HLS,
    "Lab": cv2.COLOR_BGR2Lab,
    "Luv": cv2.COLOR_BGR2Luv,
    "YCbCr": cv2.COLOR_BGR2YCrCb,  # OpenCV gives Y, Cr, Cb; reordered in `convert` to Y, Cb, Cr.
    "Gray": cv2.COLOR_BGR2GRAY,  # One channel only.
}

# Value range of each channel. In OpenCV, Hue goes from 0 to 180 and the other 8-bit channels
# from 0 to 255. The two hand-made spaces (rgb, Opponent) are float: their upper limits get a
# small margin because calcHist excludes the upper limit and float32 rounding can land on it.
EPS: float = 1e-3
RANGES: dict[str, list[tuple[float, float]]] = {
    "RGB": [(0, 256)] * 3,
    "HSV": [(0, 180), (0, 256), (0, 256)],
    "HLS": [(0, 180), (0, 256), (0, 256)],
    "Lab": [(0, 256)] * 3,
    "Luv": [(0, 256)] * 3,
    "YCbCr": [(0, 256)] * 3,
    "rgb": [(0, 1 + EPS)] * 3,
    "Opponent": [
        (-255 / np.sqrt(2), 255 / np.sqrt(2) + EPS),
        (-510 / np.sqrt(6), 510 / np.sqrt(6) + EPS),
        (0, 765 / np.sqrt(3) + EPS),
    ],
    "Gray": [(0, 256)],
}

COLOR_SPACES: tuple[str, ...] = tuple(RANGES)


def convert(image: NDArray[np.uint8], color_space: str) -> NDArray:
    """Convert a BGR image to another colour space.

    Args:
        image: Image in BGR colour order, as returned by OpenCV.
        color_space: One of `COLOR_SPACES`.

    Returns:
        The converted image: uint8 for the OpenCV spaces, float32 for `rgb` and `Opponent`.
        Two-dimensional for `Gray`, three-dimensional otherwise.

    Raises:
        ValueError: If the colour space is unknown.
    """
    if color_space in CONVERSIONS:
        converted = cv2.cvtColor(image, CONVERSIONS[color_space])
        if color_space == "YCbCr":
            converted = converted[:, :, [0, 2, 1]]  # Y, Cr, Cb -> Y, Cb, Cr.
        if color_space in ("HSV", "HLS"):
            hue = converted[:, :, 0]
            hue[hue == 180] = 0  # OpenCV can round hue up to 180; hue is circular, so 180 = 0.
        return np.ascontiguousarray(converted)

    blue, green, red = [channel.astype(np.float32) for channel in cv2.split(image)]
    if color_space == "rgb":
        # Normalised rgb: each channel divided by R + G + B, so only the colour remains.
        total = red + green + blue
        total[total == 0] = 1  # Black pixels: avoid dividing by zero.
        return np.dstack([red / total, green / total, blue / total])
    if color_space == "Opponent":
        # Two colour channels (red-green, yellow-blue) and one intensity channel.
        return np.dstack([
            (red - green) / np.sqrt(2),
            (red + green - 2 * blue) / np.sqrt(6),
            (red + green + blue) / np.sqrt(3),
        ]).astype(np.float32)
    raise ValueError(f"Unknown colour space: {color_space}")


# Index of the brightness channel in each colour space (normalised rgb and RGB have none).
LUMINANCE_CHANNEL: dict[str, int] = {"HSV": 2, "HLS": 1, "Lab": 0, "Luv": 0, "YCbCr": 0, "Opponent": 2, "Gray": 0}


def gray_world(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Scale the three BGR channels to the same mean, removing the colour cast of the lights."""
    pixels = image.astype(np.float32)
    means = pixels.reshape(-1, 3).mean(axis=0)
    pixels *= means.mean() / np.maximum(means, 1e-6)
    return np.clip(pixels, 0, 255).astype(np.uint8)


def contrast_stretch(channel: NDArray, value_range: tuple[float, float]) -> NDArray:
    """Stretch a channel linearly so its 1st and 99th percentiles fill the whole value range.

    Values outside the percentiles are clipped. A flat channel is returned unchanged.
    """
    low, high = np.percentile(channel, (1, 99))
    if high <= low:
        return channel
    bottom, top = value_range
    top = 255.0 if top == 256 else top - EPS / 2  # stay inside the last histogram bin
    stretched = bottom + (channel.astype(np.float32) - low) / (high - low) * (top - bottom)
    return np.clip(stretched, bottom, top).astype(np.float32)


def channel_histograms(
    image: NDArray[np.uint8],
    color_space: str,
    bins: int = 32,
    *,
    stretch: bool = False,
    luminance_weight: float = 1.0,
) -> list[NDArray[np.float32]]:
    """Compute one 1D histogram per channel of an image in one colour space.

    Args:
        image: Image in BGR colour order.
        color_space: One of `COLOR_SPACES`.
        bins: Number of bins of each histogram.
        stretch: Contrast-stretch the brightness channel (`LUMINANCE_CHANNEL`) before its histogram.
        luminance_weight: Weight of the brightness histogram (the other channels keep weight 1).

    Returns:
        One histogram per channel (3, or 1 for `Gray`), each normalised to sum 1 and then
        multiplied by its weight.

    Raises:
        ValueError: If `bins` is not positive or a channel histogram has zero total.
    """
    if bins <= 0:
        raise ValueError("bins must be greater than 0")

    converted = convert(image, color_space)
    histograms = []
    for channel, (low, high) in enumerate(RANGES[color_space]):
        is_luminance = LUMINANCE_CHANNEL.get(color_space) == channel
        weight = luminance_weight if is_luminance else 1.0
        if stretch and is_luminance:
            values = converted if converted.ndim == 2 else converted[:, :, channel]
            values = np.ascontiguousarray(contrast_stretch(values, (low, high)))
            histogram = cv2.calcHist([values], [0], None, [bins], [float(low), float(high)]).ravel()
        else:
            histogram = cv2.calcHist(
                [converted], [channel], None, [bins], [float(low), float(high)]
            ).ravel()
        total = histogram.sum()
        if total == 0:
            raise ValueError(f"Histogram total is zero for {color_space} channel {channel}.")
        histograms.append(histogram / total * weight)
    return histograms


def describe(
    image: NDArray[np.uint8],
    color_spaces: list[str] | tuple[str, ...] | str = ("HSV",),
    bins: int = 32,
    *,
    gray_world_on: bool = False,
    stretch: bool = False,
    luminance_weight: float = 1.0,
) -> NDArray[np.float32]:
    """Compute the descriptor of an image: its 1D histograms, concatenated.

    Every channel of every requested colour space gets its own histogram; all of them are
    concatenated into one vector, normalised to sum 1. Optionally the image is gray-world corrected
    first, the brightness channel of each space is contrast-stretched, and the brightness histograms
    are weighted (see the submitted methods in main.py).

    Examples:
        describe(image, ["HSV"])         -> 3 histograms -> 3 * bins values
        describe(image, ["HSV", "Lab"])  -> 6 histograms -> 6 * bins values
        describe(image, ["Gray"])        -> 1 histogram  -> bins values

    Args:
        image: Image in BGR colour order, as returned by OpenCV.
        color_spaces: Colour space name, or list of names, from `COLOR_SPACES`.
        bins: Number of bins of each histogram.
        gray_world_on: Apply gray-world colour correction before the conversion.
        stretch: Contrast-stretch the brightness channel of each colour space.
        luminance_weight: Weight of the brightness histograms.

    Returns:
        1D float32 vector that sums to 1.

    Raises:
        ValueError: If no colour spaces are selected, a space is unknown,
            bins is not positive, or a channel histogram has zero total.
    """
    if isinstance(color_spaces, str):
        color_spaces = [color_spaces]
    if not color_spaces:
        raise ValueError("At least one colour space must be selected.")
    if gray_world_on:
        image = gray_world(image)
    histograms = []
    for color_space in color_spaces:
        histograms += channel_histograms(image, color_space, bins, stretch=stretch,
                                         luminance_weight=luminance_weight)
    vector = np.concatenate(histograms)
    return (vector / vector.sum()).astype(np.float32)


def compute_descriptors(
    paths: list[Path],
    color_spaces: list[str],
    bins: int,
    **options: Any,
) -> dict[int, NDArray[np.float32]]:
    """Load images and compute descriptors indexed by image ID.

    Args:
        paths: Image paths.
        color_spaces: Colour spaces used to compute each descriptor.
        bins: Number of histogram bins per channel.
        **options: Optional corrections passed to `describe` (gray_world_on, stretch, luminance_weight).

    Returns:
        Dictionary mapping each image ID to its descriptor.

    Raises:
        ValueError: If no paths are provided or an image ID is duplicated.
    """
    if not paths:
        raise ValueError("No image paths were provided.")

    descriptors: dict[int, NDArray[np.float32]] = {}

    for path in paths:
        image_id = get_image_id(path)

        if image_id in descriptors:
            raise ValueError(
                f"Duplicate image ID {image_id} found in path '{path}'."
            )

        descriptors[image_id] = describe(load_image(path), color_spaces, bins, **options)

    return descriptors
