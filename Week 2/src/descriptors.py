"""Global, block, pyramid and joint color histograms with optional foreground masks."""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


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
    are weighted (see the Week 1 methods in main.py).

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


# Spatial and joint histograms use the same color conversions and corrections.
# Masks restrict both preprocessing statistics and histogram counts.

# Chroma channels (without brightness) of each space for the 2D histograms.
CHROMA_2D: dict[str, tuple[int, int]] = {"HSV": (0, 1), "Lab": (1, 2), "YCbCr": (1, 2)}



def _as_bool_mask(mask: NDArray | None, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} differs from image shape {shape}")
    mask = mask > 0 if mask.dtype != np.bool_ else mask
    if not mask.any():
        raise ValueError("mask must contain foreground pixels")
    return mask


def masked_gray_world(img: NDArray[np.uint8], mask: NDArray[np.bool_] | None) -> NDArray[np.uint8]:
    """Week 1 gray-world with the channel means taken from the mask pixels only.

    Pixels outside the mask are left unchanged (they are never counted in a histogram).
    """
    if mask is None:
        return gray_world(img)
    out = img.copy()
    if mask.any():
        out[mask] = gray_world(img[mask].reshape(-1, 1, 3)).reshape(-1, 3)
    return out


def masked_stretch(channel: NDArray, value_range: tuple[float, float], mask: NDArray[np.bool_] | None) -> NDArray:
    """Week 1 contrast stretch with the 1st/99th percentiles taken from the mask pixels only."""
    if mask is None:
        return contrast_stretch(channel, value_range)
    out = channel.astype(np.float32)
    if mask.any():
        out[mask] = contrast_stretch(channel[mask], value_range)
    return out


def prepare_channels(img: NDArray[np.uint8], spaces: set[str], gray_world_on: bool, stretch_on: bool,
            mask: NDArray[np.bool_] | None) -> dict[str, list[NDArray]]:
    """Corrected channels of every needed colour space (gray-world once, then per-space stretch)."""
    base = masked_gray_world(img, mask) if gray_world_on else img
    channels = {}
    for space in spaces:
        conv = convert(base, space)
        if conv.ndim == 2:
            conv = conv[:, :, None]
        chans = []
        for c, value_range in enumerate(RANGES[space]):
            values = conv[:, :, c]
            if stretch_on and LUMINANCE_CHANNEL.get(space) == c:
                values = masked_stretch(values, value_range, mask)
            chans.append(np.ascontiguousarray(values))
        channels[space] = chans
    return channels



def _hist_1d(channels: dict, config: dict, mask_u8: NDArray[np.uint8] | None) -> NDArray[np.float32]:
    """Week 1 descriptor of one region: per-channel histograms, each sum 1, weighted, concatenated, sum 1."""
    parts = []
    for space in config["color_spaces"]:
        for c, (low, high) in enumerate(RANGES[space]):
            hist = cv2.calcHist([channels[space][c]], [0], mask_u8, [config["bins"]], [float(low), float(high)]).ravel()
            total = hist.sum()
            if total == 0:
                return np.zeros(sum(len(RANGES[s]) for s in config["color_spaces"]) * config["bins"], np.float32)
            weight = config.get("luminance_weight", 1.0) if LUMINANCE_CHANNEL.get(space) == c else 1.0
            parts.append(hist / total * weight)
    vector = np.concatenate(parts)
    return (vector / vector.sum()).astype(np.float32)


def _hist_nd(channels: dict, config: dict, mask_u8: NDArray[np.uint8] | None) -> NDArray[np.float32]:
    """Joint 2D (chroma) or 3D histogram of one region, sum 1 (all zeros if the region has no pixels)."""
    space = config["space"]
    idx = CHROMA_2D[space] if config["kind"] == "2d" else (0, 1, 2)
    stack = cv2.merge([channels[space][i].astype(np.float32) for i in idx])
    ranges = [float(v) for i in idx for v in RANGES[space][i]]
    hist = cv2.calcHist([stack], list(range(len(idx))), mask_u8, [config["bins"]] * len(idx), ranges).ravel()
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else np.zeros_like(hist, np.float32)


def _region_descriptor(channels: dict, config: dict, mask: NDArray[np.bool_] | None,
                       box: tuple[int, int, int, int] | None) -> NDArray[np.float32]:
    """Descriptor of the whole image (box None) or of one block; uses only mask pixels inside it."""
    if box is not None:
        y0, y1, x0, x1 = box
        channels = {s: [np.ascontiguousarray(ch[y0:y1, x0:x1]) for ch in chans] for s, chans in channels.items()}
        mask = None if mask is None else mask[y0:y1, x0:x1]
    mask_u8 = None if mask is None else mask.astype(np.uint8)
    return _hist_1d(channels, config, mask_u8) if config["kind"] == "1d" else _hist_nd(channels, config, mask_u8)


def _grid_boxes(height: int, width: int, n: int) -> list[tuple[int, int, int, int]]:
    return [(r * height // n, (r + 1) * height // n, c * width // n, (c + 1) * width // n)
            for r in range(n) for c in range(n)]


def pyramid_level_weights(levels: int, scheme: str) -> list[float]:
    """Weights of levels 0..levels-1 (grids 1, 2, 4, ...), normalised to sum 1.

    "lazebnik" (Lazebnik et al. 2006): 1/2^L for level 0 and 1/2^(L-l+1) for level l >= 1, with L = levels - 1.
    """
    if scheme == "equal":
        raw = [1.0] * levels
    elif scheme == "lazebnik":
        top = levels - 1
        raw = [1.0 / 2 ** top] + [1.0 / 2 ** (top - level + 1) for level in range(1, levels)]
    else:
        raise ValueError(f"unknown pyramid weights {scheme}")
    total = sum(raw)
    return [w / total for w in raw]


def _from_prepared(channels: dict, config: dict, mask: NDArray[np.bool_] | None, shape: tuple[int, int]) -> NDArray[np.float32]:
    height, width = shape
    if config.get("pyramid_levels"):
        weights = pyramid_level_weights(config["pyramid_levels"], config.get("pyramid_weights", "equal"))
        parts = []
        for level, weight in enumerate(weights):
            boxes = _grid_boxes(height, width, 2 ** level)
            parts += [_region_descriptor(channels, config, mask, b) * (weight / len(boxes)) for b in boxes]
    else:
        n = config.get("grid", 1)
        if n == 1:  # global descriptor: already sums to 1 (renormalising again would change the last bits)
            return _region_descriptor(channels, config, mask, None)
        parts = [_region_descriptor(channels, config, mask, b) for b in _grid_boxes(height, width, n)]
    vector = np.concatenate(parts)
    total = vector.sum()
    return (vector / total).astype(np.float32) if total > 0 else vector.astype(np.float32)


def _spaces(config: dict) -> set[str]:
    return set(config["color_spaces"]) if config["kind"] == "1d" else {config["space"]}



def crop_to_mask(img: NDArray[np.uint8], mask: NDArray[np.bool_] | None) -> tuple[NDArray[np.uint8], NDArray[np.bool_] | None]:
    """Image and mask cropped to the rectangle around the True pixels; unchanged without a mask or if it is empty."""
    if mask is None or not mask.any():
        return img, mask
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def compute_descriptor(img: NDArray[np.uint8], config: dict[str, Any], mask: NDArray | None = None) -> NDArray[np.float32]:
    """Compute a normalized 1D vector from a global, block, pyramid or joint histogram config.

    With a mask, the image and the mask are first cropped to the bounding box of the mask, so the grid / pyramid
    blocks are laid over the painting. Without a mask nothing is cropped.
    """
    img, bool_mask = crop_to_mask(img, _as_bool_mask(mask, img.shape[:2]))
    channels = prepare_channels(img, _spaces(config), config.get("gray_world", False), config.get("contrast_stretch", False),
                       bool_mask)
    return _from_prepared(channels, config, bool_mask, img.shape[:2])


def compute_descriptors_multi(img: NDArray[np.uint8], configs: dict[str, dict[str, Any]],
                              mask: NDArray | None = None) -> dict[str, NDArray[np.float32]]:
    """compute_descriptor for many configs, sharing the corrections (same result, faster for a grid search)."""
    img, bool_mask = crop_to_mask(img, _as_bool_mask(mask, img.shape[:2]))
    out, cache = {}, {}
    for name, config in configs.items():
        key = (config.get("gray_world", False), config.get("contrast_stretch", False))
        if key not in cache:
            cache[key] = {}
        missing = _spaces(config) - set(cache[key])
        if missing:
            cache[key].update(prepare_channels(img, missing, *key, bool_mask))
        out[name] = _from_prepared(cache[key], config, bool_mask, img.shape[:2])
    return out


def descriptor_length(config: dict[str, Any]) -> int:
    per_region = (sum(len(RANGES[s]) for s in config["color_spaces"]) * config["bins"] if config["kind"] == "1d"
                  else config["bins"] ** (2 if config["kind"] == "2d" else 3))
    if config.get("pyramid_levels"):
        return per_region * sum(4 ** level for level in range(config["pyramid_levels"]))
    return per_region * config.get("grid", 1) ** 2
