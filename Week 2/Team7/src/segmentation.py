from typing import Literal
import cv2
import numpy as np
from numpy.typing import NDArray


def remove_background(
    image: NDArray[np.uint8],
    *,
    border_fraction: float = 0.04,
    lightness_weight: float = 0.75,
    distance_threshold: float = 18.0,
    representative: Literal["mean", "median"] = "median",
    norm: Literal["l1", "l2"] = "l2",
    closing_kernel_size: int = 5,
    fill_holes: bool = True,
    keep_largest: bool = True,
) -> NDArray[np.uint8]:
    """Return the final 0/255 painting mask for a uint8 BGR image, without resizing.

    Compare standard Lab colors with the border reference, then close small gaps,
    fill enclosed holes and keep the largest component. Defaults reproduce the
    selected method: median, L2, L x 0.75, distance > 18, 4% border and 5x5 closing.

    border_fraction is a fraction of the shorter image side, rounded to pixels
    (at least one). closing_kernel_size is an odd pixel size; 0 disables closing.
    To obtain the color-only mask, disable closing, fill_holes and keep_largest.
    L1 uses a different distance scale and needs its own threshold.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image with three channels.")
    if min(image.shape[:2]) < 2:
        raise ValueError("Both image dimensions must be at least two pixels.")
    if not np.isfinite(border_fraction) or not 0 < border_fraction <= 0.5:
        raise ValueError("border_fraction must be in (0, 0.5].")
    if not np.isfinite(lightness_weight) or lightness_weight < 0:
        raise ValueError("lightness_weight must be finite and nonnegative.")
    if not np.isfinite(distance_threshold) or distance_threshold < 0:
        raise ValueError("distance_threshold must be finite and nonnegative.")
    if representative not in ("mean", "median") or norm not in ("l1", "l2"):
        raise ValueError("Expected mean/median and l1/l2.")

    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab[:, :, 0] *= lightness_weight
    width = min(max(1, round(border_fraction * min(image.shape[:2]))), min(image.shape[:2]) // 2)
    border = extract_border_pixels(lab, width)
    center = np.median(border, axis=0) if representative == "median" else border.mean(axis=0)
    delta = lab - center
    distance = np.linalg.norm(delta, axis=2) if norm == "l2" else np.abs(delta).sum(axis=2)
    mask = (distance > distance_threshold).astype(np.uint8) * 255
    mask = close_mask(mask, closing_kernel_size)
    if fill_holes:
        mask = fill_mask_holes(mask)
    if keep_largest:
        mask = keep_largest_component(mask)
    return mask


def extract_border_pixels(image: NDArray, width: int) -> NDArray:
    """Return the four border strips, counting corners only once."""
    if width <= 0 or 2 * width > min(image.shape[:2]):
        raise ValueError("Border width must be positive and at most half the short side.")
    border = np.zeros(image.shape[:2], dtype=bool)
    border[:width, :] = border[-width:, :] = True
    border[:, :width] = border[:, -width:] = True
    return image[border]


def close_mask(mask: NDArray[np.uint8], kernel_size: int = 5) -> NDArray[np.uint8]:
    """Seal small gaps with an odd square kernel; 0 disables closing."""
    if (not isinstance(kernel_size, (int, np.integer)) or kernel_size < 0
            or (kernel_size > 0 and kernel_size % 2 == 0)):
        raise ValueError("kernel_size must be zero or a positive odd integer.")
    if kernel_size <= 1:
        return mask.copy()
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))


def fill_mask_holes(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Fill background regions without an 8-connected path to an image border."""
    _, labels = cv2.connectedComponents((mask == 0).astype(np.uint8), connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]  # Label 0 is original foreground.
    return (~np.isin(labels, border_labels)).astype(np.uint8) * 255


def keep_largest_component(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Keep the largest 8-connected foreground region; assumes one painting."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count == 1:
        return mask.copy()
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255
