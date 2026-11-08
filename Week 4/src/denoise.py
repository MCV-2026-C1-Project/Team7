"""Noise detection and filtering for painting crops; only some query images are noisy.

Introduced for Week 3 Task 1 (noise detection and filtering before the colour and texture descriptors, through
`denoise_method` in src/descriptors.py) and reused in Week 4, where src/local_pipeline.py denoises every crop
before keypoint detection. Two detectors, neither assumes in advance which noise is present:
- impulse noise: `noise_level`, the density of isolated outlier pixels (a few pixels replaced by random values);
- additive grain (e.g. Gaussian): `noise_sigma`, Donoho's robust estimate of the noise standard deviation
  from the diagonal Haar band, measured in the flattest blocks so the painting's own texture does not count.
`noise_type` reports which one (impulse first) or None, and `denoise` applies the filter that suits it: a
3x3 median for impulses (strong Gaussian grain, sigma >= 15, also triggers the impulse detector; the median
filters it too) and a bilateral filter for moderate grain. Clean images are returned untouched. Thresholds and
the grain filter are configured in src/config.py.
"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import GRAIN_FILTER, NOISE_THRESHOLD, OUTLIER_THRESHOLD, SIGMA_THRESHOLD


# ---- Detectors

def _outliers(image: NDArray[np.uint8], threshold: int = OUTLIER_THRESHOLD
              ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.uint8]]:
    """Outlier pixels (any channel more than `threshold` from its 3x3 median), the isolated ones, and the median."""
    median = cv2.medianBlur(image, 3)
    residual = np.abs(image.astype(np.int16) - median.astype(np.int16))  # int16: uint8 subtraction would wrap
    outlier = residual.max(axis=2) > threshold
    # Unnormalised 3x3 sum minus the centre = number of outlier neighbours.
    neighbours = cv2.boxFilter(outlier.astype(np.float32), -1, (3, 3), normalize=False) - outlier
    return outlier, outlier & (neighbours == 0), median


def _block_values(values: NDArray, grid: int, reduce) -> NDArray:
    """A value per block of a grid x grid split, computed with reduce."""
    h, w = values.shape
    return np.array([reduce(values[r * h // grid:(r + 1) * h // grid, c * w // grid:(c + 1) * w // grid])
                     for r in range(grid) for c in range(grid)])


def noise_level(image: NDArray[np.uint8], threshold: int = OUTLIER_THRESHOLD, grid: int = 8) -> float:
    """Estimate impulse noise as the block-median density of isolated outlier pixels.

    An outlier differs from its 3x3 median by more than `threshold` in some channel; it is isolated when none
    of its 8 neighbours is an outlier. Edges and texture give clustered outliers, and the block median
    ignores textured regions, so the value stays near 0 on clean images and rises with noise.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image.
        threshold (int): Grey levels from the 3x3 median above which a pixel is an outlier.
        grid (int): The image is split into grid x grid blocks.

    Returns:
        float: Median over blocks of the fraction of isolated outlier pixels (0 to 1); compared with
        NOISE_THRESHOLD in `noise_type`.
    """
    isolated = _outliers(image, threshold)[1]
    return float(np.median(_block_values(isolated, grid, np.mean)))


def noise_sigma(image: NDArray[np.uint8], grid: int = 8, percentile: float = 10) -> float:
    """Estimate the standard deviation of additive noise (Donoho's MAD / 0.6745 of the diagonal Haar band).

    Computed per block of a grid on the grey image and summarised by a low percentile over blocks, so flat
    regions (where only noise remains) decide and textured regions of the painting do not.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image.
        grid (int): The Haar band is split into grid x grid blocks.
        percentile (float): Percentile (0-100) of the per-block estimates that is returned.

    Returns:
        float: Estimated noise sigma in grey levels; compared with SIGMA_THRESHOLD in `noise_type`.
    """
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2]  # even size, so the 2x2 Haar blocks tile the image
    # Diagonal (HH) detail of a one-level orthonormal Haar transform.
    hh =(g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    sigmas = _block_values(hh, grid, lambda b: np.median(np.abs(b)) / 0.6745)
    return float(np.percentile(sigmas, percentile))


def noise_type(image: NDArray[np.uint8]) -> str | None:
    """Classify the noise of an image; impulse noise is tested first.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image.

    Returns:
        str | None: "impulse" if `noise_level` exceeds NOISE_THRESHOLD, otherwise "grain" if `noise_sigma`
        exceeds SIGMA_THRESHOLD, otherwise None (clean).
    """
    if noise_level(image) > NOISE_THRESHOLD:
        return "impulse"
    if noise_sigma(image) > SIGMA_THRESHOLD:
        return "grain"
    return None


# ---- Final filter: the one that suits the detected noise

def denoise(image: NDArray[np.uint8], kind: str | None = "detect") -> NDArray[np.uint8]:
    """Filter an image according to its noise: 3x3 median (impulse), GRAIN_FILTER (grain), nothing (clean).

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image.
        kind (str | None): "detect" runs `noise_type` on the image; otherwise a result of `noise_type`
            ("impulse", "grain" or None) computed beforehand. Any other value leaves the image unfiltered.

    Returns:
        NDArray[np.uint8]: The filtered (H, W, 3) uint8 BGR image, or the input object itself when no filter
        applies.
    """
    kind = noise_type(image) if kind == "detect" else kind
    if kind == "impulse":
        return cv2.medianBlur(image, 3)
    if kind == "grain":
        return remove_noise(image, **GRAIN_FILTER)
    return image


# ---- Filters available to the descriptor configurations

def remove_noise(image: NDArray[np.uint8], method: str = "median", **kwargs: Any) -> NDArray[np.uint8]:
    """Apply one denoising filter: median, bilateral, non-local means (NLM) or median followed by NLM.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image.
        method (str): "median", "bilateral", "nlm" or "median_nlm".
        **kwargs: Optional filter parameters: `ksize` (median aperture, default 3); `d`, `sigma_color`,
            `sigma_space` (bilateral, defaults 5, 50, 50); `h` (NLM strength for luminance and colour,
            default 10). Unused keys are ignored.

    Returns:
        NDArray[np.uint8]: The filtered (H, W, 3) uint8 BGR image.

    Raises:
        ValueError: If the method is unknown.
    """
    if method == "median":
        return cv2.medianBlur(image, kwargs.get("ksize", 3))
    if method == "bilateral":
        return cv2.bilateralFilter(image, kwargs.get("d", 5), kwargs.get("sigma_color", 50),
                                   kwargs.get("sigma_space", 50))
    if method == "nlm":
        h = kwargs.get("h", 10)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    if method == "median_nlm":  # the median removes isolated specks, then NLM smooths the rest
        return remove_noise(cv2.medianBlur(image, 3), "nlm", **kwargs)
    raise ValueError(f"Unknown denoising method: {method}")
