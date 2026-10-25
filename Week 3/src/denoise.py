"""Noise detection and removal for an unknown noise model; only some query images are noisy (Week 3, Task 1).

Two detectors, neither assumes in advance which noise is present:
- impulse noise: `noise_level`, the density of isolated outlier pixels (a few pixels replaced by random values);
- additive grain (e.g. Gaussian): `noise_sigma`, Donoho's robust estimate of the noise standard deviation
  from the diagonal Haar band, measured in the flattest blocks so the painting's own texture does not count.
`noise_type` reports which one (impulse first) or None, and `denoise` applies the filter that suits it: a
3x3 median for impulses (strong Gaussian grain, sigma >= 15, also triggers the impulse detector; the median
filters it too) and a bilateral filter for moderate grain. Clean images are returned untouched. The thresholds
(src/config.py) and the filters were chosen on development data (experiments/task1_noise.py,
results/week3/task1/). The other filters of `remove_noise` are the ones compared in the Task 1 tables.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import GRAIN_FILTER, NOISE_THRESHOLD, OUTLIER_THRESHOLD, SIGMA_THRESHOLD

DENOISERS = ("median", "bilateral", "switching_median", "gaussian", "mean", "open_close", "nlm", "median_nlm")


# ---- Detectors

def _outliers(image: NDArray[np.uint8], threshold: int = OUTLIER_THRESHOLD
              ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.uint8]]:
    """Outlier pixels (any channel more than `threshold` from its 3x3 median), the isolated ones, and the median."""
    median = cv2.medianBlur(image, 3)
    residual = np.abs(image.astype(np.int16) - median.astype(np.int16))
    outlier = residual.max(axis=2) > threshold
    neighbours = cv2.boxFilter(outlier.astype(np.float32), -1, (3, 3), normalize=False) - outlier
    return outlier, outlier & (neighbours == 0), median


def _block_values(values: NDArray, grid: int, reduce) -> NDArray:
    """A value per block of a grid x grid split, computed with reduce."""
    h, w = values.shape
    return np.array([reduce(values[r * h // grid:(r + 1) * h // grid, c * w // grid:(c + 1) * w // grid])
                     for r in range(grid) for c in range(grid)])


def noise_level(image: NDArray[np.uint8], threshold: int = OUTLIER_THRESHOLD, grid: int = 8) -> float:
    """Median over grid blocks of the density of isolated outlier pixels (impulse-noise estimate).

    An outlier differs from its 3x3 median by more than `threshold`; it is isolated when none of its
    8 neighbours is an outlier. Edges and texture give clustered outliers, and the block median
    ignores textured regions, so the value stays near 0 on clean images and rises with noise.
    """
    isolated = _outliers(image, threshold)[1]
    return float(np.median(_block_values(isolated, grid, np.mean)))


def noise_sigma(image: NDArray[np.uint8], grid: int = 8, percentile: float = 10) -> float:
    """Robust estimate of additive noise: Donoho's MAD / 0.6745 of the diagonal Haar band of the grey image.

    Computed per block of a grid and summarised by a low percentile over blocks, so flat regions (where only
    noise remains) decide and textured regions of the painting do not.
    """
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g[: g.shape[0] // 2 * 2, : g.shape[1] // 2 * 2]
    hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]) / 2
    sigmas = _block_values(hh, grid, lambda b: np.median(np.abs(b)) / 0.6745)
    return float(np.percentile(sigmas, percentile))


def noise_type(image: NDArray[np.uint8]) -> str | None:
    """"impulse", "grain" or None (clean)."""
    if noise_level(image) > NOISE_THRESHOLD:
        return "impulse"
    if noise_sigma(image) > SIGMA_THRESHOLD:
        return "grain"
    return None


# ---- Final filter: the one that suits the detected noise

def denoise(image: NDArray[np.uint8], kind: str | None = "detect") -> NDArray[np.uint8]:
    """Filter according to the detected noise: 3x3 median (impulse), GRAIN_FILTER (grain), nothing (clean)."""
    kind = noise_type(image) if kind == "detect" else kind
    if kind == "impulse":
        return cv2.medianBlur(image, 3)
    if kind == "grain":
        return remove_noise(image, **GRAIN_FILTER)
    return image


# ---- Filters compared in Task 1 (results/week3/task1/filters_psnr.csv, grain_filters.csv)

def switching_median(image: NDArray[np.uint8], threshold: int = OUTLIER_THRESHOLD,
                     isolated_only: bool = False) -> NDArray[np.uint8]:
    """Replace only the outlier pixels by their 3x3 median; every other pixel keeps its value (not adopted)."""
    outlier, isolated, median = _outliers(image, threshold)
    out = image.copy()
    replace = isolated if isolated_only else outlier
    out[replace] = median[replace]
    return out


def open_close(image: NDArray[np.uint8], ksize: int = 3) -> NDArray[np.uint8]:
    """Morphological opening then closing, removing bright and dark specks (Pol Pérez's idea)."""
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def remove_noise(image: NDArray[np.uint8], method: str = "median", **kwargs) -> NDArray[np.uint8]:
    """Filter a BGR uint8 image with one of `DENOISERS`; kwargs override the filter parameters.

    Raises:
        ValueError: If the method is unknown.
    """
    if method == "median":
        return cv2.medianBlur(image, kwargs.get("ksize", 3))
    if method == "bilateral":
        return cv2.bilateralFilter(image, kwargs.get("d", 5), kwargs.get("sigma_color", 50),
                                   kwargs.get("sigma_space", 50))
    if method == "switching_median":
        return switching_median(image, kwargs.get("threshold", OUTLIER_THRESHOLD), kwargs.get("isolated_only", False))
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kwargs.get("ksize", 3),) * 2, kwargs.get("sigma", 0))
    if method == "mean":
        return cv2.blur(image, (kwargs.get("ksize", 3),) * 2)
    if method == "open_close":
        return open_close(image, kwargs.get("ksize", 3))
    if method == "nlm":
        h = kwargs.get("h", 10)
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    if method == "median_nlm":  # the median removes isolated specks, then NLM smooths the rest
        return remove_noise(cv2.medianBlur(image, 3), "nlm", **kwargs)
    raise ValueError(f"Unknown denoising method: {method}")
