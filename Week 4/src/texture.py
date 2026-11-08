"""Block-based texture descriptors (LBP, DCT, wavelet) computed on the grayscale image.

Week 3 Task 2 compares these descriptors (weeks/week3/experiments/task2_texture.py); the submitted Week 3
method fuses LBP, DCT and wavelet with the colour histogram (Task 4, src.descriptors.compute_descriptor), and that
fused descriptor is also the Week 4 global fallback.

The image is resized to a fixed square so every image gives a vector of the same length, then split into a grid
of blocks; the per-block features are concatenated. With a mask (QS2 paintings), the LBP
histograms count only painting pixels, and for DCT and wavelets the wall pixels around the painting are
replaced by the mean grey level of the painting, so the transform does not see the wall.
"""

from functools import lru_cache
from typing import Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

IMAGE_SIZE: int = 256  # side (pixels) of the square grayscale image every texture descriptor is computed on


def split_blocks(gray: NDArray, grid: int) -> list[NDArray]:
    """Split a square image into `grid` x `grid` equal blocks, row by row.

    Args:
        gray (NDArray): (S, S) image or mask, any dtype.
        grid (int): Number of blocks per side.

    Returns:
        list[NDArray]: grid * grid views of shape (S // grid, S // grid); leftover rows / columns are dropped.
    """
    step = gray.shape[0] // grid
    return [gray[r * step:(r + 1) * step, c * step:(c + 1) * step]
            for r in range(grid) for c in range(grid)]


def lbp_image(gray: NDArray[np.float32], radius: int, points: int = 8) -> NDArray[np.int64]:
    """LBP code of every pixel: bit k is 1 when neighbour k (on a circle) is >= the centre pixel.

    Neighbours that fall between pixels are bilinearly interpolated by `cv2.remap`; borders are replicated.

    Args:
        gray (NDArray[np.float32]): (H, W) float32 grayscale image.
        radius (int): Radius of the neighbour circle, in pixels.
        points (int): Number of neighbours P on the circle.

    Returns:
        NDArray[np.int64]: (H, W) int64 raw codes in [0, 2^P).
    """
    height, width = gray.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    code = np.zeros(gray.shape, dtype=np.int64)
    for k in range(points):
        angle = 2 * np.pi * k / points
        neighbour = cv2.remap(gray, (xs + radius * np.cos(angle)).astype(np.float32),
                              (ys - radius * np.sin(angle)).astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        code |= (neighbour >= gray).astype(np.int64) << k
    return code


@lru_cache(maxsize=8)
def lbp_mapping(points: int, mapping: str) -> tuple[NDArray[np.int64], int]:
    """Lookup table from raw LBP codes to histogram bins, and the number of bins.

    "none": every code is a bin (2^P). "uniform": codes with at most two 0/1 transitions keep their own bin
    and all the others share one (P(P-1)+3 bins, 59 for P=8; Ojala et al. 2002). "riu2": rotation-invariant
    uniform, the number of ones of uniform codes (P+2 bins). Cached because it only depends on its arguments.

    Args:
        points (int): Number of neighbours P.
        mapping (str): "none", "uniform" or "riu2".

    Returns:
        tuple[NDArray[np.int64], int]: (2^P,) int64 table from raw code to bin, and the number of bins.

    Raises:
        ValueError: If the mapping is unknown.
    """
    codes = np.arange(2 ** points, dtype=np.int64)
    if mapping == "none":
        return codes, 2 ** points
    bits = (codes[:, None] >> np.arange(points)) & 1
    transitions = (bits != np.roll(bits, 1, axis=1)).sum(axis=1)  # circular: last bit compared with the first
    uniform = transitions <= 2
    if mapping == "uniform":
        table = np.full(codes.shape, points * (points - 1) + 2, dtype=np.int64)
        table[uniform] = np.arange(int(uniform.sum()))
        return table, points * (points - 1) + 3
    if mapping == "riu2":
        return np.where(uniform, bits.sum(axis=1), points + 1).astype(np.int64), points + 2
    raise ValueError(f"Unknown LBP mapping: {mapping}")


def lbp_descriptor(gray: NDArray[np.uint8], grid: int = 4, radii: tuple[int, ...] = (1, 2, 3),
                   points: int = 8, mapping: str = "none",
                   mask: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
    """Multiscale LBP: per block and radius, the histogram of LBP codes (sums to 1; zeros for an empty block).

    Args:
        gray (NDArray[np.uint8]): (IMAGE_SIZE, IMAGE_SIZE) uint8 grayscale image, from `texture_gray`.
        grid (int): Number of blocks per side.
        radii (tuple[int, ...]): LBP radii in pixels; one set of block histograms per radius.
        points (int): Number of neighbours P.
        mapping (str): Code-to-bin mapping, see `lbp_mapping`.
        mask (NDArray[np.bool_] | None): (IMAGE_SIZE, IMAGE_SIZE) bool painting mask; only its pixels are
            counted. None counts every pixel.

    Returns:
        NDArray[np.float32]: (len(radii) * grid^2 * n_bins,) float32 vector, radius-major then block by block.
    """
    gray = gray.astype(np.float32)
    table, n_bins = lbp_mapping(points, mapping)
    mask_blocks = split_blocks(mask, grid) if mask is not None else None
    histograms = []
    for radius in radii:
        for i, block in enumerate(split_blocks(table[lbp_image(gray, radius, points)], grid)):
            values = block[mask_blocks[i]] if mask_blocks is not None else block.ravel()
            histograms.append(np.bincount(values, minlength=n_bins) / max(values.size, 1))
    return np.concatenate(histograms).astype(np.float32)


def zigzag_indices(size: int) -> list[tuple[int, int]]:
    """(row, col) of a size x size block in JPEG zig-zag order (low frequencies first).

    Args:
        size (int): Side of the block.

    Returns:
        list[tuple[int, int]]: size * size (row, col) pairs.
    """
    return sorted(((r, c) for r in range(size) for c in range(size)),
                  key=lambda p: (p[0] + p[1], p[1] if (p[0] + p[1]) % 2 == 0 else p[0]))


def dct_descriptor(gray: NDArray[np.uint8], grid: int = 8, n_coefs: int = 10, normalize: str = "none",
                   mask: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
    """First `n_coefs` zig-zag 2D-DCT coefficients of every block, concatenated (optionally L2-normalised).

    Args:
        gray (NDArray[np.uint8]): (IMAGE_SIZE, IMAGE_SIZE) uint8 grayscale image, from `texture_gray`.
        grid (int): Number of blocks per side.
        n_coefs (int): Coefficients kept per block, in zig-zag order (the first is the DC term).
        normalize (str): "none" or "l2" (see `_normalize`).
        mask (NDArray[np.bool_] | None): (IMAGE_SIZE, IMAGE_SIZE) bool painting mask; pixels outside it are
            replaced by the mean grey level inside it. None uses the image as it is.

    Returns:
        NDArray[np.float32]: (grid^2 * n_coefs,) float32 vector.

    Raises:
        ValueError: If `normalize` is unknown.
    """
    blocks = split_blocks(_fill_outside(gray, mask), grid)
    rows, cols = zip(*zigzag_indices(blocks[0].shape[0])[:n_coefs])
    return _normalize(np.concatenate([cv2.dct(block)[rows, cols] for block in blocks]), normalize)


def haar_level(gray: NDArray[np.float32]) -> tuple[NDArray, tuple[NDArray, NDArray, NDArray]]:
    """One 2D Haar step: the approximation and the horizontal, vertical and diagonal details.

    Args:
        gray (NDArray[np.float32]): (H, W) float32 image with even H and W.

    Returns:
        tuple[NDArray, tuple[NDArray, NDArray, NDArray]]: The (H/2, W/2) approximation (mean of each 2 x 2
        cell) and the three (H/2, W/2) detail bands.
    """
    a, b = gray[0::2, 0::2], gray[0::2, 1::2]
    c, d = gray[1::2, 0::2], gray[1::2, 1::2]
    return (a + b + c + d) / 4, ((a + b - c - d) / 4, (a - b + c - d) / 4, (a - b - c + d) / 4)


_S3 = np.sqrt(3.0)
# Both kernels float32 (the dtype the delivery was computed with): cv2.sepFilter2D needs them to match, and with
# NumPy 2 the division by the float64 scalar would turn DB2_LOW into float64.
DB2_LOW = (np.array([1 + _S3, 3 + _S3, 3 - _S3, 1 - _S3], np.float32) / (4 * np.sqrt(2.0))).astype(np.float32)
DB2_HIGH = np.array([DB2_LOW[3], -DB2_LOW[2], DB2_LOW[1], -DB2_LOW[0]]).astype(np.float32)


def db2_level(gray: NDArray[np.float32]) -> tuple[NDArray, tuple[NDArray, NDArray, NDArray]]:
    """One 2D Daubechies-2 step (separable filter bank, reflected borders, downsampling by 2).

    Args:
        gray (NDArray[np.float32]): (H, W) float32 image.

    Returns:
        tuple[NDArray, tuple[NDArray, NDArray, NDArray]]: The low-low approximation and the three detail
        bands (low-high, high-low, high-high), each (ceil(H/2), ceil(W/2)) float32.
    """
    def filt(image, kx, ky):
        """Filter rows and columns, then keep every second sample."""
        return cv2.sepFilter2D(image, cv2.CV_32F, kx, ky, borderType=cv2.BORDER_REFLECT)[0::2, 0::2]
    return filt(gray, DB2_LOW, DB2_LOW), (filt(gray, DB2_LOW, DB2_HIGH), filt(gray, DB2_HIGH, DB2_LOW),
                                          filt(gray, DB2_HIGH, DB2_HIGH))


WAVELET_STEPS = {"haar": haar_level, "db2": db2_level}


def wavelet_descriptor(gray: NDArray[np.uint8], grid: int = 4, levels: int = 3, family: str = "haar",
                       normalize: str = "none", mask: NDArray[np.bool_] | None = None) -> NDArray[np.float32]:
    """Wavelet: mean absolute value and std of each detail band, per block and level (optionally L2-normalised).

    Each block is decomposed on its own; every level continues from the previous approximation.

    Args:
        gray (NDArray[np.uint8]): (IMAGE_SIZE, IMAGE_SIZE) uint8 grayscale image, from `texture_gray`.
        grid (int): Number of blocks per side.
        levels (int): Number of decomposition levels.
        family (str): "haar" or "db2" (`WAVELET_STEPS`).
        normalize (str): "none" or "l2" (see `_normalize`).
        mask (NDArray[np.bool_] | None): (IMAGE_SIZE, IMAGE_SIZE) bool painting mask; pixels outside it are
            replaced by the mean grey level inside it. None uses the image as it is.

    Returns:
        NDArray[np.float32]: (grid^2 * levels * 3 * 2,) float32 vector.

    Raises:
        KeyError: If the family is unknown.
        ValueError: If `normalize` is unknown.
    """
    step = WAVELET_STEPS[family]
    features = []
    for block in split_blocks(_fill_outside(gray, mask), grid):
        for _ in range(levels):
            block, details = step(block)
            for band in details:
                features += [np.abs(band).mean(), band.std()]
    return _normalize(np.asarray(features), normalize)


def _normalize(vector: NDArray, normalize: str) -> NDArray[np.float32]:
    """"none", or "l2": unit Euclidean norm, so only the shape of the vector counts (like a cosine distance)."""
    if normalize == "l2":
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)  # floor: an all-zero vector stays zero
    elif normalize != "none":
        raise ValueError(f"Unknown normalisation: {normalize}")
    return np.asarray(vector, dtype=np.float32)


def _fill_outside(gray: NDArray[np.uint8], mask: NDArray[np.bool_] | None) -> NDArray[np.float32]:
    """Float copy of the image with the pixels outside the mask set to the mean of the pixels inside."""
    out = gray.astype(np.float32)
    if mask is not None and mask.any() and not mask.all():
        out[~mask] = out[mask].mean()
    return out


TEXTURE_DESCRIPTORS: dict[str, Callable[..., NDArray[np.float32]]] = {
    "lbp": lbp_descriptor,
    "dct": dct_descriptor,
    "wavelet": wavelet_descriptor,
}


def texture_gray(image: NDArray[np.uint8], mask: NDArray | None = None
                 ) -> tuple[NDArray[np.uint8], NDArray[np.bool_] | None]:
    """The fixed-size grayscale image (and mask) every texture descriptor starts from.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image (usually cropped to the painting).
        mask (NDArray | None): (H, W) mask, bool or 0/255; None for the whole image.

    Returns:
        tuple[NDArray[np.uint8], NDArray[np.bool_] | None]: The (IMAGE_SIZE, IMAGE_SIZE) uint8 grayscale image
        and the resized bool mask. The mask is None when none was given, when it covers the whole image or
        when nothing of it survives the resize.
    """
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_AREA)
    if mask is None or np.all(mask):
        return gray, None
    # Nearest neighbour keeps the resized mask binary.
    small = cv2.resize(np.asarray(mask, np.uint8) * 255, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST) > 0
    return gray, (small if small.any() else None)


def describe_texture(image: NDArray[np.uint8], kind: str, mask: NDArray | None = None,
                     **params: Any) -> NDArray[np.float32]:
    """Texture descriptor of a BGR image (optionally restricted to a mask), on its fixed-size grayscale version.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR image.
        kind (str): "lbp", "dct" or "wavelet" (`TEXTURE_DESCRIPTORS`).
        mask (NDArray | None): (H, W) painting mask, bool or 0/255; None for the whole image.
        **params: Keyword arguments of the selected descriptor function (e.g. grid, radii, n_coefs).

    Returns:
        NDArray[np.float32]: 1D float32 descriptor whose length depends only on `kind` and `params`.

    Raises:
        ValueError: If `kind` is unknown.
    """
    if kind not in TEXTURE_DESCRIPTORS:
        raise ValueError(f"Unknown texture descriptor: {kind}")
    gray, small_mask = texture_gray(image, mask)
    return TEXTURE_DESCRIPTORS[kind](gray, mask=small_mask, **params)
