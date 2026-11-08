"""Week 4 local appearance descriptors (SIFT, ORB, BRISK), computed on separately detected keypoints.

``describe`` computes descriptors on keypoints from ``src.keypoints`` and adapts their metadata when a detector
does not provide what the extractor expects (Harris orientations, SIFT pyramid octave, ORB patch size). Task 1
compares every detector / descriptor pair (``weeks/week4/experiments/local_descriptors.ipynb``); the submitted
Week 4 method uses DoG + ``SIFTConfig()`` through ``src.matching.extract_features``, and the SIFT vectors are
compared as RootSIFT in ``src.matching``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.keypoints import DoG, HarrisCorner, HarrisLaplacian


@dataclass(frozen=True)
class SIFTConfig:
    """SIFT scale-space settings; match these to the DoG detector settings.

    DoG orientation and scale are retained. Harris points are prepared by
    the orientation and pyramid adapters before OpenCV computes descriptors.
    SIFT descriptors have 128 float32 values and are compared with L2 distance.

    Attributes:
        nOctaveLayers: Layers per octave of the scale space (same meaning as in ``DoG``).
        sigma: Gaussian sigma at octave 0 (same meaning as in ``DoG``).
    """

    nOctaveLayers: int = 3
    sigma: float = 1.6


@dataclass(frozen=True)
class ORBConfig:
    """ORB on external keypoints: 256 packed bits, Hamming distance.

    Position and orientation are retained; octave is reset to zero and the
    patch size is fixed in original-image pixels. SIFT/Harris-Laplace pyramid
    levels cannot be interpreted as ORB levels. This is not scale-invariant.
    WTA_K is fixed to 2. The detector is always supplied separately.

    Attributes:
        patchSize: Side of the described patch in pixels (odd, >= 3).
        edgeThreshold: Border in pixels where OpenCV drops keypoints (nonnegative).
    """

    patchSize: int = 31
    edgeThreshold: int = 31


@dataclass(frozen=True)
class BRISKConfig:
    """BRISK description (512 packed bits, compared with Hamming distance).

    OpenCV estimates orientation from the image even with supplied points.
    Detector settings are omitted because describe() never detects points.

    Attributes:
        patternScale: Scale of the sampling pattern relative to ``KeyPoint.size`` (finite, positive).
    """

    patternScale: float = 1.0


LocalDescriptorConfig = SIFTConfig | ORBConfig | BRISKConfig  # any supported descriptor configuration
DescriptorArray = NDArray[np.float32] | NDArray[np.uint8]   # SIFT float32 rows, ORB / BRISK packed uint8 rows
DetectorConfig = HarrisCorner | HarrisLaplacian | DoG       # any supported detector configuration


def _create_extractor(config: LocalDescriptorConfig) -> cv2.Feature2D:
    """Validate settings and create only the requested OpenCV extractor."""
    if isinstance(config, SIFTConfig):
        if (not isinstance(config.nOctaveLayers, int) or config.nOctaveLayers < 1
                or not np.isfinite(config.sigma) or config.sigma <= 0):
            raise ValueError("nOctaveLayers must be a positive integer and sigma finite and positive.")
        return cv2.SIFT_create(nOctaveLayers=config.nOctaveLayers, sigma=config.sigma)
    if isinstance(config, ORBConfig):
        if (not isinstance(config.patchSize, int) or config.patchSize < 3 or config.patchSize % 2 == 0
                or not isinstance(config.edgeThreshold, int) or config.edgeThreshold < 0):
            raise ValueError("ORB needs an odd patchSize >= 3 and nonnegative edgeThreshold.")
        # One level: external keypoints carry no ORB pyramid level (see ORBConfig).
        return cv2.ORB_create(patchSize=config.patchSize, edgeThreshold=config.edgeThreshold,
                              nlevels=1, WTA_K=2)
    if isinstance(config, BRISKConfig):
        if not np.isfinite(config.patternScale) or config.patternScale <= 0:
            raise ValueError("patternScale must be finite and positive.")
        return cv2.BRISK_create(patternScale=config.patternScale)
    raise TypeError(f"Unknown descriptor config type: {type(config)}")


def _dominant_orientation(image: NDArray[np.uint8], point: cv2.KeyPoint) -> float:
    """Dominant gradient orientation in degrees of a scale-normalised patch (36 bins, one peak, for Harris points)."""
    # Single peak, unlike SIFT's multiple-peak assignment; DoG orientations are never replaced by this.
    width = 31
    offsets = np.linspace(-2.25, 2.25, width, dtype=np.float32) * point.size
    x, y = np.meshgrid(offsets + point.pt[0], offsets + point.pt[1])
    patch = cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    patch = cv2.GaussianBlur(patch.astype(np.float32), (3, 3), 0.8)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    yy, xx = np.mgrid[:width, :width].astype(np.float32) - width // 2
    weights = np.exp(-(xx * xx + yy * yy) / (2 * (width / 6) ** 2))
    bins = np.floor(angle / 10).astype(np.int32) % 36
    histogram = np.bincount(bins.ravel(), weights=(weights * magnitude).ravel(), minlength=36)
    for _ in range(6):
        histogram = (np.roll(histogram, 1) + histogram + np.roll(histogram, -1)) / 3
    peak = int(np.argmax(histogram))
    left, center, right = histogram[(peak - 1) % 36], histogram[peak], histogram[(peak + 1) % 36]
    curvature = left - 2 * center + right
    correction = 0.5 * (left - right) / curvature if abs(curvature) > 1e-12 else 0.0
    return float(((peak + 0.5 + correction) * 10) % 360)


def _prepare_points(
    image: NDArray[np.uint8], keypoints: Sequence[cv2.KeyPoint],
    config: LocalDescriptorConfig, detector: DetectorConfig | None,
) -> list[cv2.KeyPoint]:
    """Validated copies of the points with orientation, octave and size adapted to the extractor."""
    supplied = []
    for point in keypoints:
        x, y = point.pt
        if (not np.isfinite([x, y, point.size]).all() or point.size <= 0
                or not 0 <= x < image.shape[1] or not 0 <= y < image.shape[0]):
            raise ValueError("Keypoints must have finite in-image coordinates and positive sizes.")
        octave, size, angle = point.octave, point.size, point.angle
        unoriented = isinstance(detector, (HarrisCorner, HarrisLaplacian)) or angle < 0
        if not isinstance(config, BRISKConfig) and unoriented:
            angle = _dominant_orientation(image, point)
        if isinstance(config, SIFTConfig) and (isinstance(detector, (HarrisCorner, HarrisLaplacian)) or point.angle < 0):
            # Harris-Laplace levels are not SIFT's packed octave/layer field.
            level = np.log2(size / (2 * config.sigma))
            largest = max(-1, int(np.floor(np.log2(min(image.shape)))) - 2)
            sift_octave = int(np.clip(np.floor(level), -1, largest))
            layer = int(np.clip(round((level - sift_octave) * config.nOctaveLayers),
                                0, config.nOctaveLayers + 2))
            octave = (sift_octave & 255) | (layer << 8)
        elif isinstance(config, ORBConfig):
            if not np.isfinite(angle) or not 0 <= angle < 360:
                raise ValueError("ORB needs a finite orientation in [0, 360).")
            octave, size = 0, float(config.patchSize)
        supplied.append(cv2.KeyPoint(x, y, size, angle, point.response, octave, point.class_id))
    return supplied


def describe(
    image: NDArray[np.uint8],
    keypoints: Sequence[cv2.KeyPoint],
    config: LocalDescriptorConfig = SIFTConfig(),
    *, detector: DetectorConfig | None = None,
) -> tuple[list[cv2.KeyPoint], DescriptorArray]:
    """Describe supplied keypoints on the same grayscale image as detection.

    Use the returned keypoints: descriptor row i belongs to returned point i, and the extractor may discard
    input points or change their orientation. The image and input keypoints are not modified (points are
    copied). For Harris / Harris-Laplace points the missing orientations are estimated and their scale is
    mapped into SIFT's pyramid convention; DoG metadata is retained. ORB always uses a fixed pixel-scale patch
    (see ``ORBConfig``). Detection, preprocessing, matching and unknown rejection are separate operations.

    Args:
        image: (H, W) uint8 grayscale image the keypoints were detected on.
        keypoints: Keypoints with finite in-image ``pt`` (x, y pixels) and positive ``size``.
        config: Descriptor settings; its type selects SIFT, ORB or BRISK.
        detector: Configuration of the detector that produced ``keypoints``; pass it for Harris /
            Harris-Laplace points so their orientation and octave are adapted.

    Returns:
        Tuple ``(points, descriptors)``: the kept keypoints and an (N, D) array, row i for point i. SIFT gives
        float32 (N, 128); ORB uint8 (N, 32); BRISK uint8 (N, 64). Binary values are packed bytes: do not cast
        them to float or use L2. Without points, an empty list and a correctly typed (0, D) array.

    Raises:
        TypeError: If the descriptor or detector configuration is unsupported.
        ValueError: If the image, configuration, keypoint coordinates or ORB orientations are invalid.
    """
    if not isinstance(config, (SIFTConfig, ORBConfig, BRISKConfig)):
        raise TypeError(f"Unknown descriptor config type: {type(config)}")
    if detector is not None and not isinstance(detector, (HarrisCorner, HarrisLaplacian, DoG)):
        raise TypeError(f"Unknown detector config type: {type(detector)}")
    if (not isinstance(image, np.ndarray) or image.ndim != 2
            or image.dtype != np.uint8 or image.size == 0):
        raise ValueError("Expected a nonempty grayscale uint8 image.")
    extractor = _create_extractor(config)
    dtype = np.float32 if isinstance(config, SIFTConfig) else np.uint8
    if len(keypoints) == 0:
        return [], np.empty((0, extractor.descriptorSize()), dtype=dtype)

    supplied = _prepare_points(image, keypoints, config, detector)
    points, descriptors = extractor.compute(image, supplied)
    if descriptors is None:
        return [], np.empty((0, extractor.descriptorSize()), dtype=dtype)
    return list(points), descriptors
