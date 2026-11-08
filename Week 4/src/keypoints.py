"""Keypoint detection for Week 4 (Task 1 study and the submitted local-feature method).

``detect_keypoints`` runs one of three OpenCV detectors on a grayscale image; the type of the configuration
dataclass passed to it selects the detector. Task 1 compares the three (``weeks/week4/experiments/``); the
submitted Week 4 method uses ``DoG`` with its default settings through ``src.matching.extract_features``.

Detectors:
    - HarrisCorner:     Harris corners, no scale information.
    - HarrisLaplacian:  scale-invariant Harris-Laplace keypoints.
    - DoG:              Difference-of-Gaussians keypoints (SIFT detector), the final choice.

Typical usage:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    kps = detect_keypoints(gray, HarrisCorner(qualityLevel=0.05))
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class HarrisCorner:
    """Configuration for Harris corner detection via ``cv2.goodFeaturesToTrack``.

    Harris corners have no scale, so every keypoint gets the same fixed size
    (``patch_size``). Descriptors computed on them are therefore not scale
    invariant.

    Attributes:
        maxCorners: Maximum number of corners returned (strongest are kept).
        qualityLevel: Minimum accepted response, relative to the best corner
            found (e.g. 0.01 discards corners weaker than 1% of the best).
        minDistance: Minimum Euclidean distance between returned corners,
            in pixels.
        blockSize: Neighborhood size used to compute the structure tensor.
        k: Harris detector free parameter (typically 0.04 to 0.06).
        patch_size: Value assigned to ``KeyPoint.size`` of every keypoint.
            Descriptors such as SIFT use it as the size of the described region.
    """

    maxCorners: int = 1000
    qualityLevel: float = 0.01
    minDistance: float = 5
    blockSize: int = 3
    k: float = 0.04
    patch_size: float = 7


@dataclass
class HarrisLaplacian:
    """Configuration for Harris-Laplace detection.

    Requires ``opencv-contrib-python`` (uses ``cv2.xfeatures2d``). Keypoints
    are Harris corners that are also maxima of the scale-normalized Laplacian
    across scales, so each keypoint carries a scale in ``KeyPoint.size``.
    OpenCV returns angle 0 as a placeholder, without orientation assignment.

    Attributes:
        numOctaves: Number of octaves in the scale space.
        corn_thresh: Threshold on the Harris corner response.
        DOG_thresh: Threshold on the Laplacian/DoG response used for scale
            selection.
        maxCorners: Maximum number of keypoints returned.
        num_layers: Number of scale layers per octave.
    """

    numOctaves: int = 6
    corn_thresh: float = 0.01
    DOG_thresh: float = 0.01
    maxCorners: int = 5000
    num_layers: int = 4


@dataclass
class DoG:
    """Configuration for Difference-of-Gaussians detection (SIFT detector).

    Keypoints are extrema of the DoG scale space, refined to sub-pixel
    accuracy, with low-contrast and edge-like points rejected. They carry
    scale (``size``) and orientation (``angle``). The same location can appear
    more than once with different orientations.

    Attributes:
        nfeatures: Number of best features to keep (0 keeps all).
        nOctaveLayers: Number of layers in each octave.
        contrastThreshold: Threshold to filter out weak features in
            low-contrast regions. Larger values give fewer keypoints.
        edgeThreshold: Threshold to filter out edge-like features. Larger
            values keep more edge-like keypoints.
        sigma: Sigma of the Gaussian applied to the input image at octave 0.
    """

    nfeatures: int = 0
    nOctaveLayers: int = 3
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10
    sigma: float = 1.6


def detect_keypoints(
    img: NDArray[np.uint8], config: HarrisCorner | HarrisLaplacian | DoG,
) -> list[cv2.KeyPoint]:
    """Detect keypoints in an image.

    The type of ``config`` selects the detection method, and its fields are
    forwarded as parameters to the corresponding OpenCV function.

    Args:
        img: (H, W) uint8 grayscale image.
        config: A ``HarrisCorner``, ``HarrisLaplacian`` or ``DoG`` instance
            holding the detector parameters.

    Returns:
        List of ``cv2.KeyPoint`` (empty if none is found). ``kp.pt`` is ``(x, y)`` in pixels of ``img``
        and ``kp.size`` is the diameter of the keypoint's support region. For Harris corners the size
        is the fixed ``patch_size`` and the angle is -1 (no orientation).

    Raises:
        TypeError: If ``config`` is not one of the supported config types.
        ValueError: If ``img`` is not a nonempty single-channel uint8 image.
        RuntimeError: If Harris-Laplace is unavailable in the OpenCV build.
    """
    if not isinstance(config, (HarrisCorner, HarrisLaplacian, DoG)):
        raise TypeError(f"Unknown config type: {type(config)}")
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8 or img.size == 0:
        raise ValueError("Expected a nonempty grayscale uint8 image.")
    p = asdict(config)  # field names match the OpenCV keyword arguments
    if isinstance(config, HarrisCorner):
        size = p.pop("patch_size")  # not an OpenCV argument: the fixed KeyPoint.size
        pts = cv2.goodFeaturesToTrack(img, useHarrisDetector=True, **p)
        return [cv2.KeyPoint(float(x), float(y), size) for x, y in pts[:, 0]] if pts is not None else []
    if isinstance(config, HarrisLaplacian):
        if not hasattr(cv2, "xfeatures2d") or not hasattr(cv2.xfeatures2d, "HarrisLaplaceFeatureDetector_create"):
            raise RuntimeError("Harris-Laplace requires opencv-contrib-python; install the project dependencies.")
        return list(cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create(**p).detect(img))
    if isinstance(config, DoG):
        return list(cv2.SIFT_create(**p).detect(img, None))
    raise TypeError(f"Unknown config type: {type(config)}")
