"""Week 4 Task 2-I local-feature matching: tentative pairs, one geometric verification and museum scores.

``extract_features`` detects and describes keypoints (Task 1 modules) on a resized image; ``tentative_matches``
applies the Lowe ratio test and the optional cross-check; ``match_pair`` verifies them with one RANSAC homography
and scores the museum image. The submitted method (``LOCAL_MATCH`` in ``src.config``, used by
``src.local_pipeline``) compares SIFT as RootSIFT (Hellinger), ratio 0.75, cross-check, RANSAC 5 px, and scores
distinct inliers (inliers counted once per museum keypoint). The comparison is in
``weeks/week4/results/task2/task2_results.md``.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import MATCH_DEFAULTS, TOP_K
from src.keypoints import detect_keypoints
from src.local_descriptors import DetectorConfig, LocalDescriptorConfig, describe


@dataclass(frozen=True)
class Features:
    """Keypoints of one image: positions and descriptors, row i with point i.

    Attributes:
        points: (N, 2) float32 keypoint xy in pixels of the original (not resized) image.
        descriptors: (N, D) descriptors: float32 for SIFT (RootSIFT after ``prepare_descriptors``), packed
            uint8 for ORB / BRISK.
        shape: (height, width) of the described image, or None.
    """

    points: NDArray[np.float32]
    descriptors: NDArray[np.float32] | NDArray[np.uint8]
    shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class MatchConfig:
    """Matching settings; defaults from ``MATCH_DEFAULTS``, the submitted values are ``LOCAL_MATCH``.

    Attributes:
        ratio: Lowe ratio test, best/second best distance (smaller is stricter), in (0, 1).
        mutual: Also require the match to be the best one in the other direction (cross-check).
        reprojection_px: RANSAC inlier distance in pixels of the museum image.
        max_side: Longest image side in pixels used for detection (keeps the cost bounded).
        max_points: Strongest keypoints (by detector response) kept per image.
        metric: "l2" (float) or "hellinger" (RootSIFT: L1-normalise, square root, then L2); binary descriptors
            always use Hamming.
        score: "inliers" (RANSAC homography inliers), "distinct" (inliers counted once per museum keypoint,
            which stops many-to-one degenerate homographies) or "matches" (tentative matches, no RANSAC).

    Raises:
        ValueError: On construction, if a value is out of range or the metric / score is unknown.
    """

    ratio: float = MATCH_DEFAULTS["ratio"]
    mutual: bool = MATCH_DEFAULTS["mutual"]
    reprojection_px: float = MATCH_DEFAULTS["reprojection_px"]
    max_side: int = MATCH_DEFAULTS["max_side"]
    max_points: int = MATCH_DEFAULTS["max_points"]
    metric: str = MATCH_DEFAULTS["metric"]
    score: str = MATCH_DEFAULTS["score"]

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not np.isfinite(self.ratio) or not 0 < self.ratio < 1:
            raise ValueError("ratio must be between 0 and 1.")
        if not np.isfinite(self.reprojection_px) or self.reprojection_px <= 0:
            raise ValueError("reprojection_px must be finite and positive.")
        if any(type(value) is not int or value < 1 for value in (self.max_side, self.max_points)):
            raise ValueError("max_side and max_points must be positive integers.")
        if self.metric not in ("l2", "hellinger") or self.score not in ("matches", "inliers", "distinct"):
            raise ValueError("Unsupported matching metric or score.")


@dataclass(frozen=True)
class MatchResult:
    """The score of one query / museum pair and the evidence that produced it; geometry is computed once.

    Attributes:
        score: Score selected by ``MatchConfig.score`` (0 when verification is impossible or fails).
        pairs: Tentative matches as (query point index, museum point index).
        inliers: (len(pairs),) bool RANSAC inlier mask, all False without a homography.
        homography: (3, 3) float64 query-to-museum homography, or None.
    """

    score: int
    pairs: list[tuple[int, int]]
    inliers: NDArray[np.bool_]
    homography: NDArray[np.float64] | None

    def query_inliers(self, count: int) -> NDArray[np.bool_]:
        """Mark the query points explained by this verification.

        Args:
            count: Number of query keypoints.

        Returns:
            (count,) bool array, True for query points that are RANSAC inliers.
        """
        explained = np.zeros(count, bool)
        for (query_index, _), keep in zip(self.pairs, self.inliers):
            if keep:
                explained[query_index] = True
        return explained


def prepare_descriptors(descriptors: NDArray, metric: str) -> NDArray:
    """Descriptors as the matcher compares them: RootSIFT for "hellinger", unchanged otherwise.

    Args:
        descriptors: (N, D) float32 (SIFT) or packed uint8 (binary) descriptors.
        metric: "l2" or "hellinger".

    Returns:
        (N, D) array; for "hellinger" and float32 input, each row L1-normalised and square-rooted (float32),
        so that L2 between rows is the Hellinger distance. Binary descriptors are returned unchanged.

    Raises:
        ValueError: If ``metric`` is unknown.
    """
    if metric == "hellinger" and descriptors.dtype == np.float32:
        return np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)
    if metric not in ("l2", "hellinger"):
        raise ValueError(f"Unknown metric: {metric}")
    return descriptors


def extract_features(
    gray: NDArray[np.uint8], detector: DetectorConfig, descriptor: LocalDescriptorConfig,
    config: MatchConfig = MatchConfig(),
    mask: NDArray | None = None,
) -> Features:
    """Detect and describe keypoints on a copy resized to ``config.max_side``; positions are in the original frame.

    Args:
        gray: (H, W) uint8 grayscale image (a painting crop or a museum image).
        detector: Keypoint detector configuration (``src.keypoints``).
        descriptor: Local descriptor configuration (``src.local_descriptors``).
        config: Uses ``max_side`` (longest side in pixels for detection) and ``max_points``.
        mask: Optional (H, W) array, nonzero = painting. Keeps only keypoints on the painting, so the wall left
            in a rectangular crop does not count. Without a mask, all detected points are eligible.

    Returns:
        ``Features`` with (N, 2) float32 xy in pixels of ``gray``, N <= ``config.max_points``, and the raw
        descriptors (not yet RootSIFT; see ``prepare_descriptors``).
    """
    scale = min(1.0, config.max_side / max(gray.shape))  # only downscale
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    keypoints = detect_keypoints(small, detector)
    if mask is not None:
        small_mask = cv2.resize((mask > 0).astype(np.uint8), (small.shape[1], small.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        h, w = small_mask.shape
        keypoints = [k for k in keypoints
                     if small_mask[min(int(k.pt[1]), h - 1), min(int(k.pt[0]), w - 1)]]
    keypoints = sorted(keypoints, key=lambda k: -k.response)[: config.max_points]
    points, descriptors = describe(small, keypoints, descriptor, detector=detector)
    xy = np.array([p.pt for p in points], np.float32).reshape(-1, 2) / scale
    return Features(xy, descriptors, gray.shape[:2])


def _norm(descriptors: NDArray) -> int:
    """OpenCV norm for the descriptor type: L2 for float (SIFT / RootSIFT), Hamming for packed binary."""
    return cv2.NORM_L2 if descriptors.dtype == np.float32 else cv2.NORM_HAMMING


def tentative_matches(query: Features, reference: Features, config: MatchConfig) -> list[tuple[int, int]]:
    """Index pairs that pass the Lowe ratio test and, if ``config.mutual``, the cross-check.

    Args:
        query: Query painting features.
        reference: Museum image features (same descriptor type and preparation as ``query``).
        config: Uses ``ratio`` and ``mutual``.

    Returns:
        List of (query point index, reference point index); empty if either side has fewer than 2 points
        (the ratio test needs two neighbours).
    """
    if len(query.points) < 2 or len(reference.points) < 2:
        return []
    matcher = cv2.BFMatcher(_norm(query.descriptors))
    pairs = []
    for first, second in matcher.knnMatch(query.descriptors, reference.descriptors, k=2):
        if first.distance < config.ratio * second.distance:
            pairs.append((first.queryIdx, first.trainIdx))
    if config.mutual and pairs:
        # Cross-check: keep a pair only if the query point is also the museum point's nearest neighbour.
        # This makes the matches one-to-one and gave +1 top-1 on both dev sets (task2_results.md, ablation).
        back = matcher.match(reference.descriptors, query.descriptors)
        best_for_reference = {m.queryIdx: m.trainIdx for m in back}
        pairs = [(q, r) for q, r in pairs if best_for_reference.get(r) == q]
    return pairs


def match_pair(query: Features, reference: Features, config: MatchConfig) -> MatchResult:
    """Match two images, verify with one RANSAC homography and score the museum image.

    Args:
        query: Query painting features.
        reference: Museum image features.
        config: Matching settings; ``score`` selects what is counted.

    Returns:
        ``MatchResult`` holding the score, the tentative pairs and the same RANSAC mask used for scoring
        (reused by recovery and visualisation). The score is 0 with fewer than 4 pairs or if RANSAC fails.
    """
    pairs = tentative_matches(query, reference, config)
    empty = np.zeros(len(pairs), bool)
    if config.score == "matches":
        return MatchResult(len(pairs), pairs, empty, None)
    if len(pairs) < 4:  # a homography needs 4 correspondences
        return MatchResult(0, pairs, empty, None)
    src = query.points[[q for q, _ in pairs]]
    dst = reference.points[[r for _, r in pairs]]
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, config.reprojection_px)
    if homography is None:
        return MatchResult(0, pairs, empty, homography)
    inliers = inliers.ravel().astype(bool)
    # "distinct": count each museum keypoint once, so many query points mapped onto one museum point
    # (a degenerate homography, weeks/week4/results/task2/degenerate_00005_qsd2w3.png) cannot inflate the score.
    score = (len({r for (_, r), keep in zip(pairs, inliers) if keep})
             if config.score == "distinct" else int(inliers.sum()))
    return MatchResult(score, pairs, inliers, homography)


def match_database(query: Features, database: dict[int, Features], config: MatchConfig
                    ) -> dict[int, MatchResult]:
    """Verify each museum candidate once, keeping its score and inlier evidence.

    Args:
        query: Query painting features.
        database: Museum image id -> features.
        config: Matching settings.

    Returns:
        Museum image id -> ``MatchResult``, in database order.
    """
    return {image_id: match_pair(query, reference, config) for image_id, reference in database.items()}


def score_database(
    query: Features, database: dict[int, Features], config: MatchConfig = MatchConfig(),
) -> dict[int, int]:
    """Score of the query against every database image.

    Args:
        query: Query painting features.
        database: Museum image id -> features.
        config: Matching settings.

    Returns:
        Museum image id -> score (see ``MatchConfig.score``).
    """
    return {image_id: result.score for image_id, result in match_database(query, database, config).items()}


def rank_scores(scores: dict[int, int], top_k: int = TOP_K) -> list[int]:
    """Database IDs by decreasing score; equal scores keep database order (stable sort).

    Args:
        scores: Museum image id -> score.
        top_k: Number of ids returned.

    Returns:
        The ``top_k`` best museum ids.
    """
    return sorted(scores, key=lambda image_id: -scores[image_id])[:top_k]
