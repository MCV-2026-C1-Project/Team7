"""Week 4 submitted method (Tasks 1, 2-I and 2-II): segment, match local features, recover paintings, decide unknowns.

Per query photo: background removal into up to ``max_paintings`` regions (three by default) in reading order;
per region, DoG keypoints + SIFT on the denoised grayscale crop, compared as RootSIFT with ratio 0.75,
cross-check and one RANSAC homography (5 px), score = distinct inliers; extra known paintings inside a merged
region are recovered from the unexplained keypoints (>= ``RECOVERY_MIN_INLIERS``, 10); finally each painting
is accepted with a local score >= 10, else with the Week 3 global descriptor if its best / second distance
ratio is < 0.85, else answered [-1] (``src.unknowns``). ``run`` is used by ``main.py`` (``--week 4``), the
Week 4 experiments and the inspection notebook. Museum features and global descriptors are cached in
``.cache/``. Method and API: ``weeks/week4/task2_matching.md``.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import (GLOBAL_METHOD, GLOBAL_RATIO, LOCAL_MATCH, MAX_PAINTINGS, PRESETS, RECOVERY_MIN_INLIERS, TOP_K,
                        UNKNOWN_THRESHOLD)
from src.data_loading import dataset_signature, get_image_id, list_images, load_image
from src.denoise import denoise
from src.descriptors import compute_descriptor, crop_to_mask
from src.keypoints import DoG
from src.local_descriptors import DetectorConfig, LocalDescriptorConfig, SIFTConfig
from src.matching import Features, MatchConfig, MatchResult, extract_features, match_database, prepare_descriptors, rank_scores
from src.pipeline import Progress, build_database
from src.retrieval import retrieve
from src.segmentation import Painting, order_paintings, remove_background
from src.unknowns import Decision, UNKNOWN, decide, reject_unknown

CACHE_DIR: Path = Path(__file__).resolve().parents[1] / ".cache" / "bbdd_local"  # museum local-feature cache


def _cache_key(corpus_dir: Path, settings: object, sources: tuple[str, ...]) -> str:
    """16-hex cache key of the corpus signature, settings, OpenCV / NumPy versions and the bytes of src/<sources>."""
    # Hashing the source files means any edit to them (even a comment) invalidates the cache.
    digest = hashlib.sha256(repr((dataset_signature(corpus_dir), settings, cv2.__version__, np.__version__)).encode())
    for name in sources:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class LocalMethod:
    """Settings of the Week 4 method: detection, description, matching, recovery and rejection.

    The defaults are the submitted method (values in ``src.config``).

    Attributes:
        detector: Keypoint detector (DoG).
        descriptor: Local descriptor (SIFT).
        match: Matching settings (``LOCAL_MATCH``: RootSIFT, ratio 0.75, cross-check, distinct inliers).
        unknown_threshold: Minimum best local score to accept a painting (``UNKNOWN_THRESHOLD``).
        max_paintings: Maximum paintings per query photo (``MAX_PAINTINGS[4]``).
        recovery_min_inliers: Minimum score to recover an extra painting inside a merged region (>= 4).
        global_ratio: Global fallback accepted only below this best / second distance ratio.

    Raises:
        ValueError: On construction, if a value is out of range.
    """

    detector: DetectorConfig = field(default_factory=DoG)
    descriptor: LocalDescriptorConfig = field(default_factory=SIFTConfig)
    match: MatchConfig = field(default_factory=lambda: MatchConfig(**LOCAL_MATCH))
    unknown_threshold: int = UNKNOWN_THRESHOLD
    max_paintings: int = MAX_PAINTINGS[4]
    recovery_min_inliers: int = RECOVERY_MIN_INLIERS
    global_ratio: float = GLOBAL_RATIO

    def __post_init__(self) -> None:
        """Validate the settings."""
        if type(self.max_paintings) is not int or self.max_paintings < 1:
            raise ValueError("max_paintings must be a positive integer.")
        if type(self.unknown_threshold) is not int or self.unknown_threshold < 1:
            raise ValueError("unknown_threshold must be a positive integer.")
        if type(self.recovery_min_inliers) is not int or self.recovery_min_inliers < 4:  # a homography needs 4
            raise ValueError("recovery_min_inliers must be an integer >= 4.")
        if not np.isfinite(self.global_ratio) or not 0 < self.global_ratio < 1:
            raise ValueError("global_ratio must be between 0 and 1.")

    def cache_key(self, corpus_dir: Path) -> str:
        """Key of the museum feature cache.

        Museum features depend on the corpus and extraction settings, not on matching or rejection, so only
        those settings and the sources of this module, matching.py, keypoints.py and local_descriptors.py are
        hashed.

        Args:
            corpus_dir: Museum (BBDD) folder.

        Returns:
            16-character hexadecimal key.
        """
        settings = (self.detector, self.descriptor, self.match.max_side, self.match.max_points)
        return _cache_key(corpus_dir, settings,
                          ("local_pipeline.py", "matching.py", "keypoints.py", "local_descriptors.py"))

    def score_cache_key(self, query_dir: Path, corpus_dir: Path) -> str:
        """Key of the query-score caches of the Week 4 experiments.

        Query scores also depend on segmentation, recovery and the query files, so the whole method, the query
        signature, the museum key and every ``src/*.py`` file are hashed.

        Args:
            query_dir: Query dataset folder.
            corpus_dir: Museum (BBDD) folder.

        Returns:
            16-character hexadecimal key.
        """
        digest = hashlib.sha256(repr((self, dataset_signature(query_dir), self.cache_key(corpus_dir))).encode())
        for path in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]


@dataclass(frozen=True)
class PaintingMatch:
    """A final painting region and its scores; the optional evidence is retained only with debug=True.

    Attributes:
        region: Painting region (``mask`` (H, W) uint8 0/255 in photo pixels, ``bbox`` (x0, y0, x1, y1)).
        scores: Museum id -> local score for this painting.
        features: Query features this painting was matched with (all region keypoints, or for a recovered
            painting those still unexplained when it was found), debug only.
        verification: ``MatchResult`` of the best museum candidate, debug only.
        gray: (h, w) uint8 denoised grayscale crop that was matched, debug only.
        source_bbox: bbox of the segmented region this painting came from, debug only.
        source_keypoints: Keypoints extracted from that region, debug only (0 otherwise).
    """

    region: Painting
    scores: dict[int, int]
    features: Features | None = None
    verification: MatchResult | None = None
    gray: NDArray[np.uint8] | None = None
    source_bbox: tuple[int, int, int, int] | None = None
    source_keypoints: int = 0


def _gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """BGR to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def museum_features(corpus_dir: Path, method: LocalMethod, cache_dir: Path = CACHE_DIR) -> dict[int, Features]:
    """Load or compute the raw museum features, cached as ``local-<cache_key>.npz``.

    The key includes the corpus signature, so caches cannot be shared accidentally across corpora.

    Args:
        corpus_dir: Museum (BBDD) folder.
        method: Uses its detector, descriptor and ``match.max_side`` / ``match.max_points``.
        cache_dir: Cache folder.

    Returns:
        Museum image id -> ``Features`` (raw descriptors, not RootSIFT; see ``matching_database``).
    """
    path = cache_dir / f"local-{method.cache_key(corpus_dir)}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return {int(i): Features(data[f"p{i}"], data[f"d{i}"], tuple(data[f"s{i}"].tolist()))
                    for i in data["ids"].tolist()}
    database = {get_image_id(p): extract_features(_gray(load_image(p)), method.detector, method.descriptor, method.match)
                for p in list_images(corpus_dir)}
    arrays = {"ids": np.array(list(database), np.int64)}
    for i, features in database.items():
        arrays[f"p{i}"], arrays[f"d{i}"], arrays[f"s{i}"] = features.points, features.descriptors, np.array(features.shape)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return database


def matching_database(corpus_dir: Path, method: LocalMethod) -> dict[int, Features]:
    """Cached museum features with descriptors prepared for the matching metric (RootSIFT for Hellinger).

    Args:
        corpus_dir: Museum (BBDD) folder.
        method: Local method; ``match.metric`` selects the preparation.

    Returns:
        Museum image id -> ``Features`` ready for ``match_database``.
    """
    return {i: Features(f.points, prepare_descriptors(f.descriptors, method.match.metric), f.shape)
            for i, f in museum_features(corpus_dir, method).items()}


def global_database(corpus_dir: Path, cache_dir: Path = CACHE_DIR.parent / "global_descriptors", *,
                    progress: Progress | None = None) -> dict[int, NDArray[np.float32]]:
    """Load or compute the Week 3 global descriptors of the museum, used by the unknown fallback.

    Cached as ``global-<key>.npz`` with the same corpus-aware invalidation as the local features; the key hashes
    the descriptor configuration and pipeline.py, descriptors.py, colour.py, texture.py, denoise.py, config.py.

    Args:
        corpus_dir: Museum (BBDD) folder.
        cache_dir: Cache folder.
        progress: Optional callback ``(done, total)`` while descriptors are computed (see ``build_database``).

    Returns:
        Museum image id -> 1-D float32 ``GLOBAL_PRESET`` descriptor.
    """
    config = PRESETS[GLOBAL_METHOD]["descriptor"]
    key = _cache_key(corpus_dir, config, ("pipeline.py", "descriptors.py", "colour.py", "texture.py", "denoise.py", "config.py"))
    path = cache_dir / f"global-{key}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return {int(i): data[f"d{i}"] for i in data["ids"].tolist()}
    database = build_database(corpus_dir, config, progress=progress)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(list(database), np.int64), **{f"d{i}": d for i, d in database.items()})
    return database


def _query_regions(image: NDArray[np.uint8], segment: bool, max_paintings: int) -> list[Painting]:
    """Segmented regions in reading order, or the whole image if segmentation is disabled or finds none."""
    regions = remove_background(image, max_paintings=max_paintings)[1] if segment else []
    height, width = image.shape[:2]
    return order_paintings(regions) or [
        {"mask": np.full((height, width), 255, np.uint8), "bbox": (0, 0, width, height)}]


def query_paintings(image: NDArray[np.uint8], segment: bool = True, *, max_paintings: int = 3
                    ) -> list[tuple[NDArray[np.uint8], NDArray[np.bool_] | None]]:
    """Denoised grayscale crops and masks of the query paintings, as matched by the method.

    Also used by the descriptor / metric studies of Task 2-I.

    Args:
        image: (H, W, 3) uint8 BGR query photo.
        segment: Remove the background; if False (or nothing is found) the whole image is one painting.
        max_paintings: Maximum regions returned by the segmentation.

    Returns:
        Per painting in reading order: (h, w) uint8 grayscale crop of the region's bounding box, and its
        (h, w) bool mask of the painting inside that crop.
    """
    crops = [crop_to_mask(image, p["mask"] > 0) for p in _query_regions(image, segment, max_paintings)]
    return [(_gray(denoise(crop)), mask) for crop, mask in crops]


def _recover_paintings(features: Features, database: dict[int, Features], matches: dict[int, MatchResult],
                       method: LocalMethod, limit: int
                       ) -> list[tuple[tuple[int, int, int, int] | None, Features, dict[int, MatchResult]]]:
    """Up to ``limit`` (crop bbox, features, matches) of known paintings found greedily in one region's keypoints."""
    # Recovery uses its own minimum support (recovery_min_inliers), so the unknown threshold never changes
    # the number of regions. Only known paintings with RANSAC support are split off; an unknown painting
    # merged into the region is not separated.
    original = [(None, features, matches)]
    if limit < 2:  # no free slot for an extra painting
        return original
    found, used = [], set()
    rest, current = features, matches
    for _ in range(limit):
        if not current:
            break
        best = max(current, key=lambda i: current[i].score)
        verification = current[best]
        if verification.score < method.recovery_min_inliers:
            break
        explained = verification.query_inliers(len(rest.points))
        if np.count_nonzero(explained) < 4:
            break
        # The painting's box (crop pixels) is the bounding box of its inlier keypoints.
        points = rest.points[explained]
        x0, y0 = np.floor(points.min(axis=0)).astype(int)
        x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 1
        found.append(((int(x0), int(y0), int(x1), int(y1)), rest, current))
        used.add(best)
        # Match only the unexplained keypoints again, against the museum images not yet used.
        rest = Features(rest.points[~explained], rest.descriptors[~explained], rest.shape)
        if len(rest.points) < 4 or len(found) == limit:
            break
        current = match_database(rest, {i: ref for i, ref in database.items() if i not in used}, method.match)
    # One painting found = no split: keep the whole region and its original matches.
    return found if len(found) > 1 else original


def _match_regions(image: NDArray[np.uint8], regions: list[Painting], database: dict[int, Features],
                    method: LocalMethod, segment: bool, debug: bool) -> list[PaintingMatch]:
    """Match every region (with recovery inside it) and return the final paintings in reading order."""
    matched = []
    for index, region in enumerate(regions):
        crop, mask = crop_to_mask(image, region["mask"] > 0)
        gray = _gray(denoise(crop))
        features = extract_features(gray, method.detector, method.descriptor, method.match, mask=mask)
        features = Features(features.points, prepare_descriptors(features.descriptors, method.match.metric), features.shape)
        matches = match_database(features, database, method.match)
        # Paintings this region may become: the query limit minus the paintings already found and one slot
        # reserved for each region still to process, so separately segmented (possibly unknown) regions keep
        # their slots. Without segmentation the whole image is one painting.
        budget = method.max_paintings - len(matched) - (len(regions) - index - 1) if segment else 1
        for box, used_features, candidates in _recover_paintings(features, database, matches, method, budget):
            found = region
            if box is not None:
                # Recovered painting: shift its crop box to photo pixels, clip it to the region and keep the
                # region mask inside it.
                rx0, ry0, rx1, ry1 = region["bbox"]
                x0, y0, x1, y1 = box
                x0, y0 = max(rx0, rx0 + x0), max(ry0, ry0 + y0)
                x1, y1 = min(rx1, rx0 + x1), min(ry1, ry0 + y1)
                found_mask = np.zeros_like(region["mask"])
                found_mask[y0:y1, x0:x1] = region["mask"][y0:y1, x0:x1]
                found = {"bbox": (x0, y0, x1, y1), "mask": found_mask}
            scores = {i: candidates[i].score if i in candidates else 0 for i in database}
            best = rank_scores(scores, 1)[0]
            matched.append(PaintingMatch(found, scores, used_features if debug else None,
                                          candidates[best] if debug else None, gray if debug else None,
                                          region["bbox"] if debug else None, len(features.points) if debug else 0))
    # Recovered paintings change the layout, so the final regions are put in reading order again.
    by_region = {id(m.region): m for m in matched}
    return [by_region[id(region)] for region in order_paintings([m.region for m in matched])]


def global_candidates(image: NDArray[np.uint8], global_database: dict[int, NDArray[np.float32]], segment: bool = True,
                      *, paintings: list[Painting] | None = None, max_paintings: int = 3
                      ) -> list[tuple[list[int], float]]:
    """Week 3 global rankings and best / second distance ratios for the supplied regions.

    Args:
        image: (H, W, 3) uint8 BGR query photo.
        global_database: Museum id -> global descriptor (``global_database``).
        segment: Used only when ``paintings`` is None: segment the photo to get the regions.
        paintings: Regions to describe (the final regions of the local matching); None segments again.
        max_paintings: Maximum regions when segmenting here.

    Returns:
        Per region: (top-``TOP_K`` museum ids, best / second distance ratio; inf with a single candidate).
    """
    config, similarity = PRESETS[GLOBAL_METHOD]["descriptor"], PRESETS[GLOBAL_METHOD]["similarity"]
    regions = paintings if paintings is not None else _query_regions(image, segment, max_paintings)
    candidates = []
    for region in regions:
        result = retrieve({0: compute_descriptor(image, config, region["mask"])}, global_database,
                          similarity_function=similarity, top_k=TOP_K)[0]
        distances = result["scores"]
        ratio = float(distances[0] / max(distances[1], 1e-12)) if len(distances) > 1 else float("inf")
        candidates.append((result["top_k_ids"], ratio))
    return candidates


def _decisions(image: NDArray[np.uint8], rankings: list[list[int]], scores: list[dict[int, int]], method: LocalMethod,
                global_db: dict[int, NDArray[np.float32]], segment: bool, paintings: list[Painting] | None
                ) -> list[Decision]:
    """One ``Decision`` per painting; global descriptors are computed only if some local score is too low."""
    if len(rankings) != len(scores) or (paintings is not None and len(paintings) != len(scores)):
        raise ValueError("Rankings, scores and supplied regions must have the same length.")
    low = [reject_unknown(ranking, s, method.unknown_threshold) == [UNKNOWN] for ranking, s in zip(rankings, scores)]
    candidates = global_candidates(image, global_db, segment, paintings=paintings,
                                   max_paintings=method.max_paintings) if any(low) else []
    return [decide(ranking, s, method.unknown_threshold, candidates[i] if low[i] and candidates else None,
                    global_ratio=method.global_ratio) for i, (ranking, s) in enumerate(zip(rankings, scores))]


def process_query(image: NDArray[np.uint8], database: dict[int, Features], method: LocalMethod, *,
                  global_database: dict[int, NDArray[np.float32]] | None = None, segment: bool = True,
                  top_k: int | None = TOP_K, debug: bool = False) -> dict:
    """Run the Week 4 method on one query photo.

    Args:
        image: (H, W, 3) uint8 BGR query photo.
        database: Prepared museum features (``matching_database``).
        method: Local method settings.
        global_database: Museum global descriptors; if given, unknown decisions are made (local threshold,
            then global fallback). If None, ``decisions`` and ``answers`` are empty.
        segment: Remove the background; if False the whole photo is one painting.
        top_k: Ranking depth; None ranks the whole museum.
        debug: Also return the actual matching inputs, inliers and initial regions.

    Returns:
        Dict with, per final painting in reading order: ``rankings`` (museum ids), ``scores`` (museum id ->
        score), ``paintings`` (regions), ``decisions`` (``Decision``) and ``answers`` (ranking or [-1]);
        plus ``matches`` (``PaintingMatch`` list) and ``initial_regions`` (empty unless ``debug``).
    """
    initial = _query_regions(image, segment, method.max_paintings)
    matched = _match_regions(image, initial, database, method, segment, debug)
    regions, scores = [m.region for m in matched], [m.scores for m in matched]
    rankings = [rank_scores(s, top_k if top_k is not None else len(s)) for s in scores]
    # The fallback describes the same final regions (passed as paintings) instead of segmenting again.
    decisions = (_decisions(image, rankings, scores, method, global_database, segment, regions)
                 if global_database is not None else [])
    return {"rankings": rankings, "scores": scores, "paintings": regions, "decisions": decisions,
            "answers": [d.ranking for d in decisions], "matches": matched if debug else [],
            "initial_regions": initial if debug else []}


def rank_query(image: NDArray[np.uint8], database: dict[int, Features], method: LocalMethod, *,
               segment: bool = True, top_k: int | None = TOP_K) -> tuple[list[list[int]], list[dict[int, int]]]:
    """Raw local rankings and scores of one photo, before unknown rejection.

    Args:
        image: (H, W, 3) uint8 BGR query photo.
        database: Prepared museum features (``matching_database``).
        method: Local method settings.
        segment: Remove the background.
        top_k: Ranking depth; None ranks the whole museum.

    Returns:
        (rankings, scores): one museum-id list and one museum id -> score dict per painting.
    """
    result = process_query(image, database, method, segment=segment, top_k=top_k)
    return result["rankings"], result["scores"]


def run(query_dir: Path, corpus_dir: Path, method: LocalMethod, *, segment: bool = True, top_k: int | None = TOP_K,
        unknowns: bool = False, debug: bool = False, progress: Progress | None = None) -> dict:
    """Run the Week 4 method on every image of a query folder (``process_query`` per image).

    The same API is used by ``main.py`` and the inspection notebook.

    Args:
        query_dir: Query dataset folder.
        corpus_dir: Museum (BBDD) folder.
        method: Local method settings.
        segment: Remove the background.
        top_k: Ranking depth; None ranks the whole museum.
        unknowns: Also build the global database and return ``answers`` (ranking or [-1] per painting).
        debug: Keep the full ``process_query`` output per query under ``debug``.
        progress: Optional ``(done, total)`` callback while the global database is computed.

    Returns:
        Dict of query id -> per-painting lists under ``rankings``, ``scores``, ``paintings``, ``decisions``
        and ``answers`` (empty unless ``unknowns``); ``seconds`` (query id -> time), ``n_database`` and,
        with ``debug``, ``debug``.
    """
    database = matching_database(corpus_dir, method)
    global_db = global_database(corpus_dir, progress=progress) if unknowns else None
    result = {key: {} for key in ("rankings", "scores", "answers", "paintings", "decisions", "seconds")}
    if debug:
        result["debug"] = {}
    for path in list_images(query_dir):
        start = time.perf_counter()
        q = get_image_id(path)
        query = process_query(load_image(path), database, method, global_database=global_db,
                              segment=segment, top_k=top_k, debug=debug)
        for key in ("rankings", "scores", "paintings", "decisions"):
            result[key][q] = query[key]
        if unknowns:
            result["answers"][q] = query["answers"]
        if debug:
            result["debug"][q] = query
        result["seconds"][q] = time.perf_counter() - start
    result["n_database"] = len(database)
    return result
