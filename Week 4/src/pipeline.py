"""Global-descriptor retrieval pipeline of the submitted Week 1-3 methods; Week 4 reuses its database helpers.

`run_pipeline` (main.py for weeks 1-3) builds the database descriptors, optionally removes the background and
splits each query into paintings (Week 2: one painting, Week 3: up to two, Week 4 segmentation: up to three), and
ranks the database for every painting with src.descriptors.compute_descriptor and src.retrieval.retrieve.
src/local_pipeline.py (Week 4) reuses `build_database` for its global fallback. No annotations are read and no
files are written here. Each detected painting has its own ranking, in reading order (Week 3: left to right;
Week 4 ordering: rows top to bottom, then left to right). Failed slots keep an empty ranking.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.config import GLOBAL_METHOD, PRESETS, DescriptorConfig
from src.data_loading import get_image_id, list_images, load_image
from src.descriptors import compute_descriptor, crop_to_mask
from src.retrieval import retrieve
from src.segmentation import Painting, order_paintings, remove_background

Progress = Callable[[int, int], None]  # progress(done, total), e.g. to print database progress


class Predictions(TypedDict):
    """Everything run_pipeline returns: paths, database, masks, rankings, metadata.

    Attributes:
        query_paths (dict[int, Path]): Query ID -> image path.
        database (dict[int, NDArray[np.float32]]): Database image ID -> (D,) float32 descriptor.
        masks (dict[int, NDArray[np.uint8]]): Query ID -> (H, W) uint8 0/255 union mask (empty without segmentation).
        paintings (dict[int, list[Painting]]): Query ID -> detected paintings (empty without segmentation).
        rankings (dict[int, list[list[int]]]): Query ID -> one ranking of database IDs per painting, in reading
            order; an empty list for a failed painting.
        statuses (dict[int, list[str]]): Query ID -> per-painting status ("ok", "empty_mask",
            "image_smaller_than_grid"), aligned with `rankings`.
        metadata (dict[str, object]): Run settings and counts (method, descriptor, similarity, segmentation,
            dataset folders, query order, failed query IDs).
    """
    query_paths: dict[int, Path]
    database: dict[int, NDArray[np.float32]]
    masks: dict[int, NDArray[np.uint8]]
    paintings: dict[int, list[Painting]]
    rankings: dict[int, list[list[int]]]
    statuses: dict[int, list[str]]
    metadata: dict[str, object]


def index_images(folder: Path) -> dict[int, Path]:
    """Map image ID -> path for the images of a folder.

    Args:
        folder (Path): Dataset folder.

    Returns:
        dict[int, Path]: Image ID (src.data_loading.get_image_id) -> path, in src.data_loading.list_images order.

    Raises:
        ValueError: If two images have the same ID.
    """
    paths = list_images(folder)
    indexed = {get_image_id(p): p for p in paths}
    if len(indexed) != len(paths):
        raise ValueError(f"Duplicate image IDs in {folder}")
    return indexed


def build_database(corpus_dir: Path, config: DescriptorConfig, *, progress: Progress | None = None
                   ) -> dict[int, NDArray[np.float32]]:
    """Compute the descriptor of every database image (whole image, no mask).

    Args:
        corpus_dir (Path): Database (BBDD) folder.
        config (DescriptorConfig): Descriptor configuration (src/config.py).
        progress (Progress | None): Called as progress(done, total) every 100 images and at the end.

    Returns:
        dict[int, NDArray[np.float32]]: Database image ID -> (D,) float32 descriptor.

    Raises:
        ValueError: If two database images have the same ID.
    """
    paths = index_images(corpus_dir)
    database: dict[int, NDArray[np.float32]] = {}
    for index, (image_id, path) in enumerate(paths.items(), 1):
        database[image_id] = compute_descriptor(load_image(path), config)
        if progress is not None and (index % 100 == 0 or index == len(paths)):
            progress(index, len(paths))
    return database


def segment_query(image: NDArray[np.uint8], *, max_paintings: int = 3, week: int = 4
                  ) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Union mask and painting masks of a query (src.segmentation.remove_background).

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR query photo.
        max_paintings (int): Maximum number of paintings kept.
        week (int): Week whose submitted background-removal method is used.

    Returns:
        tuple[NDArray[np.uint8], list[Painting]]: The (H, W) uint8 0/255 union mask and the detected paintings.
    """
    return remove_background(image, max_paintings=max_paintings, week=week)


def rank_paintings(image: NDArray[np.uint8], paintings: list[Painting],
                   database: dict[int, NDArray[np.float32]], config: DescriptorConfig,
                   similarity: str, *, top_k: int | None = None, week: int = 4,
                   ) -> tuple[list[list[int]], list[str]]:
    """Rank the database separately for each painting of one query, in reading order.

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR query photo.
        paintings (list[Painting]): Detected paintings; each mask is (H, W) uint8 0/255.
        database (dict[int, NDArray[np.float32]]): Database image ID -> (D,) float32 descriptor.
        config (DescriptorConfig): Descriptor configuration, the same one used for the database.
        similarity (str): Name of the measure in src.similarities.SIMILARITY_METRICS.
        top_k (int | None): Ranking depth; None ranks the whole database.
        week (int): 3 orders paintings left to right by their horizontal centre (the Week 3 submission);
            any other value uses src.segmentation.order_paintings.

    Returns:
        tuple[list[list[int]], list[str]]: One ranking of database IDs per painting (empty when the
        painting failed) and one status per painting ("ok", "empty_mask", "image_smaller_than_grid").

    Raises:
        ValueError: If a painting mask does not have the image's height and width.
    """
    rankings: list[list[int]] = []
    statuses: list[str] = []
    # Side of the finest block grid: a crop with fewer pixels than this would leave some blocks empty.
    n = 2 ** (config["pyramid_levels"] - 1) if config.get("pyramid_levels") else config.get("grid", 1)
    ordered = (sorted(paintings, key=lambda p: p["bbox"][0] + p["bbox"][2])  # x0 + x1: horizontal centre
               if week == 3 else order_paintings(paintings))
    for painting in ordered:
        mask = painting["mask"]
        if mask.shape != image.shape[:2]:
            raise ValueError("Painting masks must match the original image shape.")
        cropped, _ = crop_to_mask(image, mask > 0)
        status = "ok"
        if not np.any(mask):
            status = "empty_mask"
        elif min(cropped.shape[:2]) < n:
            status = "image_smaller_than_grid"
        ranking: list[int] = []
        if status == "ok":
            descriptor = compute_descriptor(image, config, mask)
            result = retrieve({0: descriptor}, database, similarity_function=similarity, top_k=top_k)
            ranking = result[0]["top_k_ids"]
        rankings.append(ranking)
        statuses.append(status)
    return rankings, statuses


def rank_queries(query_paths: dict[int, Path], database: dict[int, NDArray[np.float32]],
                 config: DescriptorConfig, similarity: str, *,
                 paintings: dict[int, list[Painting]] | None = None, top_k: int | None = None, week: int = 4,
                 ) -> tuple[dict[int, list[list[int]]], dict[int, list[str]]]:
    """Rank every query: each painting mask, or the whole image for already cropped queries.

    Args:
        query_paths (dict[int, Path]): Query ID -> image path.
        database (dict[int, NDArray[np.float32]]): Database image ID -> (D,) float32 descriptor.
        config (DescriptorConfig): Descriptor configuration, the same one used for the database.
        similarity (str): Name of the measure in src.similarities.SIMILARITY_METRICS.
        paintings (dict[int, list[Painting]] | None): Query ID -> paintings to rank; None treats every query
            as one painting covering the whole image (cropped QS1 queries).
        top_k (int | None): Ranking depth; None ranks the whole database.
        week (int): Painting ordering, see `rank_paintings`.

    Returns:
        tuple[dict[int, list[list[int]]], dict[int, list[str]]]: Query ID -> per-painting rankings, and
        query ID -> per-painting statuses.

    Raises:
        ValueError: If a painting mask does not match its image size.
    """
    rankings, statuses = {}, {}
    for query_id, path in query_paths.items():
        image = load_image(path)
        if paintings is None:
            height, width = image.shape[:2]
            found: list[Painting] = [{"mask": np.full((height, width), 255, np.uint8),
                                      "bbox": (0, 0, width, height)}]
        else:
            found = paintings[query_id]
        rankings[query_id], statuses[query_id] = rank_paintings(
            image, found, database, config, similarity, top_k=top_k, week=week)
    return rankings, statuses


def run_pipeline(corpus_dir: Path, query_dir: Path, *, segment: bool,
                 method: str | None = None,
                 descriptor: DescriptorConfig | None = None, similarity: str | None = None,
                 top_k: int | None = None, max_paintings: int = 3, week: int = 4,
                 progress: Progress | None = None) -> Predictions:
    """Predict at native resolution, independently of evaluation or export.

    Args:
        corpus_dir (Path): Database (BBDD) folder.
        query_dir (Path): Query set folder.
        segment (bool): Must be explicit: False for cropped QS1 images, True for QS2 images with background.
        method (str | None): Preset name in src.config.PRESETS; None selects the global fallback descriptor
            (GLOBAL_METHOD). Also stored in the metadata.
        descriptor (DescriptorConfig | None): Overrides the preset's descriptor.
        similarity (str | None): Overrides the preset's similarity measure.
        top_k (int | None): Ranking depth; None ranks the whole database.
        max_paintings (int): Maximum number of paintings per query when segmenting.
        week (int): Week whose submitted background removal and painting order are used (default 4, the
            current method).
        progress (Progress | None): Database progress callback, see `build_database`.

    Returns:
        Predictions: Paths, database, masks, paintings, rankings, statuses and metadata of the run.

    Raises:
        KeyError: If `method` is not a preset and `descriptor` or `similarity` is not given.
        ValueError: If image IDs are duplicated or a painting mask does not match its image.
    """
    method = method or GLOBAL_METHOD
    selected = PRESETS.get(method, {})
    config = descriptor if descriptor is not None else selected["descriptor"]
    similarity = similarity or selected["similarity"]
    queries = index_images(query_dir)
    database = build_database(corpus_dir, config, progress=progress)
    masks: dict[int, NDArray[np.uint8]] = {}
    paintings: dict[int, list[Painting]] = {}
    if segment:
        for query_id, path in queries.items():
            masks[query_id], paintings[query_id] = segment_query(load_image(path), max_paintings=max_paintings,
                                                                 week=week)
    rankings, statuses = rank_queries(queries, database, config, similarity,
                                      paintings=paintings if segment else None, top_k=top_k, week=week)
    metadata = {
        "method": method, "descriptor": config, "similarity": similarity,
        "segment": segment, "segmentation": {"week": week} if segment else {}, "resize": None,
        "max_paintings": max_paintings,
        "corpus_dir": str(corpus_dir.resolve()), "query_dir": str(query_dir.resolve()),
        "n_database": len(database), "n_queries": len(queries),
        "query_order": [p.name for p in queries.values()],
        # A query fails when no painting was found or any of its paintings could not be described.
        "failed_query_ids": [q for q, states in statuses.items() if not states or any(s != "ok" for s in states)],
    }
    return {"query_paths": queries, "database": database, "masks": masks, "paintings": paintings,
            "rankings": rankings, "statuses": statuses, "metadata": metadata}
