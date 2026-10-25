"""Inference (no annotations read, no files written): images -> painting masks (QS2) -> descriptors -> rankings.

No annotations are read and no files are written here. Each detected painting
has its own ranking, ordered left to right. Failed slots keep an empty ranking.
"""

from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.config import FINAL_METHOD, PRESETS, DescriptorConfig
from src.data_loading import get_image_id, list_images, load_image
from src.descriptors import compute_descriptor, crop_to_mask
from src.retrieval import retrieve
from src.segmentation import Painting, remove_background


class Predictions(TypedDict):
    """Everything run_pipeline returns: paths, database, masks, rankings, metadata."""
    query_paths: dict[int, Path]
    database: dict[int, NDArray[np.float32]]
    masks: dict[int, NDArray[np.uint8]]
    paintings: dict[int, list[Painting]]
    rankings: dict[int, list[list[int]]]
    statuses: dict[int, list[str]]
    metadata: dict[str, object]


def index_images(folder: Path) -> dict[int, Path]:
    """Image id -> path of a folder (ids must be unique)."""
    paths = list_images(folder)
    indexed = {get_image_id(p): p for p in paths}
    if len(indexed) != len(paths):
        raise ValueError(f"Duplicate image IDs in {folder}")
    return indexed


def build_database(corpus_dir: Path, config: DescriptorConfig) -> dict[int, NDArray[np.float32]]:
    """Descriptor of every database image."""
    paths = index_images(corpus_dir)
    database: dict[int, NDArray[np.float32]] = {}
    for index, (image_id, path) in enumerate(paths.items(), 1):
        database[image_id] = compute_descriptor(load_image(path), config)
        if index % 100 == 0 or index == len(paths):
            print(f"Database descriptors: {index}/{len(paths)}", flush=True)
    return database


def segment_query(image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Union mask and painting masks of a query (src/segmentation.py)."""
    return remove_background(image)


def rank_paintings(image: NDArray[np.uint8], paintings: list[Painting],
                   database: dict[int, NDArray[np.float32]], config: DescriptorConfig,
                   similarity: str, *, top_k: int | None = None,
                   ) -> tuple[list[list[int]], list[str]]:
    """Apply the existing descriptor separately to each painting mask."""
    rankings: list[list[int]] = []
    statuses: list[str] = []
    n = 2 ** (config["pyramid_levels"] - 1) if config.get("pyramid_levels") else config.get("grid", 1)
    ordered = sorted(paintings, key=lambda painting: (painting["bbox"][0] + painting["bbox"][2]) / 2)
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
                 paintings: dict[int, list[Painting]] | None = None, top_k: int | None = None,
                 ) -> tuple[dict[int, list[list[int]]], dict[int, list[str]]]:
    """Rank individual masks, or the whole image for already cropped queries."""
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
            image, found, database, config, similarity, top_k=top_k)
    return rankings, statuses


def run_pipeline(corpus_dir: Path, query_dir: Path, *, segment: bool,
                 method: str | None = None,
                 descriptor: DescriptorConfig | None = None, similarity: str | None = None,
                 top_k: int | None = None) -> Predictions:
    """Predict at native resolution, independently of evaluation or export.

    segment must be explicit: False for cropped QS1 images, True for QS2 images
    with background. Omitting method selects the final method (FINAL_METHOD);
    descriptor / similarity override the preset's. Background removal uses
    the fixed robust method, independently of the descriptor.
    """
    method = method or FINAL_METHOD
    selected = PRESETS.get(method, {})
    config = descriptor if descriptor is not None else selected["descriptor"]
    similarity = similarity or selected["similarity"]
    queries = index_images(query_dir)
    database = build_database(corpus_dir, config)
    masks: dict[int, NDArray[np.uint8]] = {}
    paintings: dict[int, list[Painting]] = {}
    if segment:
        for query_id, path in queries.items():
            masks[query_id], paintings[query_id] = segment_query(load_image(path))
    rankings, statuses = rank_queries(queries, database, config, similarity,
                                      paintings=paintings if segment else None, top_k=top_k)
    metadata = {
        "method": method, "descriptor": config, "similarity": similarity,
        "segment": segment, "segmentation": {"method": "robust"} if segment else {}, "resize": None,
        "corpus_dir": str(corpus_dir.resolve()), "query_dir": str(query_dir.resolve()),
        "n_database": len(database), "n_queries": len(queries),
        "query_order": [p.name for p in queries.values()],
        "failed_query_ids": [q for q, states in statuses.items() if not states or any(s != "ok" for s in states)],
    }
    return {"query_paths": queries, "database": database, "masks": masks, "paintings": paintings,
            "rankings": rankings, "statuses": statuses, "metadata": metadata}
