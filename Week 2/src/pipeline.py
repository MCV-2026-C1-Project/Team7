"""Inference only: images -> optional color masks -> descriptors -> rankings.

No annotations are read and no files are written here. One painting per query
is assumed. Empty or tiny masks produce an explicit failure and empty ranking.
"""

import inspect
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.config import DEFAULT_WEEK2_METHOD, DescriptorConfig, METHODS, WEEK2_METHODS
from src.data_loading import get_image_id, list_images, load_image
from src.descriptors import compute_descriptor, crop_to_mask
from src.retrieval import retrieve
from src.segmentation import remove_background


class Predictions(TypedDict):
    query_paths: dict[int, Path]
    database: dict[int, NDArray[np.float32]]
    masks: dict[int, NDArray[np.uint8]]
    rankings: dict[int, list[int]]
    statuses: dict[int, str]
    metadata: dict[str, object]


def index_images(folder: Path) -> dict[int, Path]:
    paths = list_images(folder)
    indexed = {get_image_id(p): p for p in paths}
    if len(indexed) != len(paths):
        raise ValueError(f"Duplicate image IDs in {folder}")
    return indexed


def rank_queries(query_paths: dict[int, Path], database: dict[int, NDArray[np.float32]],
                 config: DescriptorConfig, similarity: str, *,
                 masks: dict[int, NDArray[np.uint8]] | None = None, top_k: int | None = None,
                 ) -> tuple[dict[int, list[int]], dict[int, str]]:
    """Rank each query using an optional explicit mask; keep failures in output."""
    rankings, statuses = {}, {}
    n = 2 ** (config["pyramid_levels"] - 1) if config.get("pyramid_levels") else config.get("grid", 1)
    for query_id, path in query_paths.items():
        image = load_image(path)
        mask = masks[query_id] if masks is not None else None
        cropped, _ = crop_to_mask(image, None if mask is None else mask > 0)
        status = "ok"
        if mask is not None and not np.any(mask):
            status = "empty_mask"
        elif min(cropped.shape[:2]) < n:
            status = "image_smaller_than_grid"
        rankings[query_id] = []
        if status == "ok":
            descriptor = compute_descriptor(image, config, mask)
            result = retrieve({query_id: descriptor}, database, similarity_function=similarity, top_k=top_k)
            rankings[query_id] = result[query_id]["top_k_ids"]
        statuses[query_id] = status
    return rankings, statuses


def run_pipeline(corpus_dir: Path, query_dir: Path, *, segment: bool,
                 method: str | None = None,
                 segmentation_options: dict[str, float | str | bool] | None = None,
                 descriptor: DescriptorConfig | None = None, similarity: str | None = None,
                 top_k: int | None = None) -> Predictions:
    """Predict at native resolution, independently of evaluation or export.

    segment must be explicit: False for cropped QS1 images, True for QS2 images
    with background. Omitting method selects Week 2 delivery method1 for both
    modes; use method="method2" for delivery method2. Segmentation uses
    the same defaults regardless of the descriptor.
    """
    if segmentation_options and not segment:
        raise ValueError("Segmentation options require segment=True.")
    method = method or DEFAULT_WEEK2_METHOD
    if method in METHODS:
        old = METHODS[method]
        selected = {"descriptor": {"kind": "1d", "grid": 1, **{k: v for k, v in old.items() if k != "similarity"}},
                    "similarity": old["similarity"]}
    else:
        selected = WEEK2_METHODS.get(method, {})
    config = descriptor if descriptor is not None else selected["descriptor"]
    similarity = similarity or selected["similarity"]
    db_paths, queries = index_images(corpus_dir), index_images(query_dir)
    parameters = inspect.signature(remove_background).parameters
    segmentation = {name: p.default for name, p in parameters.items() if name != "image"} if segment else {}
    segmentation.update(segmentation_options or {})
    database = {}
    for index, (image_id, path) in enumerate(db_paths.items(), 1):
        database[image_id] = compute_descriptor(load_image(path), config)
        if index % 100 == 0 or index == len(db_paths):
            print(f"Database descriptors: {index}/{len(db_paths)}", flush=True)
    masks = {q: remove_background(load_image(p), **segmentation) for q, p in queries.items()} if segment else {}
    rankings, statuses = rank_queries(queries, database, config, similarity,
                                      masks=masks if segment else None, top_k=top_k)
    metadata = {
        "method": method, "descriptor": config, "similarity": similarity,
        "segment": segment, "segmentation": segmentation, "resize": None,
        "corpus_dir": str(corpus_dir.resolve()), "query_dir": str(query_dir.resolve()),
        "n_database": len(database), "n_queries": len(queries),
        "query_order": [p.name for p in queries.values()],
        "failed_query_ids": [q for q, status in statuses.items() if status != "ok"],
    }
    return {"query_paths": queries, "database": database, "masks": masks,
            "rankings": rankings, "statuses": statuses, "metadata": metadata}
