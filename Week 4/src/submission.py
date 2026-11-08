"""Writing and validating the delivery ``result.pkl`` files of every week.

A delivery is a list over queries (sorted by filename) of Top-10 museum-id rankings: two list levels
(``[[ids], ...]``, one painting per query, Weeks 1-2) or three (``[[[ids], ...], ...]``, one ranking per painting,
Weeks 3-4); in Week 4 a painting not in the museum is answered ``[-1]``. ``save_submission`` validates and writes
a PKL (and, with background removal, one PNG mask per query next to it); ``validate_delivery`` checks an existing
one. Both are used by ``main.py predict`` / ``main.py validate``. Annotations are never read here.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import TOP_K
from src.data_loading import get_image_id, list_images, load_image, resolve_dataset


SUBMISSION_K: int = TOP_K  # exact length of every delivered ranking
UNKNOWN_RANKING: list[int] = [-1]  # Week 4: the painting is not in the museum


def validate_results(
    results: list[list[int]] | list[list[list[int]]],
    n_queries: int,
    valid_db_ids: set[int],
    *, max_paintings: int = 3, levels: int = 3,
) -> None:
    """Validate rankings without changing query or database ID order.

    Args:
        results: Queries -> paintings -> Top-10 IDs or [-1]; omit the painting level when levels=2.
        n_queries: Expected number of query images.
        valid_db_ids: Actual database image IDs, which need not be sequential.
        max_paintings: Maximum paintings per query (levels=3).
        levels: Number of list levels of ``results``: 2 or 3.

    Raises:
        ValueError: If ``max_paintings``, ``levels`` or the query count is invalid, results are not lists,
            a query has 0 or more than ``max_paintings`` paintings, or a ranking is neither [-1] nor exactly
            10 distinct, valid Python int IDs.
    """
    if type(max_paintings) is not int or max_paintings < 1:
        raise ValueError("max_paintings must be a positive integer.")
    if n_queries <= 0:
        raise ValueError("n_queries must be greater than 0.")
    if type(results) is not list or len(results) != n_queries:
        raise ValueError(f"Results must be a list with {n_queries} query rankings.")
    if levels == 2:
        results = [[ranking] for ranking in results]
    elif levels != 3:
        raise ValueError("Submission format must have 2 or 3 list levels.")

    for query_index, paintings in enumerate(results):
        if type(paintings) is not list or not 1 <= len(paintings) <= max_paintings:
            raise ValueError(f"Query at index {query_index} must contain 1 to {max_paintings} painting rankings.")
        for painting_index, ranking in enumerate(paintings):
            location = f"Query {query_index}, painting {painting_index}"
            # [-1] must hold a Python int (not e.g. -1.0 or np.int64, which also compare equal).
            if ranking == UNKNOWN_RANKING and type(ranking[0]) is int:
                continue
            if type(ranking) is not list or len(ranking) != SUBMISSION_K:
                raise ValueError(f"{location} needs exactly {SUBMISSION_K} IDs in a list.")
            for image_id in ranking:
                if type(image_id) is not int:
                    raise ValueError(f"{location} contains a non-Python-int ID: {image_id!r}.")
                if image_id not in valid_db_ids:
                    raise ValueError(f"{location} contains unknown database ID {image_id}.")
            if len(set(ranking)) != len(ranking):
                raise ValueError(f"{location} contains duplicate database IDs.")


def validate_masks(masks: dict[int, NDArray[np.uint8]], query_paths: dict[int, Path]) -> None:
    """Require one native-size binary uint8 mask per query.

    Args:
        masks: Query id -> (H, W) uint8 mask with values 0 / 255, at the query image's resolution.
        query_paths: Query id -> image path (the image is loaded to check its size).

    Raises:
        ValueError: If the ids differ, or a mask has the wrong dtype, shape or non-binary values.
    """
    if set(masks) != set(query_paths):
        raise ValueError("Masks must cover exactly the query IDs.")
    for q, path in query_paths.items():
        mask = masks[q]
        if mask.dtype != np.uint8 or mask.shape != load_image(path).shape[:2]:
            raise ValueError(f"Mask dimensions or dtype are invalid for {path.name}")
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError(f"Mask must be binary (0/255): {path.name}")


def save_masks(masks: dict[int, NDArray[np.uint8]], query_paths: dict[int, Path], directory: Path) -> None:
    """Validate and write one PNG mask per query next to result.pkl.

    Args:
        masks: Query id -> (H, W) uint8 0 / 255 mask.
        query_paths: Query id -> image path; each PNG takes the image's file name.
        directory: Output folder (created if needed).

    Raises:
        ValueError: If the masks are invalid (see ``validate_masks``).
        OSError: If a mask cannot be encoded as PNG.
    """
    validate_masks(masks, query_paths)
    directory.mkdir(parents=True, exist_ok=True)
    for q, path in query_paths.items():
        target = directory / path.with_suffix(".png").name
        ok, encoded = cv2.imencode(".png", masks[q])  # imencode + tofile: also works with non-ASCII paths
        if not ok:
            raise OSError(f"Could not encode mask: {target}")
        encoded.tofile(str(target))


def save_submission(rankings: dict[int, list[list[int]]], query_paths: dict[int, Path], output_path: Path, *,
                    valid_db_ids: set[int], masks: dict[int, NDArray[np.uint8]] | None = None,
                    max_paintings: int = 3, levels: int = 3) -> None:
    """Validate all predictions, then save Top-10 lists and optional sibling PNGs.

    Outer-list order follows sorted query filenames, exactly as list_images.
    Annotation files are never read. Any failed query prevents export.

    Args:
        rankings: Query id -> per-painting rankings (longer rankings are cut to 10; [-1] for unknown).
        query_paths: Query id -> image path.
        output_path: Destination ``result.pkl`` (parent folders are created).
        valid_db_ids: Actual museum image ids.
        masks: Optional query id -> (H, W) uint8 0 / 255 mask, written as PNGs next to the PKL.
        max_paintings: Maximum paintings per query.
        levels: 3 (one ranking per painting) or 2 (flat; requires exactly one painting per query).

    Raises:
        ValueError: If the ids differ, a query has too few predictions or paintings, or validation fails.
        OSError: If a mask cannot be encoded or the PKL read back differs from the computed rankings.
    """
    if set(rankings) != set(query_paths):
        raise ValueError("Rankings must cover exactly the query IDs.")
    ordered = sorted(query_paths, key=lambda q: query_paths[q].name)
    invalid = [query_paths[q].name for q in ordered
               if not 1 <= len(rankings[q]) <= max_paintings
               or any(len(r) < SUBMISSION_K and r != UNKNOWN_RANKING for r in rankings[q])]
    if invalid:
        raise ValueError(f"Cannot export Top-10; insufficient predictions for: {invalid}")
    results = [[ranking[:SUBMISSION_K] for ranking in rankings[q]] for q in ordered]
    # Validate the three-level form first, then again in the exported form.
    validate_results(results, len(ordered), valid_db_ids, max_paintings=max_paintings)
    if levels == 2:
        if any(len(paintings) != 1 for paintings in results):
            raise ValueError("Two-level export requires exactly one painting per query; use three levels.")
        results = [paintings[0] for paintings in results]
    validate_results(results, len(ordered), valid_db_ids, max_paintings=max_paintings, levels=levels)
    if masks is not None:
        save_masks(masks, query_paths, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(results, file)
    with output_path.open("rb") as file:  # read back: the delivered file must be exactly what was computed
        restored = pickle.load(file)
    if restored != results:
        raise OSError("Saved submission does not match the computed rankings.")


def validate_delivery(delivery: Path, data_dir: Path, *, query_set: str = "qst1_w4",
                      max_paintings: int = 3, levels: int = 3, method: str = "method1") -> dict[str, int]:
    """Validate an existing PKL against its query count and actual museum IDs.

    Annotations are never read and unknown rankings [-1] are accepted.

    Args:
        delivery: A ``.pkl`` path, or a root containing ``QS*/method*/result.pkl``.
        data_dir: Folder containing BBDD and the query datasets.
        query_set: Query dataset name (e.g. "qst1_w4"); gives the query count and the ``QST1`` folder.
        max_paintings: Maximum paintings per query.
        levels: Expected number of list levels: 2 or 3.
        method: Method folder name when ``delivery`` is a root.

    Returns:
        ``{"queries", "paintings", "unknown", "levels"}``: counts of queries, paintings and unknown
        paintings, and the number of levels.

    Raises:
        ValueError: If the PKL does not pass ``validate_results``.
    """
    path = delivery if delivery.suffix == ".pkl" else delivery / query_set.split("_")[0].upper() / method / "result.pkl"
    query_paths = list_images(resolve_dataset(data_dir, query_set))
    database_ids = {get_image_id(p) for p in list_images(resolve_dataset(data_dir, "BBDD"))}
    with path.open("rb") as file:  # PKL is the course delivery format; only our own result.pkl files are read
        results = pickle.load(file)
    validate_results(results, len(query_paths), database_ids, max_paintings=max_paintings, levels=levels)
    if levels == 2:
        results = [[ranking] for ranking in results]
    paintings = sum(len(p) for p in results)
    unknown = sum(r == UNKNOWN_RANKING for p in results for r in p)
    return {"queries": len(results), "paintings": paintings, "unknown": unknown, "levels": levels}
