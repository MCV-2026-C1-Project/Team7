"""Validate and save Top-10 submission rankings."""

import pickle
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.data_loading import load_image


SUBMISSION_K: int = 10


def validate_results(
    results: list[list[int]],
    n_queries: int,
    valid_db_ids: set[int],
) -> None:
    """Validate rankings without changing query or database ID order.

    Args:
        results: Ranked database IDs in the required query order.
        n_queries: Expected number of query images.
        valid_db_ids: Actual database image IDs, which need not be sequential.

    Raises:
        ValueError: If the query count is invalid, results are not lists, or a
            ranking does not contain exactly 10 distinct, valid Python int IDs.
    """
    if n_queries <= 0:
        raise ValueError("n_queries must be greater than 0.")
    if type(results) is not list or len(results) != n_queries:
        raise ValueError(f"Results must be a list with {n_queries} query rankings.")

    for query_index, ranking in enumerate(results):
        if type(ranking) is not list:
            raise ValueError(
                f"Query at index {query_index} must be a list."
            )
        if len(ranking) != SUBMISSION_K:
            raise ValueError(
                f"Query at index {query_index} has {len(ranking)} IDs; "
                f"the submission needs exactly {SUBMISSION_K}."
            )
        for image_id in ranking:
            if type(image_id) is not int:
                raise ValueError(
                    f"Query at index {query_index} contains a non-Python-int ID: "
                    f"{image_id!r}."
                )
            if image_id not in valid_db_ids:
                raise ValueError(
                    f"Query at index {query_index} contains unknown database ID {image_id}."
                )
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"Query at index {query_index} contains duplicate database IDs.")


def save_results(
    results: list[list[int]],
    output_path: Path,
    *,
    n_queries: int,
    valid_db_ids: set[int],
) -> None:
    """Validate and pickle rankings, preserving their query order.

    Args:
        results: Ranked database IDs in the required query order.
        output_path: Destination pickle file; parent directories are created.
        n_queries: Expected number of query images.
        valid_db_ids: Actual database image IDs.

    Raises:
        ValueError: If results fail submission validation.
        OSError: If the output directory or file cannot be written.
    """
    validate_results(results, n_queries, valid_db_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(results, file)


def validate_masks(masks: dict[int, NDArray[np.uint8]], query_paths: dict[int, Path]) -> None:
    """Require one native-size binary uint8 mask per query."""
    if set(masks) != set(query_paths):
        raise ValueError("Masks must cover exactly the query IDs.")
    for q, path in query_paths.items():
        mask = masks[q]
        if mask.dtype != np.uint8 or mask.shape != load_image(path).shape[:2]:
            raise ValueError(f"Mask dimensions or dtype are invalid for {path.name}")
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError(f"Mask must be binary (0/255): {path.name}")


def save_masks(masks: dict[int, NDArray[np.uint8]], query_paths: dict[int, Path], directory: Path) -> None:
    validate_masks(masks, query_paths)
    directory.mkdir(parents=True, exist_ok=True)
    for q, path in query_paths.items():
        target = directory / path.with_suffix(".png").name
        if not cv2.imwrite(str(target), masks[q]):
            raise OSError(f"Could not write mask: {target}")


def save_submission(rankings: dict[int, list[int]], query_paths: dict[int, Path], output_path: Path, *,
                    valid_db_ids: set[int], masks: dict[int, NDArray[np.uint8]] | None = None) -> None:
    """Validate all predictions, then save Top-10 lists and optional sibling PNGs.

    Outer-list order follows sorted query filenames, exactly as list_images.
    Annotation files are never read. Any failed query prevents export.
    """
    if set(rankings) != set(query_paths):
        raise ValueError("Rankings must cover exactly the query IDs.")
    ordered = sorted(query_paths, key=lambda q: query_paths[q].name)
    invalid = [query_paths[q].name for q in ordered if len(rankings[q]) < SUBMISSION_K]
    if invalid:
        raise ValueError(f"Cannot export Top-10; insufficient predictions for: {invalid}")
    results = [rankings[q][:SUBMISSION_K] for q in ordered]
    validate_results(results, len(ordered), valid_db_ids)
    if masks is not None:
        save_masks(masks, query_paths, output_path.parent)
    save_results(results, output_path, n_queries=len(ordered), valid_db_ids=valid_db_ids)
    with output_path.open("rb") as file:
        restored = pickle.load(file)
    validate_results(restored, len(ordered), valid_db_ids)
    if restored != results:
        raise OSError("Saved submission does not match the computed rankings.")
