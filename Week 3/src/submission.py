"""Delivery: validate and save the Top-10 result.pkl (2 or 3 levels) and the PNG masks of each query, and check
the whole delivery folder (`validate_delivery`)."""

import pickle
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.data_loading import get_image_id, list_images, load_image, load_mask, resolve_dataset


SUBMISSION_K: int = 10


def validate_results(
    results: list[list[list[int]]],
    n_queries: int,
    valid_db_ids: set[int],
) -> None:
    """Validate rankings without changing query or database ID order.

    Args:
        results: Queries -> paintings (left to right) -> Top-10 database IDs.
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

    for query_index, paintings in enumerate(results):
        if type(paintings) is not list or not 1 <= len(paintings) <= 2:
            raise ValueError(f"Query at index {query_index} must contain one or two painting rankings.")
        for painting_index, ranking in enumerate(paintings):
            location = f"Query {query_index}, painting {painting_index}"
            if type(ranking) is not list or len(ranking) != SUBMISSION_K:
                raise ValueError(f"{location} needs exactly {SUBMISSION_K} IDs in a list.")
            for image_id in ranking:
                if type(image_id) is not int:
                    raise ValueError(f"{location} contains a non-Python-int ID: {image_id!r}.")
                if image_id not in valid_db_ids:
                    raise ValueError(f"{location} contains unknown database ID {image_id}.")
            if len(set(ranking)) != len(ranking):
                raise ValueError(f"{location} contains duplicate database IDs.")


def save_results(
    results: list[list[list[int]]],
    output_path: Path,
    *,
    n_queries: int,
    valid_db_ids: set[int],
) -> None:
    """Validate and pickle rankings, preserving their query order.

    Args:
        results: Queries -> paintings (left to right) -> Top-10 database IDs.
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
    """Validate and write one PNG mask per query next to result.pkl."""
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
                    levels: int = 3) -> None:
    """Validate all predictions, then save Top-10 lists and optional sibling PNGs.

    Outer-list order follows sorted query filenames, exactly as list_images.
    Annotation files are never read. Any failed query prevents export.
    """
    if set(rankings) != set(query_paths):
        raise ValueError("Rankings must cover exactly the query IDs.")
    ordered = sorted(query_paths, key=lambda q: query_paths[q].name)
    invalid = [query_paths[q].name for q in ordered
               if not 1 <= len(rankings[q]) <= 2 or any(len(r) < SUBMISSION_K for r in rankings[q])]
    if invalid:
        raise ValueError(f"Cannot export Top-10; insufficient predictions for: {invalid}")
    results = [[ranking[:SUBMISSION_K] for ranking in rankings[q]] for q in ordered]
    validate_results(results, len(ordered), valid_db_ids)
    if levels == 2:  # one painting per query (cropped QS1): [[ids], ...]
        if any(len(paintings) != 1 for paintings in results):
            raise ValueError("A 2-level export needs exactly one painting per query.")
        results = [paintings[0] for paintings in results]
        validate_results([[r] for r in results], len(ordered), valid_db_ids)
    elif levels != 3:
        raise ValueError("levels must be 2 or 3.")
    if masks is not None:
        save_masks(masks, query_paths, output_path.parent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(results, file)
    with output_path.open("rb") as file:
        restored = pickle.load(file)
    if restored != results:
        raise OSError("Saved submission does not match the computed rankings.")


# ---- Format check of the whole Week 3 delivery folder (main.py validate / predict)

def _check(condition: bool, message: str) -> None:
    """Raise ValueError with the message when the condition is false."""
    if not condition:
        raise ValueError(message)


def _check_ranking(ranking: object, where: str, db_ids: set[int]) -> None:
    """One ranking: SUBMISSION_K distinct native-int BBDD ids."""
    _check(type(ranking) is list and len(ranking) == SUBMISSION_K, f"{where}: need a list of {SUBMISSION_K} ids")
    _check(all(type(x) is int for x in ranking), f"{where}: ids must be native Python int")
    _check(len(set(ranking)) == SUBMISSION_K and set(ranking) <= db_ids, f"{where}: ids must be distinct BBDD ids")


def _load(path: Path) -> object:
    """Read a result.pkl written by this repository."""
    with open(path, "rb") as f:  # written by save_submission of this repository
        return pickle.load(f)


def validate_delivery(delivery: Path, data_dir: Path) -> None:
    """Check the delivery folder against the course format; raises ValueError on the first problem.

      QST1/method1/result.pkl          list of 50 lists holding one list of K=10 distinct native-int BBDD ids
      QST2/method1/result.pkl          list of 30 lists of 1 or 2 such lists (paintings left to right)
      QST2/method1/<query>.png         one per query image, single-channel uint8 0/255, image size
    No other files.
    """
    db_ids = {get_image_id(p) for p in list_images(resolve_dataset(data_dir, "BBDD"))}
    qst1 = list_images(resolve_dataset(data_dir, "qst1_w3"))
    qst2 = list_images(resolve_dataset(data_dir, "qst2_w3"))

    folder = delivery / "QST1" / "method1"
    _check({p.name for p in folder.iterdir()} == {"result.pkl"}, f"{folder}: unexpected files")
    result = _load(folder / "result.pkl")
    _check(type(result) is list and len(result) == len(qst1), f"QST1: need {len(qst1)} query lists")
    for i, paintings in enumerate(result):
        _check(type(paintings) is list and len(paintings) == 1, f"QST1 query {i}: need one painting list (3 levels)")
        _check_ranking(paintings[0], f"QST1 query {i}", db_ids)
    print(f"QST1/method1: result.pkl {len(result)} queries x 1 painting x {SUBMISSION_K} native int ids (3 levels) - OK")

    folder = delivery / "QST2" / "method1"
    expected = {"result.pkl"} | {p.with_suffix(".png").name for p in qst2}
    found = {p.name for p in folder.iterdir()}
    _check(found == expected, f"{folder}: unexpected {sorted(found - expected)}, missing {sorted(expected - found)}")
    result = _load(folder / "result.pkl")
    _check(type(result) is list and len(result) == len(qst2), f"QST2: need {len(qst2)} query lists")
    for i, paintings in enumerate(result):
        _check(type(paintings) is list and 1 <= len(paintings) <= 2, f"QST2 query {i}: need 1 or 2 painting lists")
        for j, ranking in enumerate(paintings):
            _check_ranking(ranking, f"QST2 query {i} painting {j}", db_ids)
    for image in qst2:
        mask = load_mask(folder / image.with_suffix(".png").name)
        _check(mask.dtype == np.uint8 and mask.ndim == 2, f"{image.name}: mask must be single-channel uint8")
        _check(mask.shape == load_image(image).shape[:2], f"{image.name}: mask size differs from the image")
        _check(set(np.unique(mask)) <= {0, 255}, f"{image.name}: mask values must be 0/255")
    n_paintings = sum(len(p) for p in result)
    print(f"QST2/method1: result.pkl {len(result)} queries, {n_paintings} paintings x {SUBMISSION_K} native int ids "
          f"(3 levels), {len(qst2)} masks 0/255 at image size - OK")
