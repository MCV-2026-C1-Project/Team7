"""Validate and save Top-10 submission rankings."""

import pickle
from pathlib import Path


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
        ValueError: If results fail the submission validation.
        OSError: If the output directory or file cannot be written.
    """
    validate_results(results, n_queries, valid_db_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(results, file)
