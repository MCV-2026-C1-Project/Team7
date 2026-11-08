"""Retrieval: rank every database descriptor against each query descriptor with a measure of src/similarities.py.

Used by the Week 1-3 global pipeline (src/pipeline.py, one call per detected painting), by the Week 3 experiments
and by the Week 4 global fallback (src/local_pipeline.py).
"""

import warnings

import numpy as np
from numpy.typing import NDArray

from src.similarities import SIMILARITY_METRICS, SimilarityKind


def retrieve(
    query_descriptors: dict[int, NDArray[np.float32]],
    db_descriptors: dict[int, NDArray[np.float32]],
    top_k: int | None = None,
    similarity_function: str = "euclidean",
) -> dict[int, dict[str, list[int] | NDArray[np.float64]]]:
    """Retrieve the top-k most similar database images for each query.

    Args:
        query_descriptors (dict[int, NDArray[np.float32]]): Query ID -> (D,) float32 descriptor.
        db_descriptors (dict[int, NDArray[np.float32]]): Database image ID -> (D,) float32 descriptor.
        top_k (int | None): Number of database images to retrieve per query; None ranks the whole
            database. Values above the database size are clipped with a UserWarning.
        similarity_function (str): Name of the measure, a key of `SIMILARITY_METRICS`.

    Returns:
        dict[int, dict[str, list[int] | NDArray[np.float64]]]: Query ID -> {"top_k_ids": database IDs from
        best to worst, "scores": (top_k,) float64 scores in the same order}. Equal scores keep the
        insertion order of the database IDs (stable sort).

    Raises:
        ValueError: If inputs are empty, top_k is not positive, the similarity function is unknown,
            or the database descriptors do not all have the same shape.
    """
    if not query_descriptors:
        raise ValueError("Query descriptors cannot be empty.")

    if not db_descriptors:
        raise ValueError("Database descriptors cannot be empty.")

    if top_k is None:  # no limit: rank the whole database
        top_k = len(db_descriptors)

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if top_k > len(db_descriptors):
        top_k = len(db_descriptors)
        warnings.warn(
            f"top_k exceeded the number of database images "
            f"({len(db_descriptors)}). Using top_k={top_k} instead.",
            UserWarning,
        )

    metric = SIMILARITY_METRICS.get(similarity_function)

    if metric is None:
        raise ValueError(
            f"Unknown similarity function '{similarity_function}'. "
            f"Available: {list(SIMILARITY_METRICS)}"
        )

    db_ids = list(db_descriptors)

    try:
        db_matrix = np.stack(list(db_descriptors.values()))
    except ValueError as error:
        raise ValueError(
            "Database descriptors must all have the same shape."
        ) from error

    results = {}

    for query_id, query_descriptor in query_descriptors.items():
        scores = metric.function(query_descriptor, db_matrix)

        # Stable sort, so ties keep the database order; similarities are negated to sort best first.
        if metric.kind == SimilarityKind.DISTANCE:
            top_k_indices = np.argsort(scores, kind="stable")[:top_k]
        else:
            top_k_indices = np.argsort(-scores, kind="stable")[:top_k]

        top_k_ids = [
            db_ids[index]
            for index in top_k_indices
        ]

        results[query_id] = {
            "top_k_ids": top_k_ids,
            "scores": scores[top_k_indices],
        }

    return results
