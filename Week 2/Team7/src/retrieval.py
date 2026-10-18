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
        query_descriptors: Query descriptors indexed by image ID.
        db_descriptors: Database descriptors indexed by image ID.
        top_k: Number of database images to retrieve per query.
        similarity_function: Name of the similarity or distance measure.

    Returns:
        Retrieval results indexed by query ID. Equal scores preserve the
        insertion order of database IDs.

    Raises:
        ValueError: If inputs are empty, top_k is invalid, or the
            similarity function is unknown.
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
