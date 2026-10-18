import warnings

import numpy as np
from numpy.typing import NDArray


# Adapted from Ben Hamner's Metrics implementation:
# https://github.com/benhamner/Metrics/tree/master/Python/ml_metrics/average_precision.py
def apk(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute average precision at K for one query.

    Args:
        actual: Relevant database IDs; their order does not matter.
        predicted: Database IDs ordered from best to worst match.
        k: Maximum number of predictions to consider.

    Returns:
        Average precision at K, or zero if there are no relevant IDs.
        Repeated predictions count as hits only on their first occurrence.

    Raises:
        ValueError: If k is not positive.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(
    actual: list[list[int]],
    predicted: list[list[int]],
    k: int = 10,
) -> float:
    """Compute mean average precision at K across queries.

    Args:
        actual: Relevant database IDs for each query.
        predicted: Ranked database IDs in the same query order as actual.
        k: Maximum number of predictions to consider per query.

    Returns:
        Mean of the per-query average precision scores.

    Raises:
        ValueError: If k is not positive, the query counts differ,
            or there are no queries.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted must contain the same number of queries."
        )

    if not actual:
        raise ValueError("Cannot compute mAP with no queries.")

    return float(np.mean([apk(a, p, k) for a, p in zip(actual, predicted)]))


def evaluate_retrieval(
    results: dict[int, dict[str, list[int] | NDArray[np.float64]]],
    ground_truth_mapping: dict[int, list[int]],
    top_k: int | None = None,
) -> float:
    """Evaluate retrieval results using Mean Average Precision at K.

    Args:
        results: Retrieval results indexed by query ID.
        ground_truth_mapping: List of matching database IDs for each query ID.
        top_k: Number of retrieved images to consider.

    Returns:
        Mean Average Precision at K.

    Raises:
        ValueError: If query IDs differ, top_k is not positive, there are no
            queries, or a query has fewer than top_k predictions.
    """
    expected_query_ids = set(ground_truth_mapping)
    retrieved_query_ids = set(results)

    if expected_query_ids != retrieved_query_ids:
        missing = expected_query_ids - retrieved_query_ids
        unexpected = retrieved_query_ids - expected_query_ids

        raise ValueError(
            "Retrieval query IDs do not match ground truth. "
            f"Missing: {sorted(missing)}, "
            f"unexpected: {sorted(unexpected)}"
        )

    shortest = min(len(result["top_k_ids"]) for result in results.values())
    if top_k is None:  # no limit: use every retrieved image
        top_k = shortest

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if top_k > shortest:
        warnings.warn(
            f"top_k={top_k} exceeds the number of retrieved images ({shortest}). Using top_k={shortest} instead.",
            UserWarning,
        )
        top_k = shortest

    query_ids = list(ground_truth_mapping)

    actual = [
        ground_truth_mapping[query_id]
        for query_id in query_ids
    ]

    predicted = [
        results[query_id]["top_k_ids"][:top_k]
        for query_id in query_ids
    ]

    return mapk(actual, predicted, k=top_k)
