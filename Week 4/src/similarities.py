"""Vectorised distance and similarity measures between one query descriptor and a batch of database descriptors.

Week 1 compares these measures on the global colour histograms (the submitted Week 1 methods use the Hellinger
kernel and chi-squared); the Week 2-3 methods and the Week 4 global fallback use L1. src/retrieval.py looks
measures up by name in `SIMILARITY_METRICS` and uses their kind to sort the scores.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class SimilarityKind(Enum):
    """Whether a measure ranks matches ascending (distance) or descending (similarity)."""

    DISTANCE = "distance"
    SIMILARITY = "similarity"


@dataclass(frozen=True)
class SimilarityMetric:
    """A named distance or similarity measure and how to rank its scores.

    Attributes:
        name (str): Name used as the lookup key in `SIMILARITY_METRICS`.
        function (Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]): Vectorised callable taking a (D,)
            query and an (N, D) batch and returning (N,) float64 scores.
        kind (SimilarityKind): Whether lower scores are better (``DISTANCE``) or higher (``SIMILARITY``).
    """

    name: str
    function: Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]
    kind: SimilarityKind


def _validate_inputs(
    query: ArrayLike,
    descriptors: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a 1D float query and a 2D float batch (a 1D batch becomes one row); raise if sizes or values are bad."""
    query_array = np.asarray(query, dtype=float)
    descriptor_array = np.asarray(descriptors, dtype=float)

    if query_array.ndim != 1:
        raise ValueError("query must be a 1D descriptor")
    if descriptor_array.ndim == 1:
        descriptor_array = descriptor_array[None, :]
    if descriptor_array.ndim != 2 or descriptor_array.shape[1] != query_array.size:
        raise ValueError("descriptors must be 2D with the same size as query")
    if query_array.size == 0 or descriptor_array.shape[0] == 0:
        raise ValueError("query and descriptors cannot be empty")
    if not np.isfinite(query_array).all() or not np.isfinite(descriptor_array).all():
        raise ValueError("descriptors must contain only finite values")
    return query_array, descriptor_array


def _validate_histograms(
    query: ArrayLike,
    descriptors: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """`_validate_inputs`, plus a check that the histograms are non-negative."""
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    if (query_array < 0).any() or (descriptor_array < 0).any():
        raise ValueError("histogram descriptors must be non-negative")
    return query_array, descriptor_array


def euclidean(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the L2 distance; lower values indicate greater similarity.

    Args:
        query (ArrayLike): (D,) query descriptor.
        descriptors (ArrayLike): (N, D) database descriptors (a single (D,) descriptor is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 distances.

    Raises:
        ValueError: If the shapes do not match, an input is empty or a value is not finite.
    """
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    return np.linalg.norm(descriptor_array - query_array, axis=1)


def l1(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the L1 distance; lower values indicate greater similarity.

    Args:
        query (ArrayLike): (D,) query descriptor.
        descriptors (ArrayLike): (N, D) database descriptors (a single (D,) descriptor is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 distances.

    Raises:
        ValueError: If the shapes do not match, an input is empty or a value is not finite.
    """
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    return np.abs(descriptor_array - query_array).sum(axis=1)


def chi2(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the chi-squared distance, sum((h - q)^2 / (h + q)); lower values indicate greater similarity.

    Args:
        query (ArrayLike): (D,) non-negative query histogram.
        descriptors (ArrayLike): (N, D) non-negative database histograms (a single (D,) one is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 distances.

    Raises:
        ValueError: If the shapes do not match, an input is empty, or a value is negative or not finite.
    """
    query_array, descriptor_array = _validate_histograms(query, descriptors)
    # The epsilon keeps bins that are empty in both histograms at 0 / tiny = 0 instead of 0 / 0.
    return ((descriptor_array - query_array) ** 2 / (descriptor_array + query_array + 1e-12)).sum(axis=1)


def hist_intersection(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the histogram intersection; higher values indicate greater similarity.

    For histograms that sum to 1, intersection = 1 - L1 / 2, so it ranks exactly like `l1`.

    Args:
        query (ArrayLike): (D,) non-negative query histogram.
        descriptors (ArrayLike): (N, D) non-negative database histograms (a single (D,) one is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 similarities.

    Raises:
        ValueError: If the shapes do not match, an input is empty, or a value is negative or not finite.
    """
    query_array, descriptor_array = _validate_histograms(query, descriptors)
    return np.minimum(descriptor_array, query_array).sum(axis=1)


def hellinger_kernel(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the Hellinger kernel (Bhattacharyya coefficient), sum(sqrt(h * q)); higher is more similar.

    Args:
        query (ArrayLike): (D,) non-negative query histogram.
        descriptors (ArrayLike): (N, D) non-negative database histograms (a single (D,) one is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 similarities (1 for identical histograms that sum to 1).

    Raises:
        ValueError: If the shapes do not match, an input is empty, or a value is negative or not finite.
    """
    query_array, descriptor_array = _validate_histograms(query, descriptors)
    return np.sqrt(descriptor_array * query_array).sum(axis=1)


def cosine_similarity(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the cosine similarity; higher values indicate greater similarity.

    Args:
        query (ArrayLike): (D,) query descriptor.
        descriptors (ArrayLike): (N, D) database descriptors (a single (D,) descriptor is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 similarities in [-1, 1]; 0 when either vector is all zeros.

    Raises:
        ValueError: If the shapes do not match, an input is empty or a value is not finite.
    """
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    denominator = np.linalg.norm(descriptor_array, axis=1) * np.linalg.norm(query_array)
    numerator = descriptor_array @ query_array
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def correlation(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the row-wise Pearson correlation; higher values indicate greater similarity.

    Args:
        query (ArrayLike): (D,) query descriptor.
        descriptors (ArrayLike): (N, D) database descriptors (a single (D,) descriptor is accepted).

    Returns:
        NDArray[np.float64]: (N,) float64 correlations in [-1, 1]; 0 when either vector is constant.

    Raises:
        ValueError: If the shapes do not match, an input is empty or a value is not finite.
    """
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    centered_query = query_array - query_array.mean()
    centered_descriptors = descriptor_array - descriptor_array.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered_descriptors, axis=1) * np.linalg.norm(centered_query)
    numerator = centered_descriptors @ centered_query
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


SIMILARITY_METRICS: dict[str, SimilarityMetric] = {
    metric.name: metric
    for metric in (
        SimilarityMetric("euclidean", euclidean, SimilarityKind.DISTANCE),
        SimilarityMetric("l1", l1, SimilarityKind.DISTANCE),
        SimilarityMetric("chi2", chi2, SimilarityKind.DISTANCE),
        SimilarityMetric("hist_intersection", hist_intersection, SimilarityKind.SIMILARITY),
        SimilarityMetric("hellinger_kernel", hellinger_kernel, SimilarityKind.SIMILARITY),
        SimilarityMetric("cosine_similarity", cosine_similarity, SimilarityKind.SIMILARITY),
        SimilarityMetric("correlation", correlation, SimilarityKind.SIMILARITY),
    )
}


__all__ = [
    "SimilarityKind",
    "SimilarityMetric",
    "SIMILARITY_METRICS",
    "euclidean",
    "l1",
    "chi2",
    "hist_intersection",
    "hellinger_kernel",
    "cosine_similarity",
    "correlation",
]
