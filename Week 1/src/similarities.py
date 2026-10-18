"""Vectorized distance and similarity measures for image descriptors."""

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
        name: Human-readable name used as the lookup key.
        function: Vectorized callable computing scores for a query against a batch.
        kind: Whether lower scores are better (``DISTANCE``) or higher (``SIMILARITY``).
    """

    name: str
    function: Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]
    kind: SimilarityKind


def _validate_inputs(
    query: ArrayLike,
    descriptors: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    if (query_array < 0).any() or (descriptor_array < 0).any():
        raise ValueError("histogram descriptors must be non-negative")
    return query_array, descriptor_array


def euclidean(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return L2 distance; lower values indicate greater similarity."""
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    return np.linalg.norm(descriptor_array - query_array, axis=1)


def l1(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return L1 distance; lower values indicate greater similarity."""
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    return np.abs(descriptor_array - query_array).sum(axis=1)


def chi2(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return chi-squared distance for non-negative histograms."""
    query_array, descriptor_array = _validate_histograms(query, descriptors)
    return ((descriptor_array - query_array) ** 2 / (descriptor_array + query_array + 1e-12)).sum(axis=1)


def hist_intersection(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return histogram intersection; higher values indicate greater similarity.

    For histograms that sum to 1, intersection = 1 - L1 / 2, so it ranks exactly like `l1`.
    """
    query_array, descriptor_array = _validate_histograms(query, descriptors)
    return np.minimum(descriptor_array, query_array).sum(axis=1)


def hellinger_kernel(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return the Bhattacharyya coefficient for non-negative histograms."""
    query_array, descriptor_array = _validate_histograms(query, descriptors)
    return np.sqrt(descriptor_array * query_array).sum(axis=1)


def cosine_similarity(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return cosine similarity; higher values indicate greater similarity."""
    query_array, descriptor_array = _validate_inputs(query, descriptors)
    denominator = np.linalg.norm(descriptor_array, axis=1) * np.linalg.norm(query_array)
    numerator = descriptor_array @ query_array
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def correlation(query: ArrayLike, descriptors: ArrayLike) -> NDArray[np.float64]:
    """Return row-wise Pearson correlation; higher values indicate similarity."""
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
