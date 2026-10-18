import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.config import WEEK2_METHODS
from src.data_loading import has_ground_truth, load_ground_truth, load_mask
from src.pipeline import Predictions, rank_queries


def evaluate_mask(
    predicted: NDArray[np.uint8],
    ground_truth: NDArray[np.uint8],
) -> tuple[float, float, float]:
    """Compute foreground precision, recall and F1 for one pair of masks.

    Args:
        predicted: Two-dimensional binary mask; nonzero values are foreground.
        ground_truth: Reference mask with the same shape and convention.

    Returns:
        Precision, recall and F1. A metric with a zero denominator is 0,
        including when both masks contain only background.

    Raises:
        ValueError: If masks are not two-dimensional or their shapes differ.
    """
    if predicted.ndim != 2 or ground_truth.ndim != 2 or predicted.shape != ground_truth.shape:
        raise ValueError("Masks must be two-dimensional and have the same shape.")

    predicted_fg = predicted != 0
    actual_fg = ground_truth != 0
    true_positives = np.count_nonzero(predicted_fg & actual_fg)
    predicted_count = np.count_nonzero(predicted_fg)
    actual_count = np.count_nonzero(actual_fg)

    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / actual_count if actual_count else 0.0
    total = predicted_count + actual_count
    f1 = 2 * true_positives / total if total else 0.0
    return precision, recall, f1


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


def evaluate_pipeline(predictions: "Predictions", *, compare_gt_masks: bool = False
                      ) -> tuple[dict[str, object], list[dict[str, object]], dict[int, list[int]]]:
    """Score predictions after inference; optionally compare against GT-mask retrieval.

    Annotations are optional except for the explicit GT-mask comparison. Failed
    predictions remain in the denominator. Mask metrics report their sample count.
    """
    query_dir = Path(str(predictions["metadata"]["query_dir"]))
    gt = load_ground_truth(query_dir) if has_ground_truth(query_dir) else None
    if gt is not None:
        if set(gt) != set(predictions["rankings"]):
            raise ValueError("Query IDs and ground-truth IDs differ.")
        if any(len(ids) != 1 or type(ids[0]) is not int for ids in gt.values()):
            raise ValueError("This pipeline evaluates one painting per query.")
        if any(ids[0] not in predictions["database"] for ids in gt.values()):
            raise ValueError("Ground truth references images absent from the database.")
    masks = predictions["masks"]
    if compare_gt_masks and not masks:
        raise ValueError("GT-mask comparison requires segmentation mode.")
    references = {q: load_mask(p.with_suffix(".png")) for q, p in predictions["query_paths"].items()
                  if masks and p.with_suffix(".png").is_file()}
    oracle, oracle_statuses = {}, {}
    if compare_gt_masks:
        if set(references) != set(predictions["query_paths"]):
            raise ValueError("GT-mask comparison requires every reference mask.")
        selected = WEEK2_METHODS[str(predictions["metadata"]["method"])]
        oracle, oracle_statuses = rank_queries(predictions["query_paths"], predictions["database"],
                                               selected["descriptor"], selected["similarity"], masks=references)
    rows = []
    for q, ranking in predictions["rankings"].items():
        row = {"query_id": q, "status": predictions["statuses"][q],
               "top1_id": ranking[0] if ranking else None}
        if gt is not None:
            row["ground_truth_id"] = gt[q][0]
            row.update(_ranking_metrics(gt[q], ranking))
        if q in references:
            precision, recall, f1 = evaluate_mask(masks[q], references[q])
            row.update(precision=precision, recall=recall, f1=f1)
        if compare_gt_masks:
            row["gt_mask_status"] = oracle_statuses[q]
            row["gt_mask_top1_id"] = oracle[q][0] if oracle[q] else None
            if gt is not None:
                row.update({f"gt_mask_{k}": v for k, v in _ranking_metrics(gt[q], oracle[q]).items()})
        rows.append(row)
    summary = dict(predictions["metadata"])
    if gt is not None:
        for prefix in (["", "gt_mask_"] if compare_gt_masks else [""]):
            summary[f"{prefix}retrieval"] = {
                **{f"map@{k}": float(np.mean([r[f"{prefix}ap{k}"] for r in rows])) for k in (1, 5, 10)},
                "top1_hits": sum(r[f"{prefix}ap1"] == 1 for r in rows),
            }
        if compare_gt_masks:
            summary["top1_lost_vs_gt_masks"] = [r["query_id"] for r in rows if r["gt_mask_ap1"] == 1 and r["ap1"] == 0]
            summary["top1_gained_vs_gt_masks"] = [r["query_id"] for r in rows if r["gt_mask_ap1"] == 0 and r["ap1"] == 1]
    mask_rows = [r for r in rows if "f1" in r]
    if mask_rows:
        summary["mask_metrics"] = {
            "n_evaluated": len(mask_rows),
            **{f"mean_{k}": float(np.mean([r[k] for r in mask_rows])) for k in ("precision", "recall", "f1")},
            "min_f1": min(r["f1"] for r in mask_rows),
            "below_f1_0.9": [r["query_id"] for r in mask_rows if r["f1"] < .9],
        }
    return summary, rows, oracle


def _ranking_metrics(actual: list[int], ranking: list[int]) -> dict[str, float | int | None]:
    return {"ap1": apk(actual, ranking, 1), "ap5": apk(actual, ranking, 5),
            "ap10": apk(actual, ranking, 10),
            "correct_rank": next((i for i, value in enumerate(ranking, 1) if value in actual), None)}
