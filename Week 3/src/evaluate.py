"""Evaluation: mask precision / recall / F1, mAP@K (one ranking per painting, left to right) and the
per-query report of a pipeline run (`main.py --evaluate`)."""

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.config import DescriptorConfig
from src.data_loading import has_ground_truth, load_ground_truth, load_mask
from src.pipeline import Predictions, rank_queries
from src.segmentation import extract_paintings


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


def evaluate_retrieval(
    rankings: dict[int, list[list[int]]],
    ground_truth: dict[int, list[int]],
) -> tuple[dict[str, float | int], list[dict[str, int | float | None]]]:
    """Score left-to-right painting rankings; missing and extra slots score zero.

    Args:
        rankings: Query ID -> painting rankings, in left-to-right order.
        ground_truth: Query ID -> database IDs in the same spatial order.

    Returns:
        Per-painting mAP@1/5/10, hit count and instance rows. The denominator is
        the sum of max(expected, detected) per query, so extras are penalized.
        This diagnostic uses painting-weighted means, not query-weighted means.
    """
    if not ground_truth or set(rankings) != set(ground_truth):
        raise ValueError("Rankings must cover exactly the nonempty ground-truth query set.")
    rows: list[dict[str, int | float | None]] = []
    for query_id, actual in ground_truth.items():
        found = rankings[query_id]
        if not 1 <= len(actual) <= 2 or len(found) > 2:
            raise ValueError("Expected one or two reference paintings and up to two predictions.")
        for index in range(max(len(actual), len(found))):
            expected_id = actual[index] if index < len(actual) else None
            ranking = found[index] if index < len(found) else []
            relevant = [expected_id] if expected_id is not None else []
            rows.append({
                "query_id": query_id, "painting_index": index,
                "ground_truth_id": expected_id, "top1_id": ranking[0] if ranking else None,
                "correct_rank": next((i for i, value in enumerate(ranking, 1) if value in relevant), None),
                **{f"ap{k}": apk(relevant, ranking, k) for k in (1, 5, 10)},
            })
    summary: dict[str, float | int] = {
        f"map@{k}": float(np.mean([row[f"ap{k}"] for row in rows])) for k in (1, 5, 10)
    }
    summary["n_evaluated"] = len(rows)
    summary["top1_hits"] = sum(row["ap1"] == 1 for row in rows)
    return summary, rows


def evaluate_pipeline(predictions: "Predictions", *, compare_gt_masks: bool = False
                      ) -> tuple[dict[str, object], list[dict[str, object]], dict[int, list[list[int]]]]:
    """Score predictions after inference; optionally compare against GT-mask retrieval.

    Annotations are optional except for the explicit GT-mask comparison. Failed
    predictions remain in the denominator. Mask metrics report their sample count.
    """
    query_dir = Path(str(predictions["metadata"]["query_dir"]))
    gt = load_ground_truth(query_dir) if has_ground_truth(query_dir) else None
    if gt is not None:
        if set(gt) != set(predictions["rankings"]):
            raise ValueError("Query IDs and ground-truth IDs differ.")
        if any(not 1 <= len(ids) <= 2 or any(type(i) is not int for i in ids) for ids in gt.values()):
            raise ValueError("Expected one or two painting IDs per query, ordered left to right.")
        if any(i not in predictions["database"] for ids in gt.values() for i in ids):
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
        descriptor = cast(DescriptorConfig, predictions["metadata"]["descriptor"])
        similarity = str(predictions["metadata"]["similarity"])
        reference_paintings = {q: extract_paintings(mask)[1] for q, mask in references.items()}
        oracle, oracle_statuses = rank_queries(predictions["query_paths"], predictions["database"],
                                               descriptor, similarity,
                                               paintings=reference_paintings)
    summary = dict(predictions["metadata"])
    oracle_metrics = {}
    if gt is not None:
        summary["retrieval"], rows = evaluate_retrieval(predictions["rankings"], gt)
        if compare_gt_masks:
            summary["gt_mask_retrieval"], oracle_rows = evaluate_retrieval(oracle, gt)
            oracle_metrics = {(r["query_id"], r["painting_index"]): r for r in oracle_rows}
    else:
        rows = [{"query_id": q, "painting_index": index,
                 "top1_id": rankings[index][0] if index < len(rankings) and rankings[index] else None}
                for q, rankings in predictions["rankings"].items()
                for index in range(max(1, len(rankings)))]

    # Evaluate the union once per image, independently of the number of paintings.
    mask_rows = []
    for q, reference in references.items():
        precision, recall, f1 = evaluate_mask(masks[q], reference)
        mask_rows.append({"query_id": q, "precision": precision, "recall": recall, "f1": f1})
    mask_metrics = {r["query_id"]: r for r in mask_rows}
    for row in rows:
        q, index = row["query_id"], row["painting_index"]
        states = predictions["statuses"][q]
        row["status"] = states[index] if index < len(states) else "missing_painting"
        row.update(mask_metrics.get(q, {}))
        if compare_gt_masks:
            oracle_ranking = oracle[q][index] if index < len(oracle[q]) else []
            row["gt_mask_status"] = oracle_statuses[q][index] if index < len(oracle[q]) else "missing_painting"
            row["gt_mask_top1_id"] = oracle_ranking[0] if oracle_ranking else None
            if gt is not None:
                metrics = oracle_metrics.get((q, index), {"ap1": 0.0, "ap5": 0.0, "ap10": 0.0,
                                                         "correct_rank": None})
                row.update({f"gt_mask_{key}": metrics[key] for key in ("ap1", "ap5", "ap10", "correct_rank")})
    if gt is not None and compare_gt_masks:
        summary["top1_lost_vs_gt_masks"] = [r["query_id"] for r in rows if r["gt_mask_ap1"] == 1 and r["ap1"] == 0]
        summary["top1_gained_vs_gt_masks"] = [r["query_id"] for r in rows if r["gt_mask_ap1"] == 0 and r["ap1"] == 1]
    if mask_rows:
        summary["mask_metrics"] = {
            "n_evaluated": len(mask_rows),
            **{f"mean_{k}": float(np.mean([r[k] for r in mask_rows])) for k in ("precision", "recall", "f1")},
            "min_f1": min(r["f1"] for r in mask_rows),
            "below_f1_0.9": [r["query_id"] for r in mask_rows if r["f1"] < .9],
        }
    return summary, rows, oracle
