"""Evaluation of retrieval answers and background masks; ground truth is only read after inference.

mAP@K over painting rankings in reading order (Week 1 onwards; `apk`, `evaluate_retrieval`), foreground mask
precision / recall / F1 (Week 2 onwards; `evaluate_mask`), and the per-painting report of a Week 1-3 global
pipeline run (`evaluate_pipeline`, `main.py --evaluate`), optionally compared with retrieval on the ground-truth
masks. `evaluate_local` scores the Week 4 answers, which may contain unknowns ([-1]), with mAP@K plus the
precision / recall / F1 of the unknown decision. The Week 3-4 experiment scripts also call these functions.
"""

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.config import DescriptorConfig
from src.data_loading import has_ground_truth, load_ground_truth, load_mask
from src.pipeline import Predictions, rank_queries
from src.segmentation import extract_paintings
from src.unknowns import UNKNOWN, classification_scores


def evaluate_mask(
    predicted: NDArray[np.uint8],
    ground_truth: NDArray[np.uint8],
) -> tuple[float, float, float]:
    """Compute foreground precision, recall and F1 for one pair of masks.

    Args:
        predicted (NDArray[np.uint8]): (H, W) binary mask (e.g. uint8 0/255); nonzero values are foreground.
        ground_truth (NDArray[np.uint8]): (H, W) reference mask with the same convention.

    Returns:
        tuple[float, float, float]: Precision, recall and F1 of the foreground pixels. A metric with a zero
        denominator is 0, including when both masks contain only background.

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
    f1 = 2 * true_positives / total if total else 0.0  # 2PR / (P + R) written with counts: 2 TP / (|pred| + |gt|)
    return precision, recall, f1


# Adapted from Ben Hamner's Metrics implementation:
# https://github.com/benhamner/Metrics/tree/master/Python/ml_metrics/average_precision.py
def apk(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute average precision at K for one query.

    Args:
        actual (list[int]): Relevant database IDs; their order does not matter.
        predicted (list[int]): Database IDs ordered from best to worst match.
        k (int): Maximum number of predictions to consider.

    Returns:
        float: Average precision at K, or zero if there are no relevant IDs.
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
    *, max_paintings: int = 3,
) -> tuple[dict[str, float | int], list[dict[str, int | float | None]]]:
    """Score painting rankings in reading order; missing and extra slots score zero.

    Painting i of a query is compared with ground-truth entry i. An unknown ground-truth entry (-1) is only
    matched by a ranking that contains -1, i.e. the answer [-1].

    Args:
        rankings (dict[int, list[list[int]]]): Query ID -> painting rankings, in reading order.
        ground_truth (dict[int, list[int]]): Query ID -> database IDs in the same spatial order.
        max_paintings (int): Maximum number of paintings per query in the ground truth and in the predictions.

    Returns:
        tuple[dict[str, float | int], list[dict[str, int | float | None]]]: The summary ("map@1", "map@5",
        "map@10", "n_evaluated", "top1_hits") and one row per painting slot (query_id, painting_index,
        ground_truth_id, top1_id, correct_rank, ap1, ap5, ap10). The denominator is the sum of
        max(expected, detected) per query, so missing and extra paintings are penalised. The means are
        painting-weighted, not query-weighted.

    Raises:
        ValueError: If max_paintings is not a positive int, the query sets differ or are empty, or a query
            has 0 or more than max_paintings ground-truth paintings, or more than max_paintings predictions.
    """
    if type(max_paintings) is not int or max_paintings < 1:
        raise ValueError("max_paintings must be a positive integer.")
    if not ground_truth or set(rankings) != set(ground_truth):
        raise ValueError("Rankings must cover exactly the nonempty ground-truth query set.")
    rows: list[dict[str, int | float | None]] = []
    for query_id, actual in ground_truth.items():
        found = rankings[query_id]
        if not 1 <= len(actual) <= max_paintings or len(found) > max_paintings:
            raise ValueError(f"Expected 1 to {max_paintings} reference paintings and at most {max_paintings} "
                             "predictions.")
        # One slot per painting of either side: a missing prediction or an extra detection scores AP 0.
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


def evaluate_local(rankings: dict[int, list[list[int]]], ground_truth: dict[int, list[int]], *,
                    max_paintings: int = 3) -> dict[str, float | int]:
    """Evaluate final answers from any method, counting missing and extra painting slots.

    main.py uses it for the Week 4 local-feature answers, where a painting judged unknown is answered [-1].

    Args:
        rankings (dict[int, list[list[int]]]): Query ID -> per-painting answers in reading order.
        ground_truth (dict[int, list[int]]): Query ID -> database IDs (or -1 for unknown) in reading order.
        max_paintings (int): Maximum number of paintings per query, see `evaluate_retrieval`.

    Returns:
        dict[str, float | int]: The `evaluate_retrieval` summary plus the `evaluate_unknowns` metrics.

    Raises:
        ValueError: If the inputs are inconsistent (see `evaluate_retrieval`).
    """
    summary, _ = evaluate_retrieval(rankings, ground_truth, max_paintings=max_paintings)
    return {**summary, **evaluate_unknowns(rankings, ground_truth)}


def evaluate_unknowns(rankings: dict[int, list[list[int]]], ground_truth: dict[int, list[int]]) -> dict[str, float]:
    """Unknown-decision metrics, including missing and extra painting slots.

    "Unknown" (UNKNOWN = -1) is the positive class: a slot is predicted unknown when its answer is exactly
    [-1], and actually unknown when its ground-truth ID is -1.

    Args:
        rankings (dict[int, list[list[int]]]): Query ID -> per-painting answers in reading order.
        ground_truth (dict[int, list[int]]): Query ID -> database IDs (or -1) in reading order.

    Returns:
        dict[str, float]: "unknown_precision", "unknown_recall" and "unknown_f1".

    Raises:
        KeyError: If a ground-truth query has no rankings.
    """
    predicted, actual = [], []
    for q, truth in ground_truth.items():
        found = rankings[q]
        # Missing or extra slots count as "not unknown" on the side that lacks them.
        for i in range(max(len(found), len(truth))):
            predicted.append(i < len(found) and found[i] == [UNKNOWN])
            actual.append(i < len(truth) and truth[i] == UNKNOWN)
    precision, recall, f1 = classification_scores(predicted, actual)
    return {"unknown_precision": precision, "unknown_recall": recall, "unknown_f1": f1}


def evaluate_pipeline(predictions: "Predictions", *, compare_gt_masks: bool = False
                      ) -> tuple[dict[str, object], list[dict[str, object]], dict[int, list[list[int]]]]:
    """Score predictions after inference; optionally compare against GT-mask retrieval.

    Annotations are optional except for the explicit GT-mask comparison. Failed predictions remain in the
    denominator (with an empty ranking they score AP 0), so a method cannot raise its mAP by skipping
    hard paintings. Mask metrics are computed once per photo on the union mask and report their sample count.

    Args:
        predictions (Predictions): Output of src.pipeline.run_pipeline.
        compare_gt_masks (bool): Also rank the queries with the paintings of the ground-truth masks
            (``<query>.png``) and report, per painting, what the predicted masks lose or gain.

    Returns:
        tuple[dict[str, object], list[dict[str, object]], dict[int, list[list[int]]]]: The summary (run
        metadata, plus "retrieval" and "unknowns" with ground truth, "mask_metrics" with reference masks,
        and the GT-mask comparison when asked), one row per painting slot (scores when there is ground truth,
        status, mask metrics, GT-mask results), and the GT-mask rankings (empty without the comparison).

    Raises:
        ValueError: If the ground truth does not match the queries or the database, or the GT-mask comparison
            is asked without segmentation or without every reference mask.
    """
    query_dir = Path(str(predictions["metadata"]["query_dir"]))
    max_paintings = int(predictions["metadata"].get("max_paintings", 3))
    # At least 3 slots, so older methods (1-2 paintings per query) can also be evaluated on Week 4 datasets.
    evaluation_limit = max(3, max_paintings)
    gt = load_ground_truth(query_dir) if has_ground_truth(query_dir) else None
    if gt is not None:
        if set(gt) != set(predictions["rankings"]):
            raise ValueError("Query IDs and ground-truth IDs differ.")
        if any(not 1 <= len(ids) <= evaluation_limit or any(type(i) is not int for i in ids) for ids in gt.values()):
            raise ValueError(f"Expected 1 to {evaluation_limit} painting IDs per query, in reading order.")
        if any(i != UNKNOWN and i not in predictions["database"] for ids in gt.values() for i in ids):
            raise ValueError("Ground truth references images absent from the database.")
    masks = predictions["masks"]
    if compare_gt_masks and not masks:
        raise ValueError("GT-mask comparison requires segmentation mode.")
    # Reference masks are looked for only when the run segmented (cropped QS1 queries have none).
    references = {q: load_mask(p.with_suffix(".png")) for q, p in predictions["query_paths"].items()
                  if masks and p.with_suffix(".png").is_file()}
    oracle, oracle_statuses = {}, {}
    if compare_gt_masks:
        if set(references) != set(predictions["query_paths"]):
            raise ValueError("GT-mask comparison requires every reference mask.")
        descriptor = cast(DescriptorConfig, predictions["metadata"]["descriptor"])
        similarity = str(predictions["metadata"]["similarity"])
        reference_paintings = {q: extract_paintings(mask, max_paintings=max_paintings)[1]
                               for q, mask in references.items()}
        oracle, oracle_statuses = rank_queries(predictions["query_paths"], predictions["database"],
                                               descriptor, similarity,
                                               paintings=reference_paintings,
                                               week=int(predictions["metadata"].get("segmentation", {}).get("week", 4)))
    summary = dict(predictions["metadata"])
    oracle_metrics = {}
    if gt is not None:
        summary["retrieval"], rows = evaluate_retrieval(predictions["rankings"], gt, max_paintings=evaluation_limit)
        summary["unknowns"] = evaluate_unknowns(predictions["rankings"], gt)
        if compare_gt_masks:
            summary["gt_mask_retrieval"], oracle_rows = evaluate_retrieval(oracle, gt, max_paintings=evaluation_limit)
            oracle_metrics = {(r["query_id"], r["painting_index"]): r for r in oracle_rows}
    else:  # no ground truth: rows only list each slot's top-1 answer
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
