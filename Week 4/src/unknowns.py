"""Week 4 Task 2-II: final per-painting decision between a museum ranking and unknown ([-1]).

The submitted rule (``decide``, called by ``src.local_pipeline``): accept the local ranking when its best score
(distinct RANSAC inliers) is at least ``UNKNOWN_THRESHOLD`` (10); otherwise accept the Week 3 global colour +
texture ranking only if its best / second distance ratio is below ``GLOBAL_RATIO`` (0.85); otherwise answer
[-1]. Both thresholds were chosen on QSD1-W4 (``weeks/week4/README.md``). ``classification_scores`` gives the
unknown-detection precision / recall / F1 used by the evaluation and the experiments.
"""

from dataclasses import dataclass

UNKNOWN = -1  # museum id answered for a painting that is not in the museum


@dataclass(frozen=True)
class Decision:
    """A final ranking and its explanation, also used by the inspection notebook.

    Attributes:
        ranking: Museum ids (top 10) or ``[UNKNOWN]``.
        reason: "Local match", "Global fallback" or "Unknown rejection".
        global_ratio: Best / second global distance ratio when the fallback was consulted, else None.
    """

    ranking: list[int]
    reason: str
    global_ratio: float | None = None


def decide(ranking: list[int], scores: dict[int, int], threshold: int,
           fallback: tuple[list[int], float] | None = None, *, global_ratio: float) -> Decision:
    """Accept local evidence, then a confident global candidate, otherwise answer [-1].

    Args:
        ranking: Local ranking of museum ids for one painting.
        scores: Local score of every museum image for that painting (museum id -> distinct inliers).
        threshold: Minimum best local score to accept the local ranking.
        fallback: Global (ranking, best / second distance ratio) of the same painting, or None.
        global_ratio: The fallback ranking is accepted only if its ratio is below this value.

    Returns:
        The ``Decision`` for the painting.
    """
    local = reject_unknown(ranking, scores, threshold)
    if local != [UNKNOWN]:
        return Decision(local, "Local match")
    # A low ratio means the best global candidate is clearly closer than the second one: the global
    # descriptor is confident even though few keypoints matched.
    if fallback is not None and fallback[0] and fallback[1] < global_ratio:
        return Decision(fallback[0], "Global fallback", fallback[1])
    return Decision([UNKNOWN], "Unknown rejection", fallback[1] if fallback is not None else None)


def classification_scores(predicted: list[bool], actual: list[bool]) -> tuple[float, float, float]:
    """Precision, recall and F1 of unknown detection; unknown is the positive class.

    Args:
        predicted: Per painting, True if answered unknown.
        actual: Per painting, True if it is really unknown.

    Returns:
        (precision, recall, f1); each is 0 when its denominator is 0.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(predicted) != len(actual):
        raise ValueError("Predictions and labels must have the same length.")
    tp = sum(p and a for p, a in zip(predicted, actual))
    precision = tp / max(sum(predicted), 1)
    recall = tp / max(sum(actual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def reject_unknown(ranking: list[int], scores: dict[int, int], threshold: int) -> list[int]:
    """Answer [-1] when no ranking is available or the best local score is too low.

    Args:
        ranking: Local ranking of museum ids.
        scores: Museum id -> local score.
        threshold: Minimum best score to keep the ranking.

    Returns:
        ``ranking`` unchanged, or ``[UNKNOWN]``.
    """
    if not ranking or max(scores.values(), default=0) < threshold:
        return [UNKNOWN]
    return ranking
