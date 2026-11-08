"""Week 4, Task 2 (II): unknown paintings answered [-1], threshold chosen by the F1 of the "unknown" decision.

Scores come from the default local method (src/local_pipeline.py: DoG + RootSIFT, ratio 0.75, cross-check,
distinct RANSAC inliers). Experimental rule (defined in this study): a painting is unknown when its best score is below T, optionally
also when best / second best < R. T (and R) maximise F1 on QSD1-W4 (10 unknown, 29 known paintings; paintings the
segmentation misses or adds are included in the evaluation). The same rule is then applied to QSD2-W3, where every painting is
known, to count wrong rejections. mAP is computed with [-1] as the correct answer for an unknown painting.

Writes weeks/week4/results/task2/unknown_threshold.png and unknown_threshold.csv (F1 curve on QSD1-W4).
Usage (from the repository root): python weeks/week4/experiments/task2_unknowns.py [--workers 8]
"""
import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data_loading import get_image_id, list_images, load_ground_truth, load_image, resolve_dataset  # noqa: E402
from src.evaluate import evaluate_retrieval  # noqa: E402
from src.local_pipeline import CACHE_DIR, LocalMethod, matching_database, rank_query  # noqa: E402
from src.matching import rank_scores  # noqa: E402
from src.unknowns import UNKNOWN, classification_scores  # noqa: E402

@dataclass(frozen=True)
class RejectionScores:
    """Best verified-match count per development painting and whether it is truly unknown."""

    best_scores: list[int]
    is_unknown: list[bool]
    second_scores: list[int] | None = None
    detected: list[bool] | None = None


def top_two(scores: dict[int, int]) -> tuple[int, int]:
    """Best and second-best score of one painting against the museum."""
    best = sorted(scores.values(), reverse=True) + [0, 0]
    return best[0], best[1]


def is_rejected(best: int, second: int, threshold: int, min_ratio: float = 1.0) -> bool:
    """The unknown decision: best score below threshold, or (min_ratio > 1) best not min_ratio times the second."""
    return best < threshold or (min_ratio > 1.0 and best < min_ratio * max(second, 1))


def reject_unknown(ranking: list[int], scores: dict[int, int], threshold: int, min_ratio: float = 1.0) -> list[int]:
    """Return [-1] when the painting is rejected (is_rejected), else the ranking."""
    if not ranking or is_rejected(*top_two(scores), threshold, min_ratio):
        return [UNKNOWN]
    return ranking


def painting_scores(scores: dict[int, list[dict[int, int]]], ground_truth: dict[int, list[int]]) -> RejectionScores:
    """Align all painting slots, including missing and extra detections."""
    if set(scores) != set(ground_truth):
        raise ValueError("Scores and ground truth must cover the same queries.")
    best, second, unknown, detected = [], [], [], []
    for query_id, paintings in scores.items():
        truth = ground_truth[query_id]
        for i in range(max(len(paintings), len(truth))):
            found = i < len(paintings)
            b, c = top_two(paintings[i]) if found else (0, 0)
            best.append(b)
            second.append(c)
            unknown.append(i < len(truth) and truth[i] == UNKNOWN)
            detected.append(found)
    return RejectionScores(best, unknown, second, detected)


def f1_at(data: RejectionScores, threshold: int, min_ratio: float = 1.0) -> tuple[float, float, float]:
    """Precision, recall and F1 of the 'unknown' decision (unknown = positive class)."""
    seconds = data.second_scores or [0] * len(data.best_scores)
    detected = data.detected if data.detected is not None else [True] * len(data.best_scores)
    predicted = [found and is_rejected(b, c, threshold, min_ratio)
                 for b, c, found in zip(data.best_scores, seconds, detected)]
    return classification_scores(predicted, data.is_unknown)


def optimize_rule(data: RejectionScores, ratios: tuple[float, ...] = (1.0,)) -> tuple[int, float, float]:
    """(threshold, min_ratio, F1) with the best F1; ties go to the lowest threshold, then the lowest ratio."""
    best = (0, 1.0, -1.0)
    for ratio in sorted(ratios):
        candidates = sorted({0, *(score + 1 for score in data.best_scores)})
        for threshold in candidates:
            f1 = f1_at(data, threshold, ratio)[2]
            if f1 > best[2] or (f1 == best[2] and threshold < best[0]):
                best = (threshold, ratio, f1)
    return best



METHOD = LocalMethod()  # the default of src/local_pipeline.py
OUT = ROOT / "weeks" / "week4" / "results" / "task2"
RATIOS = (1.0, 1.25, 1.5, 2.0, 3.0)  # 1.0 = ratio rule off
_DB: dict = {}


def _init(corpus: str) -> None:
    _DB["db"] = matching_database(Path(corpus), METHOD)


def _query(path: str) -> tuple[int, list[dict[int, int]]]:
    return get_image_id(Path(path)), rank_query(load_image(Path(path)), _DB["db"], METHOD)[1]


def scores_for(name: str, data: Path, workers: int) -> dict[int, list[dict[int, int]]]:
    """Score of every museum image for every painting of every query (cached in .cache/ as JSON)."""
    cache = CACHE_DIR.parent / f"scores_{name}_{METHOD.score_cache_key(resolve_dataset(data, name), resolve_dataset(data, 'BBDD'))}.json"
    if cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {int(q): [{int(b): v for b, v in s.items()} for s in p] for q, p in raw.items()}
    paths = [str(p) for p in list_images(resolve_dataset(data, name))]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(resolve_dataset(data, "BBDD")),)) as pool:
        scores = dict(pool.map(_query, paths))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scores), encoding="utf-8")
    return scores


def evaluate(scores, gt, threshold=None, ratio=1.0) -> dict:
    """top-1 / mAP@1 / mAP@5 over all and known paintings; with a threshold, rejected paintings answer [-1]."""
    rankings = {q: [reject_unknown(rank_scores(s, 10), s, threshold, ratio) if threshold is not None else rank_scores(s, 10)
                    for s in p] for q, p in scores.items()}
    summary, rows = evaluate_retrieval(rankings, gt)
    known = [r for r in rows if r["ground_truth_id"] not in (None, UNKNOWN)]
    return {"top1": f"{summary['top1_hits']}/{summary['n_evaluated']}", "map1": summary["map@1"], "map5": summary["map@5"],
            "top1_known": f"{sum(r['ap1'] == 1 for r in known)}/{len(known)}",
            "map1_known": statistics.mean(r["ap1"] for r in known), "map5_known": statistics.mean(r["ap5"] for r in known)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    sets = {name: (scores_for(name, args.data, args.workers), load_ground_truth(resolve_dataset(args.data, name)))
            for name in ("qsd1_w4", "qsd2_w3")}
    scores, gt = sets["qsd1_w4"]
    data = painting_scores(scores, gt)
    t_only, _, f1_t = optimize_rule(data)
    t_r, r, f1_tr = optimize_rule(data, RATIOS)
    curve = []
    for t in range(0, max(data.best_scores) + 2):
        p_, r_, f = f1_at(data, t)
        curve.append({"threshold": t, "precision": round(p_, 4), "recall": round(r_, 4), "f1": round(f, 4)})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "unknown_threshold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    print("QSD1-W4 painting scores: unknown best", sorted(b for b, u in zip(data.best_scores, data.is_unknown) if u),
          "| known best", sorted(b for b, u in zip(data.best_scores, data.is_unknown) if not u))
    for label, (t, ratio) in (("before rejection", (None, 1.0)), ("T only", (t_only, 1.0)), ("T + ratio", (t_r, r))):
        line = {"rule": label, "T": t, "R": ratio}
        if t is not None:
            p_, r_, f = f1_at(data, t, ratio)
            line |= {"precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f, 3)}
        for name, (s, g) in sets.items():
            e = evaluate(s, g, t, ratio)
            line[name] = f"top1 {e['top1']} mAP@1 {e['map1']:.3f} mAP@5 {e['map5']:.3f} | known {e['top1_known']} mAP@1 {e['map1_known']:.3f}"
        if t is not None:
            w3 = painting_scores(*sets["qsd2_w3"])
            line["qsd2_w3_wrongly_rejected"] = f"{sum(is_rejected(b, c, t, ratio) for b, c in zip(w3.best_scores, w3.second_scores))}/{len(w3.best_scores)}"
        print(line)
    figure(data, curve, t_only, painting_scores(*sets["qsd2_w3"]))


def figure(data, curve, threshold, w3) -> None:
    """F1 / precision / recall against T on QSD1-W4, and the best score of every painting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.8), facecolor="white", gridspec_kw={"width_ratios": [3, 2]})
    ts = [c["threshold"] for c in curve]
    for key, color in (("f1", "#1d4ed8"), ("precision", "#16a34a"), ("recall", "#ea580c")):
        a.plot(ts, [c[key] for c in curve], color=color, lw=2, label=key)
    a.axvline(threshold, color="#1f2933", ls="--", lw=1.2, label=f"chosen T = {threshold}")
    a.set_xscale("symlog", linthresh=10)
    a.set_xlim(left=0)
    a.set_xlabel("T (paintings with best distinct-inlier score < T answer [-1])")
    a.set_ylabel("unknown decision on QSD1-W4")
    a.set_title("F1 of the unknown decision against T (QSD1-W4: 10 unknown, 29 known)", loc="left")
    a.legend(frameon=False)
    groups = [("QSD1-W4 unknown", [x for x, u in zip(data.best_scores, data.is_unknown) if u], "#b91c1c"),
              ("QSD1-W4 known", [x for x, u in zip(data.best_scores, data.is_unknown) if not u], "#15803d"),
              ("QSD2-W3 known", w3.best_scores, "#64748b")]
    for k, (label, values, color) in enumerate(groups):
        b.scatter([max(v, 0.5) for v in values], [k] * len(values), color=color, s=40, alpha=0.8)
    b.axvline(threshold, color="#1f2933", ls="--", lw=1.2)
    b.set_xscale("log")
    b.set_yticks(range(len(groups)), [g[0] for g in groups])
    b.set_xlabel("best score of the painting (distinct inliers, log; 0 drawn at 0.5)")
    b.set_title("Best score per painting", loc="left")
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "unknown_threshold.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
