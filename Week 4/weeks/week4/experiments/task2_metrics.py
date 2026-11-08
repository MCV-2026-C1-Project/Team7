"""Week 4, Task 2 (I): compare similarity metrics, tentative-match filters and verification scores.

Descriptors (src/local_descriptors.py, at most three options as the statement allows): SIFT, ORB and
BRISK, all on externally detected DoG keypoints, selected by the Task 1 comparison. Grid:
  similarity  SIFT: L2, L1, Hellinger (RootSIFT: L1-normalise, square root, then L2); ORB / BRISK: Hamming
  tentative   Lowe ratio 0.7 / 0.75 / 0.8, cross-check off / on
  score       matches: number of tentative matches (no RANSAC); inliers: RANSAC homography inliers (5 px);
              distinct: inliers counted once per distinct museum keypoint (stops many-to-one homographies)
Fixed: shared preprocessing per painting (segmentation, noise filter, crop, keypoints inside the mask), top 10.

Every (painting, museum image) pair is matched once per descriptor and metric (k=2 nearest neighbours one way, the
best match the other way); all ratio / cross-check / score settings are derived from that. Queries run in
parallel. Time per query = feature extraction + matching (+ the reverse match with cross-check) + filtering
(+ RANSAC), measured per query.

Writes weeks/week4/results/task2/: metrics_<set>.csv (one row per configuration), metrics_<set>_paintings.csv (rank of
the correct image per painting and configuration), metrics_<set>.png (mAP@1 per configuration),
match_example_<set>.png (a correct match with its inliers).
Usage (from the repository root): python weeks/week4/experiments/task2_metrics.py --set qsd2_w3 [--workers 8]
"""
import argparse
import csv
import itertools
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data_loading import get_image_id, list_images, load_ground_truth, load_image, resolve_dataset  # noqa: E402
from src.evaluate import evaluate_retrieval  # noqa: E402
from src.keypoints import DoG  # noqa: E402
from src.local_descriptors import BRISKConfig, ORBConfig, SIFTConfig  # noqa: E402
from src.local_pipeline import LocalMethod, museum_features, query_paintings  # noqa: E402
from src.matching import Features, MatchConfig, extract_features  # noqa: E402

OUT = ROOT / "weeks" / "week4" / "results" / "task2"
MATCH = MatchConfig(reprojection_px=5.0)  # max_side 900, max_points 1000 (the cached museum features)
METHODS = {"SIFT": LocalMethod(DoG(), SIFTConfig(), MATCH), "ORB": LocalMethod(DoG(), ORBConfig(), MATCH),
           "BRISK": LocalMethod(DoG(), BRISKConfig(), MATCH)}
METRICS = {"SIFT": ["L2", "L1", "Hellinger"], "ORB": ["Hamming"], "BRISK": ["Hamming"]}
NORMS = {"L2": cv2.NORM_L2, "L1": cv2.NORM_L1, "Hellinger": cv2.NORM_L2, "Hamming": cv2.NORM_HAMMING}
RATIOS = (0.7, 0.75, 0.8)
CROSS = (False, True)
SCORES = ("matches", "inliers", "distinct")
TOP_K = 10


def root_sift(descriptors: np.ndarray) -> np.ndarray:
    """RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise and take the square root; L2 then acts as Hellinger."""
    d = descriptors.astype(np.float32)
    return np.sqrt(d / np.maximum(d.sum(axis=1, keepdims=True), 1e-12))


def prepare(descriptors: np.ndarray, metric: str) -> np.ndarray:
    return root_sift(descriptors) if metric == "Hellinger" else descriptors


def configs():
    """Every (descriptor, metric, ratio, cross-check, score) of the grid."""
    for name, metrics in METRICS.items():
        for metric, ratio, cross, score in itertools.product(metrics, RATIOS, CROSS, SCORES):
            yield name, metric, ratio, cross, score


def scores_for_pair(query: Features, q_desc: np.ndarray, reference: Features, r_desc: np.ndarray, norm: int
                    ) -> tuple[dict, float, float]:
    """Scores of every (ratio, cross, score) for one painting against one museum image, and the time of the
    reverse match and of the RANSAC part (to charge them only to the configurations that use them)."""
    out = {}
    if len(q_desc) < 2 or len(r_desc) < 2:
        return {(r, c, s): 0 for r in RATIOS for c in CROSS for s in SCORES}, 0.0, 0.0
    matcher = cv2.BFMatcher(norm)
    knn = [m for m in matcher.knnMatch(q_desc, r_desc, k=2) if len(m) == 2]
    t0 = time.perf_counter()
    best_back = {m.queryIdx: m.trainIdx for m in matcher.match(r_desc, q_desc)}
    t_back = time.perf_counter() - t0
    t_ransac = 0.0
    for ratio, cross in itertools.product(RATIOS, CROSS):
        pairs = [(a.queryIdx, a.trainIdx) for a, b in knn if a.distance < ratio * b.distance]
        if cross:
            pairs = [(q, r) for q, r in pairs if best_back.get(r) == q]
        out[(ratio, cross, "matches")] = len(pairs)
        t0 = time.perf_counter()
        inliers = []
        if len(pairs) >= 4:
            src = query.points[[q for q, _ in pairs]]
            dst = reference.points[[r for _, r in pairs]]
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, MATCH.reprojection_px)
            if homography is not None:
                inliers = [p for p, keep in zip(pairs, mask.ravel()) if keep]
        t_ransac += time.perf_counter() - t0
        out[(ratio, cross, "inliers")] = len(inliers)
        out[(ratio, cross, "distinct")] = len({r for _, r in inliers})
    return out, t_back, t_ransac


_DB: dict = {}
_PREPARED: dict = {}


def _init(corpus: str) -> None:
    """Worker: load the cached museum features once (and their RootSIFT version)."""
    for name, method in METHODS.items():
        _DB[name] = museum_features(Path(corpus), method)
        for metric in METRICS[name]:
            _PREPARED[(name, metric)] = {bid: prepare(f.descriptors, metric) for bid, f in _DB[name].items()}


def _query(path: str) -> tuple[int, list, dict]:
    """Worker: scores of every configuration for every painting of one query, and its time per configuration."""
    image = load_image(Path(path))
    paintings = query_paintings(image)  # Segmentation and filtering shared by all descriptors.
    scores = [{} for _ in paintings]  # per painting: config -> {museum id: score}
    seconds = {}
    for name, method in METHODS.items():
        for metric in METRICS[name]:
            t_extract = t_match = 0.0
            t_back = dict.fromkeys(CROSS, 0.0)
            t_ransac = 0.0
            per_painting = []
            for gray, mask in paintings:
                t0 = time.perf_counter()
                features = extract_features(gray, method.detector, method.descriptor, method.match, mask=mask)
                q_desc = prepare(features.descriptors, metric)
                t_extract += time.perf_counter() - t0
                t0 = time.perf_counter()
                results = {}
                for bid, reference in _DB[name].items():
                    s, tb, tr = scores_for_pair(features, q_desc, reference, _PREPARED[(name, metric)][bid],
                                                NORMS[metric])
                    results[bid] = s
                    t_back[True] += tb
                    t_ransac += tr
                t_match += time.perf_counter() - t0
                per_painting.append(results)
            t_match -= t_back[True] + t_ransac  # forward matching and filtering only
            for ratio, cross, score in itertools.product(RATIOS, CROSS, SCORES):
                key = (name, metric, ratio, cross, score)
                for i, results in enumerate(per_painting):
                    scores[i][key] = {bid: s[(ratio, cross, score)] for bid, s in results.items()}
                ransac_share = t_ransac / (len(RATIOS) * len(CROSS)) if score != "matches" else 0.0
                seconds[key] = t_extract + t_match + (t_back[True] if cross else 0.0) + ransac_share
    return get_image_id(Path(path)), scores, seconds


def ranking(scores: dict[int, int]) -> list[int]:
    """Top-k museum ids by decreasing score (ties: museum order, as src.matching.rank_scores)."""
    return sorted(scores, key=lambda b: -scores[b])[:TOP_K]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="qsd2_w3")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--limit", type=int, help="only the first N queries (timing estimate)")
    args = parser.parse_args()
    folder, corpus = resolve_dataset(args.data, args.set), resolve_dataset(args.data, "BBDD")
    gt = load_ground_truth(folder)
    for method in METHODS.values():  # build the caches once, before the workers read them
        museum_features(corpus, method)
    paths = [str(p) for p in list_images(folder)][: args.limit]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(corpus),)) as pool:
        results = list(pool.map(_query, paths))
    wall = time.perf_counter() - start
    print(f"{len(paths)} queries, {len(list(configs()))} configurations, {wall:.0f} s wall clock ({args.workers} workers)")
    if args.limit:
        return
    rows, painting_rows = [], []
    for key in configs():
        rankings = {q: [ranking(p[key]) for p in scores] for q, scores, _ in results}
        summary, per = evaluate_retrieval(rankings, gt)
        name, metric, ratio, cross, score = key
        rows.append({"set": args.set, "descriptor": name, "metric": metric, "ratio": ratio,
                     "cross_check": "on" if cross else "off", "score": score,
                     "top1": f"{summary['top1_hits']}/{summary['n_evaluated']}", "map1": round(summary["map@1"], 4),
                     "map5": round(summary["map@5"], 4),
                     "sec_per_query": round(statistics.mean(s[key] for _, _, s in results), 2)})
        painting_rows += [{"descriptor": name, "metric": metric, "ratio": ratio, "cross_check": "on" if cross else "off",
                           "score": score, "painting": f"{r['query_id']:05d}.{r['painting_index']}",
                           "gt": r["ground_truth_id"], "rank_in_top10": r["correct_rank"]} for r in per]
    OUT.mkdir(parents=True, exist_ok=True)
    for path, data in ((OUT / f"metrics_{args.set}.csv", rows), (OUT / f"metrics_{args.set}_paintings.csv", painting_rows)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    figure(rows, args.set)
    best = max(rows, key=lambda r: (r["map1"], r["map5"], -r["sec_per_query"]))
    example(folder, corpus, gt, best, args.set)
    for r in sorted(rows, key=lambda r: (-r["map1"], -r["map5"], r["sec_per_query"]))[:5]:
        print(r)


def figure(rows: list[dict], name: str) -> None:
    """mAP@1 per configuration: one panel per descriptor + metric, ratio x cross-check on x, one bar per score."""
    groups = [(d, m) for d, ms in METRICS.items() for m in ms]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4.6), sharey=True, facecolor="white")
    colors = {"matches": "#94a3b8", "inliers": "#2a78d6", "distinct": "#15803d"}
    settings = list(itertools.product(RATIOS, ("off", "on")))
    for ax, (d, m) in zip(axes, groups):
        for j, score in enumerate(SCORES):
            values = [next(r["map1"] for r in rows if (r["descriptor"], r["metric"], r["ratio"], r["cross_check"], r["score"])
                           == (d, m, ratio, cross, score)) for ratio, cross in settings]
            ax.bar(np.arange(len(settings)) + (j - 1) * 0.27, values, 0.27, color=colors[score], label=score)
        ax.set_xticks(range(len(settings)), [f"{r}\n{'x' if c == 'on' else '-'}" for r, c in settings], fontsize=9)
        ax.set_title(f"{d} · {m}", loc="left")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#e5e7eb")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"mAP@1 on {name.upper().replace('_', '-')}")
    axes[0].legend(title="score", frameon=False, fontsize=9)
    fig.supxlabel("ratio threshold, cross-check (x = on)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"metrics_{name}.png", dpi=90)
    plt.close(fig)


def example(folder: Path, corpus: Path, gt: dict, best: dict, name: str) -> None:
    """A correct match of the best configuration, with its RANSAC inliers drawn (for the slides)."""
    method = METHODS[best["descriptor"]]
    database = museum_features(corpus, method)
    match = MatchConfig(ratio=best["ratio"], mutual=best["cross_check"] == "on", reprojection_px=MATCH.reprojection_px)
    for path in list_images(folder):
        q = get_image_id(path)
        if gt[q][0] == -1:
            continue
        gray, mask = query_paintings(load_image(path))[0]
        features = extract_features(gray, method.detector, method.descriptor, method.match, mask=mask)
        q_desc = prepare(features.descriptors, best["metric"])
        reference = database[gt[q][0]]
        matcher = cv2.BFMatcher(NORMS[best["metric"]])
        pairs = [(a.queryIdx, a.trainIdx) for a, b in (m for m in matcher.knnMatch(q_desc, prepare(reference.descriptors, best["metric"]), k=2) if len(m) == 2)
                 if a.distance < match.ratio * b.distance]
        if len(pairs) < 30:
            continue
        src, dst = features.points[[p for p, _ in pairs]], reference.points[[r for _, r in pairs]]
        homography, keep = cv2.findHomography(src, dst, cv2.RANSAC, MATCH.reprojection_px)
        inliers = [p for p, k in zip(pairs, keep.ravel()) if k] if homography is not None else []
        if len(inliers) < 30:
            continue
        museum = cv2.cvtColor(load_image(corpus / f"bbdd_{gt[q][0]:05d}.jpg"), cv2.COLOR_BGR2GRAY)
        h = 500
        sa, sb = h / gray.shape[0], h / museum.shape[0]
        a = cv2.resize(gray, (int(gray.shape[1] * sa), h))
        b = cv2.resize(museum, (int(museum.shape[1] * sb), h))
        fig, ax = plt.subplots(figsize=(13, 5.4), facecolor="white")
        ax.imshow(np.hstack([a, b]), cmap="gray")
        for qi, ri in inliers[:120]:
            (x1, y1), (x2, y2) = features.points[qi] * sa, reference.points[ri] * sb
            ax.plot([x1, x2 + a.shape[1]], [y1, y2], color="#16a34a", lw=0.8, alpha=0.8)
        ax.set_title(f"{name.upper().replace('_', '-')} {q:05d} (our crop) ↔ bbdd {gt[q][0]:05d}: {len(inliers)} RANSAC "
                     f"inliers of {len(pairs)} tentative matches ({best['descriptor']} {best['metric']}, ratio {best['ratio']})",
                     fontsize=11, loc="left")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"match_example_{name}.png", dpi=90)
        plt.close(fig)
        return


if __name__ == "__main__":
    main()
