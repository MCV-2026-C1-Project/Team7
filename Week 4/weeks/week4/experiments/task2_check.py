"""Compare homography plausibility on the same initially segmented painting regions.

The experimental check rejects homographies with an affine determinant outside
[0.1, 10], perspective denominators more than 0.5 from 1, or a non-convex projected
outline. It is not part of the submitted method. Both variants reuse the same
features and RANSAC result; matching-based region recovery is excluded from this
comparison so the input regions stay identical.

Writes weeks/week4/results/task2/homography_check.csv. Run from the repository root:
    python weeks/week4/experiments/task2_check.py [--workers 8]
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data_loading import get_image_id, list_images, load_ground_truth, load_image, resolve_dataset  # noqa: E402
from src.evaluate import evaluate_retrieval  # noqa: E402
from src.local_pipeline import CACHE_DIR, LocalMethod, matching_database, query_paintings  # noqa: E402
from src.matching import Features, extract_features, match_database, prepare_descriptors, rank_scores  # noqa: E402

OUT = ROOT / "weeks" / "week4" / "results" / "task2"
METHOD = LocalMethod()
VARIANTS = ("without check", "with check")
_STATE: dict = {}


def plausible_homography(homography: np.ndarray, shape: tuple[int, int]) -> bool:
    """Experimental limits on scale, perspective and convexity of a projection."""
    h = homography / homography[2, 2] if abs(homography[2, 2]) > 1e-12 else homography
    if not 0.1 <= float(np.linalg.det(h[:2, :2])) <= 10.0:
        return False
    height, width = shape
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    denominators = corners @ h[2, :2] + h[2, 2]
    if np.any(np.abs(denominators - 1) > 0.5):
        return False
    projected = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), h).reshape(-1, 2)
    return bool(cv2.isContourConvex(projected.astype(np.float32)))


def _init(corpus: str) -> None:
    _STATE["db"] = matching_database(Path(corpus), METHOD)


def _query(path: str) -> tuple[int, dict]:
    scores = {name: [] for name in VARIANTS}
    for gray, mask in query_paintings(load_image(Path(path)), max_paintings=METHOD.max_paintings):
        features = extract_features(gray, METHOD.detector, METHOD.descriptor, METHOD.match, mask=mask)
        features = Features(features.points, prepare_descriptors(features.descriptors, METHOD.match.metric), features.shape)
        matches = match_database(features, _STATE["db"], METHOD.match)
        scores["without check"].append({i: match.score for i, match in matches.items()})
        scores["with check"].append({
            i: match.score if match.homography is not None and plausible_homography(match.homography, gray.shape) else 0
            for i, match in matches.items()
        })
    return get_image_id(Path(path)), scores


def scores_for(name: str, data: Path, workers: int) -> dict:
    """Cache both variants together, including this study's source fingerprint."""
    queries, corpus = resolve_dataset(data, name), resolve_dataset(data, "BBDD")
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache = CACHE_DIR.parent / f"homography_{METHOD.score_cache_key(queries, corpus)}_{source}.json"
    if cache.is_file():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {variant: {int(q): [{int(i): score for i, score in s.items()} for s in paintings]
                          for q, paintings in values.items()} for variant, values in raw.items()}
    paths = [str(path) for path in list_images(queries)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(corpus),)) as pool:
        results = list(pool.map(_query, paths))
    scores = {variant: {q: values[variant] for q, values in results} for variant in VARIANTS}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scores), encoding="utf-8")
    return scores


def summary(scores, gt) -> dict:
    """Known-painting retrieval; unknown references are excluded from these means."""
    _, rows = evaluate_retrieval({q: [rank_scores(s, 10) for s in paintings] for q, paintings in scores.items()}, gt)
    known = [row for row in rows if row["ground_truth_id"] not in (None, -1)]
    return {"top1_known": f"{sum(row['ap1'] == 1 for row in known)}/{len(known)}",
            "map1_known": round(statistics.mean(row["ap1"] for row in known), 4),
            "map5_known": round(statistics.mean(row["ap5"] for row in known), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()
    rows = []
    for name in ("qsd1_w4", "qsd2_w3"):
        gt = load_ground_truth(resolve_dataset(args.data, name))
        variants = scores_for(name, args.data, args.workers)
        for variant, scores in variants.items():
            rows.append({"set": name, "variant": variant, **summary(scores, gt)})
            print(rows[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "homography_check.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
