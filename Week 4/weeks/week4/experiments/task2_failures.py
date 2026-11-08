"""Week 4, Task 2 (I): why does the sanity configuration (DoG + SIFT, ratio 0.75, RANSAC) miss four QSD2-W3 paintings?

For each missed painting: augmentation, whether the noise filter fires on the crop, our mask against the GT mask
(F1), keypoints on the crop, inliers with the correct museum image and with the wrong top 1, and the rank of the
correct image when one thing changes at a time (no mask restriction, ratio 0.8, no noise filter, crop resized to
the median museum size). Nothing here changes the defaults.

Writes weeks/week4/results/task2/failures_qsd2w3.png (query crop | correct painting with its inlier matches | wrong
top 1 with its inlier matches) and degenerate_00005_qsd2w3.png (the many-to-one homography of 00005 without and
with cross-check). Usage (from the repository root): python weeks/week4/experiments/task2_failures.py
[--degenerate-only] [--data DATA]
"""
import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data_loading import list_images, load_ground_truth, load_image, load_mask, resolve_dataset  # noqa: E402
from src.denoise import noise_type  # noqa: E402
from src.descriptors import crop_to_mask  # noqa: E402
from src.evaluate import evaluate_mask  # noqa: E402
from src.local_pipeline import LocalMethod, museum_features  # noqa: E402
from src.matching import extract_features, rank_scores, score_database, tentative_matches  # noqa: E402
from src.segmentation import extract_paintings, remove_background  # noqa: E402

sys.path.insert(0, str(ROOT / "weeks" / "week4" / "experiments"))
from task2_sanity import METHOD  # noqa: E402

FAILURES = [(5, 0), (12, 0), (19, 0), (19, 1)]  # (query, painting) missed by the sanity run
OUT = ROOT / "weeks" / "week4" / "results" / "task2"


def left_to_right(paintings):
    return sorted(paintings, key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2)


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def describe_crop(crop, mask, method: LocalMethod, *, use_mask=True, denoise_on=True, target_side=None):
    """Features of a painting crop under one variant of the pipeline."""
    from src.denoise import denoise
    image = denoise(crop) if denoise_on else crop
    if target_side is not None:  # resize the crop (and mask) so its longest side is target_side
        s = target_side / max(image.shape[:2])
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        mask = None if mask is None else cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                                                    interpolation=cv2.INTER_NEAREST) > 0
    return extract_features(gray(image), method.detector, method.descriptor, method.match,
                            mask=mask if use_mask else None)


def inlier_pairs(query, reference, config):
    """Tentative pairs and the RANSAC inlier subset (for the figure)."""
    pairs = tentative_matches(query, reference, config)
    if len(pairs) < 4:
        return pairs, []
    src = query.points[[q for q, _ in pairs]]
    dst = reference.points[[r for _, r in pairs]]
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.reprojection_px)
    return pairs, [] if homography is None else [p for p, keep in zip(pairs, mask.ravel()) if keep]


def main(data: Path) -> None:
    folder = resolve_dataset(data, "qsd2_w3")
    bbdd_dir = resolve_dataset(data, "BBDD")
    gt = load_ground_truth(folder)
    with open(folder / "augmentations.pkl", "rb") as f:  # course annotation file
        aug = pickle.load(f)
    database = museum_features(bbdd_dir, METHOD)
    median_side = int(np.median([max(load_image(p).shape[:2]) for p in list_images(bbdd_dir)]))
    variants = {"default": {}, "no mask": {"use_mask": False}, "no noise filter": {"denoise_on": False},
                "crop at museum size": {"target_side": median_side}}
    rows, panels = [], []
    for q, i in FAILURES:
        image = load_image(folder / f"{q:05d}.jpg")
        ours = left_to_right(remove_background(image)[1])
        ref = left_to_right(extract_paintings(load_mask(folder / f"{q:05d}.png"))[1])
        crop, mask = crop_to_mask(image, ours[i]["mask"] > 0)
        correct = gt[q][i]
        row = {"painting": f"{q:05d}.{i}", "augmentation": aug[q], "noise_filter": noise_type(crop) or "clean",
               "mask_f1": evaluate_mask(ours[i]["mask"], ref[i]["mask"])[2] if i < len(ref) else None,
               "correct": correct}
        for name, kwargs in variants.items():
            features = describe_crop(crop, mask, METHOD, **kwargs)
            scores = score_database(features, database, METHOD.match)
            ranking = rank_scores(scores, len(scores))
            wrong = next(b for b in ranking if b != correct)
            row[name] = (ranking.index(correct) + 1, scores[correct], scores[wrong], len(features.points))
            if name == "default":
                row["top1"] = ranking[0]
                panels.append((q, i, gray(crop), features, correct, wrong))
        features = describe_crop(crop, mask, METHOD)
        ratio08 = replace(METHOD.match, ratio=0.8)
        s = score_database(features, database, ratio08)
        ranking = rank_scores(s, len(s))
        row["ratio 0.8"] = (ranking.index(correct) + 1, s[correct], s[next(b for b in ranking if b != correct)],
                            len(features.points))
        rows.append(row)
        print(row)
    print(f"median museum longest side: {median_side} px")
    figure(panels, database, bbdd_dir)


def match_panel(ax, crop, query, reference_image, reference, config, title, color):
    """Crop and museum image side by side, RANSAC inlier matches drawn as lines."""
    _, inliers = inlier_pairs(query, reference, config)
    h = 400
    sa, sb = h / crop.shape[0], h / reference_image.shape[0]
    a = cv2.resize(crop, (int(crop.shape[1] * sa), h))
    b = cv2.resize(gray(reference_image), (int(reference_image.shape[1] * sb), h))
    ax.imshow(np.hstack([a, b]), cmap="gray")
    for qi, ri in inliers[:80]:
        (x1, y1), (x2, y2) = query.points[qi] * sa, reference.points[ri] * sb
        ax.plot([x1, x2 + a.shape[1]], [y1, y2], color=color, lw=0.8, alpha=0.8)
    ax.set_title(f"{title}: {len(inliers)} inliers", fontsize=11)
    ax.axis("off")


def figure(panels, database, bbdd_dir) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(panels), 3, figsize=(18, 4.2 * len(panels)),
                             gridspec_kw={"width_ratios": [1, 2, 2]}, facecolor="white")
    for row, (q, i, crop, features, correct, wrong) in zip(axes, panels):
        row[0].imshow(crop, cmap="gray")
        row[0].set_title(f"QSD2-W3 {q:05d}, painting {i + 1}\n{len(features.points)} keypoints", fontsize=11)
        row[0].axis("off")
        for ax, bid, title, color in ((row[1], correct, f"correct bbdd {correct:05d}", "#15803d"),
                                      (row[2], wrong, f"top wrong bbdd {wrong:05d}", "#b91c1c")):
            match_panel(ax, crop, features, load_image(bbdd_dir / f"bbdd_{bid:05d}.jpg"), database[bid],
                        METHOD.match, title, color)
    fig.tight_layout()
    fig.savefig(OUT / "failures_qsd2w3.png", dpi=70)
    plt.close(fig)


def degenerate_figure(data: Path) -> None:
    """QSD2-W3 00005: inliers with the correct image (bbdd 94) and with the wrong top 1 (bbdd 224), without and
    with cross-check (sanity configuration otherwise: SIFT L2, ratio 0.75, RANSAC)."""
    folder, bbdd_dir = resolve_dataset(data, "qsd2_w3"), resolve_dataset(data, "BBDD")
    database = museum_features(bbdd_dir, METHOD)
    image = load_image(folder / "00005.jpg")
    crop, mask = crop_to_mask(image, left_to_right(remove_background(image)[1])[0]["mask"] > 0)
    features = describe_crop(crop, mask, METHOD)
    fig, axes = plt.subplots(2, 2, figsize=(18, 8.6), facecolor="white")
    for row, mutual in zip(axes, (False, True)):
        config = replace(METHOD.match, mutual=mutual)
        for ax, bid, color in ((row[0], 94, "#15803d"), (row[1], 224, "#b91c1c")):
            label = f"{'cross-check on' if mutual else 'cross-check off'}: bbdd {bid:05d} ({'correct' if bid == 94 else 'wrong'})"
            match_panel(ax, gray(crop), features, load_image(bbdd_dir / f"bbdd_{bid:05d}.jpg"), database[bid], config,
                        label, color)
    fig.suptitle("QSD2-W3 00005: without cross-check, 90 query points match one museum point and RANSAC accepts the "
                 "collapsed homography", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "degenerate_00005_qsd2w3.png", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder with BBDD and qsd2_w3")
    parser.add_argument("--degenerate-only", action="store_true", help="only the degenerate-homography figure")
    args = parser.parse_args()
    if not args.degenerate_only:
        main(args.data)
    degenerate_figure(args.data)
