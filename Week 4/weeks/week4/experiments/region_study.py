"""Week 4: region selection study (the origin of MIN_REGION_FRACTION).

Read-only. Only the region selection of the Week 4 method is replaced (``src.local_pipeline._query_regions`` is
patched in this process); matching, recovery and the unknown decision are unchanged. No matching score filters
regions, so unknown paintings must still give [-1].

Options:
  a        MAX_PAINTINGS = 3
  b        MAX_PAINTINGS = 2
  c5/10/20 MAX_PAINTINGS = 3, drop regions with area < X % of the largest region in the photo
  d10/20   MAX_PAINTINGS = 5 (safety net, also the recovery budget); keep a region only if area >= X % of the
           largest AND a convex 4-corner fit of its mask has aspect (minAreaRect long / short) <= 6 and IoU with
           the mask >= SEGMENTATION["min_iou"] (0.6)

Writes to --out: options.csv (options × sets) and rejected_regions.csv (every region rejected by c and d).

The results in weeks/week4/results/regions/ were produced on commit dc4924a, before the region filter;
--min-region-fraction 0 (the default) reproduces that baseline, so option (a) is the method of that commit.

Usage (from the repository root):
    python weeks/week4/experiments/region_study.py [--sets qsd1_w4 qsd2_w3 qsd1_w3 qsd2_w2] [--data data]
        [--out weeks/week4/results/regions] [--min-region-fraction 0]
"""
import argparse
import csv
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import main as cli  # noqa: E402
import src.local_pipeline as local_pipeline  # noqa: E402
import src.segmentation as segmentation  # noqa: E402
from src.config import SEGMENTATION  # noqa: E402
from src.data_loading import (get_image_id, list_images, load_ground_truth, load_image, load_mask,  # noqa: E402
                               resolve_dataset)
from src.evaluate import evaluate_local, evaluate_mask  # noqa: E402
from src.local_pipeline import LocalMethod, global_database, matching_database, process_query  # noqa: E402

MAX_ASPECT = 6.0  # BBDD long / short side ranges 1.00-5.81
OPTIONS = {"a": (3, None, False), "b": (2, None, False), "c5": (3, 5, False), "c10": (3, 10, False),
           "c20": (3, 20, False), "d10": (5, 10, True), "d20": (5, 20, True)}  # cap, area %, quad check
LABEL = {"a": "(a) MAX_PAINTINGS 3", "b": "(b) MAX_PAINTINGS 2", "c5": "(c) area >= 5 % of largest, cap 3",
         "c10": "(c) area >= 10 % of largest, cap 3", "c20": "(c) area >= 20 % of largest, cap 3",
         "d10": "(d) area >= 10 % + quad check, cap 5", "d20": "(d) area >= 20 % + quad check, cap 5"}
state: dict = {"option": "a", "query": None, "log": []}


def mask_quad(mask: np.ndarray) -> dict:
    """Convex 4-corner fit of a region mask: convex hull -> approxPolyDP with growing tolerance -> 4 corners,
    else the minAreaRect box. Returns convexity, minAreaRect aspect (long / short) and IoU(quad, mask)."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = None
    for eps in np.arange(0.01, 0.2, 0.005):
        poly = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(poly) == 4:
            quad = poly
            break
        if len(poly) < 4:
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.int32).reshape(-1, 1, 2)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(filled, [quad], 255)
    inter, union = np.count_nonzero((filled > 0) & (mask > 0)), np.count_nonzero((filled > 0) | (mask > 0))
    (_, _), (w, h), _ = cv2.minAreaRect(quad)
    return {"convex": bool(cv2.isContourConvex(quad)), "aspect": float(max(w, h) / max(min(w, h), 1e-6)),
            "iou": float(inter / union) if union else 0.0}


def query_regions(image: np.ndarray, segment: bool, max_paintings: int) -> list:
    """Replacement of src.local_pipeline._query_regions for the current option; logs every region it rejects."""
    cap, area_pct, quad_check = OPTIONS[state["option"]]
    assert max_paintings == cap
    regions = segmentation.remove_background(image, max_paintings=cap)[1] if segment else []
    areas = [int(np.count_nonzero(r["mask"])) for r in regions]
    largest = max(areas, default=1)
    kept = []
    for r, area in zip(regions, areas):
        reasons = []
        if area_pct is not None and area < area_pct / 100 * largest:
            reasons.append(f"area {100 * area / largest:.1f}% < {area_pct}%")
        shape = mask_quad(r["mask"]) if quad_check else None
        if shape is not None:
            if not shape["convex"]:
                reasons.append("quad not convex")
            if shape["aspect"] > MAX_ASPECT:
                reasons.append(f"aspect {shape['aspect']:.2f} > {MAX_ASPECT}")
            if shape["iou"] < SEGMENTATION["min_iou"]:
                reasons.append(f"quad IoU {shape['iou']:.2f} < {SEGMENTATION['min_iou']}")
        if reasons:
            x0, y0, x1, y1 = r["bbox"]
            state["log"].append({"option": state["option"], "query": state["query"], "bbox": [x0, y0, x1, y1],
                                 "size": f"{x1 - x0}x{y1 - y0}", "area": area,
                                 "area_pct": round(100 * area / largest, 1), "shape": shape,
                                 "reason": "; ".join(reasons)})
        else:
            kept.append(r)
    height, width = image.shape[:2]
    return segmentation.order_paintings(kept) or [
        {"mask": np.full((height, width), 255, np.uint8), "bbox": (0, 0, width, height)}]


def study(name: str, data: Path) -> tuple[list[dict], list[dict]]:
    """Run every option on one set; return the options rows and the rejected-region rows."""
    corpus = resolve_dataset(data, "BBDD")
    args = cli.parse_args(["evaluate", "--week", "4", "--set", name])
    base = replace(LocalMethod(), unknown_threshold=args.threshold)
    database, global_db = matching_database(corpus, base), global_database(corpus)
    queries = resolve_dataset(data, name)
    gt = load_ground_truth(queries)
    paths = list_images(queries)
    n_gt = sum(len(v) for v in gt.values())
    cache, table, start = {}, [], time.perf_counter()
    state["log"] = []
    for option, (cap, _, _) in OPTIONS.items():
        state["option"] = option
        method = replace(base, max_paintings=cap)
        answers, mask_f1 = {}, []
        for path in paths:
            q = get_image_id(path)
            state["query"] = q
            image = load_image(path)
            regions = query_regions(image, args.segment, cap)
            # Same regions and cap -> same result: reuse it (process_query calls query_regions again; drop its log).
            key = (q, cap if args.segment else 0, tuple(hashlib.sha1(r["mask"].tobytes()).hexdigest() for r in regions))
            if key not in cache:
                n = len(state["log"])
                cache[key] = process_query(image, database, method, global_database=global_db, segment=args.segment)
                del state["log"][n:]
            answers[q] = cache[key]["answers"]
            if args.segment and path.with_suffix(".png").is_file():
                union = np.zeros(image.shape[:2], np.uint8)
                for r in regions:
                    union |= r["mask"]
                mask_f1.append(evaluate_mask(union, load_mask(path.with_suffix(".png")))[2])
        s = evaluate_local(answers, gt, max_paintings=max(3, cap))
        table.append({"option": option, "description": LABEL[option], "set": name,
                      "regions": sum(len(v) for v in answers.values()), "gt_paintings": n_gt,
                      "top1_hits": s["top1_hits"], "n_evaluated": s["n_evaluated"], "map@1": round(s["map@1"], 4),
                      "map@5": round(s["map@5"], 4), "mask_f1": round(float(np.mean(mask_f1)), 4) if mask_f1 else None,
                      "unknown_f1": round(s["unknown_f1"], 4) if name == "qsd1_w4" else ""})
        print(f"{name} {option}: top1 {s['top1_hits']}/{s['n_evaluated']}, mAP@1 {s['map@1']:.3f}, "
              f"{time.perf_counter() - start:.0f} s", flush=True)
    seen, rejected = set(), []
    for row in state["log"]:
        key = (row["option"], row["query"], tuple(row["bbox"]))
        if key in seen:
            continue
        seen.add(key)
        rejected.append({"option": row["option"], "set": name, "query": f"{row['query']:05d}",
                         "bbox_x0_y0_x1_y1": " ".join(map(str, row["bbox"])), "size_px": row["size"],
                         "area_px": row["area"], "area_pct_of_largest": row["area_pct"],
                         "quad_iou": round(row["shape"]["iou"], 4) if row["shape"] else "",
                         "reason": row["reason"], "ground_truth": " ".join(map(str, gt[row["query"]]))})
    return table, rejected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["qsd1_w4", "qsd2_w3", "qsd1_w3", "qsd2_w2"],
                        help="development sets")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder with BBDD and the query sets")
    parser.add_argument("--out", type=Path, default=ROOT / "weeks/week4/results/regions", help="output folder")
    parser.add_argument("--min-region-fraction", type=float, default=0.0,
                        help="region filter inside remove_background: 0 = the study baseline (dc4924a), 0.10 = final")
    args = parser.parse_args()
    segmentation.MIN_REGION_FRACTION = args.min_region_fraction  # read by remove_background at call time
    local_pipeline._query_regions = query_regions  # process_query looks the function up at call time
    table, rejected = [], []
    for name in args.sets:
        set_table, set_rejected = study(name, args.data)
        table += set_table
        rejected += set_rejected
    write_csv(args.out / "options.csv", table)
    if rejected:
        write_csv(args.out / "rejected_regions.csv", rejected)


if __name__ == "__main__":
    main()
