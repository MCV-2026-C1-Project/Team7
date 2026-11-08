"""Week 4, Task 2 (II): robustness of the unknown decision (local threshold × global fallback ratio).

Read-only study of the submitted Week 4 method. Per query the method runs once (regions, recovery, local scores);
the global (Week 3) candidate and its best / second distance ratio are computed for every final region, so any
(threshold, ratio) pair can be applied afterwards with ``src.unknowns.decide``. Recovery does not depend on either
value. The (10, 0.85) decisions are asserted to equal the method's own answers.

Writes to --out: grid.csv (threshold × ratio per set), unknown_margins.csv (QSD1-W4 unknowns and the known
paintings below the threshold) and wrong_rejections.csv (QSD1-W3 / QSD2-W3 regions below the threshold).

The results in weeks/week4/results/robustness/ were produced on commit dc4924a, before the region filter;
--min-region-fraction 0 (the default) reproduces that setting, 0.10 runs the final method.

Usage (from the repository root):
    python weeks/week4/experiments/robustness_study.py [--sets qsd1_w4 qsd1_w3 qsd2_w3] [--data data]
        [--out weeks/week4/results/robustness] [--min-region-fraction 0]
"""
import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import main as cli  # noqa: E402
import src.segmentation as segmentation  # noqa: E402
from src.data_loading import get_image_id, list_images, load_ground_truth, load_image, resolve_dataset  # noqa: E402
from src.evaluate import evaluate_local  # noqa: E402
from src.local_pipeline import (LocalMethod, global_candidates, global_database, matching_database,  # noqa: E402
                                process_query)
from src.unknowns import UNKNOWN, decide  # noqa: E402

THRESHOLDS = [6, 8, 10, 12, 15]
RATIOS = [0.80, 0.85, 0.90]
THRESHOLD, RATIO = 10, 0.85  # submitted values (src/config.py)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def r4(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def study(name: str, data: Path) -> tuple[list[dict], list[dict]]:
    """Run the method once per query of one set; return the per-region rows and the threshold × ratio grid."""
    args = cli.parse_args(["evaluate", "--week", "4", "--set", name])
    method = replace(LocalMethod(), unknown_threshold=args.threshold, max_paintings=args.max_paintings)
    corpus = resolve_dataset(data, "BBDD")
    database, global_db = matching_database(corpus, method), global_database(corpus)
    queries = resolve_dataset(data, name)
    gt = load_ground_truth(queries)
    start, rows, reference = time.perf_counter(), [], {}
    for path in list_images(queries):
        q, image = get_image_id(path), load_image(path)
        result = process_query(image, database, method, global_database=global_db, segment=args.segment)
        reference[q] = result["answers"]
        candidates = global_candidates(image, global_db, args.segment, paintings=result["paintings"],
                                       max_paintings=method.max_paintings)
        for i, (ranking, scores) in enumerate(zip(result["rankings"], result["scores"])):
            rows.append({"query": q, "index": i, "gt": gt[q][i] if i < len(gt[q]) else None,
                         "local_ranking": ranking, "local_score": max(scores.values(), default=0),
                         "gt_score": scores.get(gt[q][i]) if i < len(gt[q]) and gt[q][i] != UNKNOWN else None,
                         "global_ranking": candidates[i][0], "global_ratio": candidates[i][1]})

    def answers(threshold: int, ratio: float) -> dict[int, list[list[int]]]:
        out: dict[int, list[list[int]]] = {}
        for r in rows:
            d = decide(r["local_ranking"], {0: r["local_score"]}, threshold, (r["global_ranking"], r["global_ratio"]),
                       global_ratio=ratio)
            out.setdefault(r["query"], []).append(d.ranking)
        return out

    assert answers(args.threshold, method.global_ratio) == reference, f"{name}: offline decisions differ"
    grid = []
    for t in THRESHOLDS:
        for g in RATIOS:
            s = evaluate_local(answers(t, g), gt, max_paintings=3)
            grid.append({"set": name, "threshold": t, "ratio": g, "n_evaluated": s["n_evaluated"],
                         "top1_hits": s["top1_hits"], "map@1": r4(s["map@1"]), "map@5": r4(s["map@5"]),
                         "unknown_precision": r4(s["unknown_precision"]), "unknown_recall": r4(s["unknown_recall"]),
                         "unknown_f1": r4(s["unknown_f1"])})
    print(f"{name}: {len(rows)} regions, {time.perf_counter() - start:.0f} s", flush=True)
    return rows, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["qsd1_w4", "qsd1_w3", "qsd2_w3"], help="development sets")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder with BBDD and the query sets")
    parser.add_argument("--out", type=Path, default=ROOT / "weeks/week4/results/robustness", help="output folder")
    parser.add_argument("--min-region-fraction", type=float, default=0.0,
                        help="region filter: 0 = the study setting (dc4924a, before the filter), 0.10 = final method")
    args = parser.parse_args()
    segmentation.MIN_REGION_FRACTION = args.min_region_fraction  # read by remove_background at call time
    grid, margins, rejections = [], [], []
    for name in args.sets:
        rows, set_grid = study(name, args.data)
        grid += set_grid
        for r in rows:
            d = decide(r["local_ranking"], {0: r["local_score"]}, THRESHOLD, (r["global_ranking"], r["global_ratio"]),
                       global_ratio=RATIO)
            gt, answer = r["gt"], d.ranking[0]
            row = {"set": name, "query": f"{r['query']:05d}", "painting_index": r["index"],
                   "ground_truth": "" if gt is None else gt, "local_top_id": r["local_ranking"][0],
                   "local_score": r["local_score"],
                   "ground_truth_local_score": "" if r["gt_score"] is None else r["gt_score"],
                   "global_top_id": r["global_ranking"][0], "global_ratio": r4(r["global_ratio"]),
                   "answer_10_085": answer, "decision": d.reason, "correct": answer == gt}
            if name == "qsd1_w4" and (gt == UNKNOWN or r["local_score"] < THRESHOLD):
                # unknowns: how far from a wrong acceptance; knowns below the threshold: how far from being lost
                row["score_margin_to_threshold"] = THRESHOLD - r["local_score"]
                row["ratio_margin_to_0.85"] = r4(r["global_ratio"] - RATIO)
                margins.append(row)
            if name != "qsd1_w4" and r["local_score"] < THRESHOLD:
                row["wrong_rejection"] = gt is not None and gt != UNKNOWN and answer == UNKNOWN
                rejections.append(row)
    write_csv(args.out / "grid.csv", grid)
    if margins:
        write_csv(args.out / "unknown_margins.csv",
                  sorted(margins, key=lambda x: (x["ground_truth"] != UNKNOWN, x["query"])))
    if rejections:
        write_csv(args.out / "wrong_rejections.csv", rejections)


if __name__ == "__main__":
    main()
