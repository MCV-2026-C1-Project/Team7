"""Week 4, Task 2 (I) sanity run: one local-feature configuration on a development set.

DoG + SIFT, L2, Lowe ratio 0.75 (no cross-check), RANSAC homography (5 px), score = inliers; shared preprocessing per
painting (src/local_pipeline.py). Prints mAP@1 / mAP@5 over all annotated paintings (an unknown painting, gt -1,
always scores 0 here: Task 2 II adds the [-1] answer) and over the known paintings only, the number of unknown
paintings and the time per query.

Usage (from the repository root): python weeks/week4/experiments/task2_sanity.py [--set qsd1_w4] [--data data]
"""
import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data_loading import load_ground_truth, resolve_dataset  # noqa: E402
from src.evaluate import evaluate_retrieval  # noqa: E402
from src.keypoints import DoG  # noqa: E402
from src.local_descriptors import SIFTConfig  # noqa: E402
from src.local_pipeline import LocalMethod, run  # noqa: E402
from src.matching import MatchConfig  # noqa: E402

# The original sanity setting scores only the segmented regions (as task2_metrics.py does). The merged-painting
# recovery added later would turn leftover keypoints into extra paintings, so it is switched off here: no score can
# reach this support.
NO_RECOVERY = 10**9
METHOD = LocalMethod(DoG(), SIFTConfig(), MatchConfig(ratio=0.75, mutual=False, reprojection_px=5.0),
                     recovery_min_inliers=NO_RECOVERY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="qsd1_w4")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--segment", action=argparse.BooleanOptionalAction, default=True,
                        help="background removal (up to three paintings); --no-segment: whole image")
    args = parser.parse_args()
    queries = resolve_dataset(args.data, args.set)
    gt = load_ground_truth(queries)
    out = run(queries, resolve_dataset(args.data, "BBDD"), METHOD, segment=args.segment)
    summary, rows = evaluate_retrieval(out["rankings"], gt)
    known = [r for r in rows if r["ground_truth_id"] not in (None, -1)]
    n_unknown = sum(i == -1 for ids in gt.values() for i in ids)
    found = sum(len(r) for r in out["rankings"].values())
    print(f"{args.set}: {len(gt)} queries, {sum(map(len, gt.values()))} annotated paintings "
          f"({n_unknown} unknown, gt -1), {found} paintings found; database {out['n_database']} images")
    print(f"all paintings:   top-1 {summary['top1_hits']}/{summary['n_evaluated']}  "
          f"mAP@1 {summary['map@1']:.4f}  mAP@5 {summary['map@5']:.4f}")
    if known:
        print(f"known paintings: top-1 {sum(r['ap1'] == 1 for r in known)}/{len(known)}  "
              f"mAP@1 {statistics.mean(r['ap1'] for r in known):.4f}  mAP@5 {statistics.mean(r['ap5'] for r in known):.4f}")
    s = list(out["seconds"].values())
    print(f"time per query: mean {statistics.mean(s):.2f} s, median {statistics.median(s):.2f} s, max {max(s):.2f} s "
          "(museum features from the cache)")
    misses = [(r["query_id"], r["painting_index"], r["ground_truth_id"], r["correct_rank"]) for r in rows if r["ap1"] != 1]
    print("not at rank 1 (query, painting, gt, rank of gt):", misses)


if __name__ == "__main__":
    main()
