"""Run the two Week 2 submissions or the original Week 1 retrieval methods.

Week 2 delivery method1: --method method1 (HSV + Lab + rgb, 4x4, 16 bins, L1).
Week 2 delivery method2: --method method2 (HSV + YCbCr + rgb, 4x4, 32 bins, L1).
Use --background none for cropped QS1 images and --background remove for QS2;
both methods share the same segmentation.

Use --evaluate only for development metrics. --output saves Top-10 rankings in
sorted query-filename order and, for QS2, native-size binary PNG masks beside
the PKL. Test inference needs no annotations.

Week 2 test exports:
    python main.py --method method1 --background none --query_data data/qst1_w2 --output results/week2/QST1/method1/result.pkl
    python main.py --method method1 --background remove --query_data data/qst2_w2 --output results/week2/QST2/method1/result.pkl
    python main.py --method method2 --background none --query_data data/qst1_w2 --output results/week2/QST1/method2/result.pkl
    python main.py --method method2 --background remove --query_data data/qst2_w2 --output results/week2/QST2/method2/result.pkl

Original Week 1 methods are named week1_method1 / week1_method2.
The non-submitted Week 2 comparison is named week2_hellinger.
"""
import argparse
import json
from pathlib import Path

from src.config import METHODS, WEEK2_METHODS
from src.descriptors import COLOR_SPACES
from src.evaluate import evaluate_pipeline
from src.pipeline import run_pipeline
from src.similarities import SIMILARITY_METRICS
from src.submission import SUBMISSION_K, save_submission


PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"

DEFAULT_CORPUS_DATA_DIR: Path = DEFAULT_DATA_DIR / "BBDD"
DEFAULT_QUERY_DATA_DIR: Path = DEFAULT_DATA_DIR / "qsd1_w1"


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {n}")
    return n


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and validate the inference mode and export depth."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the two Week 2 retrieval methods with optional color-based segmentation, "
            "or the original Week 1 methods. Use --evaluate for development metrics."
        ),
    )
    parser.add_argument("--corpus_data", type=Path, default=DEFAULT_CORPUS_DATA_DIR,
                        help="folder containing the corpus JPG images (e.g. BBDD/)")
    parser.add_argument("--query_data", type=Path,
                        help="query JPGs and optional ground truth (default: qsd1_w1; Week 2: qsd1_w2 for none, qsd2_w2 for remove)")
    parser.add_argument("--color-spaces", nargs="+", default=["HSV", "Lab"], choices=COLOR_SPACES)
    parser.add_argument("--bins", type=positive_int, default=16)
    parser.add_argument("--similarity", default="l1", choices=tuple(SIMILARITY_METRICS))
    parser.add_argument("--gray-world", action="store_true", help="apply gray-world colour correction")
    parser.add_argument("--contrast-stretch", action="store_true",
                        help="stretch the brightness channel to its 1st-99th percentiles")
    parser.add_argument("--luminance-weight", type=float, default=1.0,
                        help="weight of the brightness histograms (default: 1)")
    parser.add_argument(
        "--method", choices=tuple(WEEK2_METHODS) + tuple(METHODS),
        help=("Week 2 delivery: method1 (Lab/L1), method2 (YCbCr/L1); "
              "week2_hellinger is the non-submitted comparison. "
              "Week 1 baselines: week1_method1, week1_method2. "
              "Named presets override the descriptor options"),
    )
    parser.add_argument("--background", choices=("none", "remove"),
                        help="required for Week 2: none for cropped QS1; remove for QS2")
    parser.add_argument("--evaluate", action="store_true",
                        help="read development annotations and print mAP@1/5/10 and available mask metrics")
    parser.add_argument(
        "--top-k-retrieval", type=positive_int,
        help="Week 1/custom retrieval depth (default: all, or 10 with --output); omit for Week 2",
    )
    parser.add_argument(
        "--output", type=Path,
        help=f"destination PKL ({SUBMISSION_K} IDs per query); QS2 also saves PNG masks in the same directory",
    )

    args = parser.parse_args()

    if args.method in WEEK2_METHODS:
        if args.background is None:
            parser.error("Week 2 requires --background none (QS1) or remove (QS2)")
        if args.top_k_retrieval is not None:
            parser.error("Week 2 evaluates mAP@1/5/10 and exports Top-10; omit the K options")
        if args.query_data is None:
            args.query_data = DEFAULT_DATA_DIR / ("qsd2_w2" if args.background == "remove" else "qsd1_w2")
        return args
    if args.background is not None:
        parser.error("--background requires a Week 2 method")
    if args.query_data is None:
        args.query_data = DEFAULT_QUERY_DATA_DIR

    if args.method is not None:
        method = METHODS[args.method]
        args.color_spaces, args.bins, args.similarity = method["color_spaces"], method["bins"], method["similarity"]
        args.gray_world, args.contrast_stretch = method["gray_world"], method["contrast_stretch"]
        args.luminance_weight = method["luminance_weight"]

    # The submission file needs exactly SUBMISSION_K (10) ids per query.
    if args.output is not None:
        if args.top_k_retrieval is None:
            args.top_k_retrieval = SUBMISSION_K
        elif args.top_k_retrieval != SUBMISSION_K:
            parser.error(f"--output needs --top-k-retrieval {SUBMISSION_K} (got {args.top_k_retrieval})")

    return args


def main() -> None:
    args = parse_args()
    week2 = args.method in WEEK2_METHODS
    descriptor = None if week2 or args.method in METHODS else {
        "kind": "1d", "grid": 1, "color_spaces": args.color_spaces, "bins": args.bins,
        "gray_world": args.gray_world, "contrast_stretch": args.contrast_stretch,
        "luminance_weight": args.luminance_weight,
    }
    predictions = run_pipeline(
        args.corpus_data, args.query_data, method=args.method or "custom",
        segment=week2 and args.background == "remove", descriptor=descriptor,
        similarity=None if week2 else args.similarity, top_k=args.top_k_retrieval,
    )
    summary = predictions["metadata"]
    if args.evaluate:
        summary, _, _ = evaluate_pipeline(predictions)
    if args.output is not None:
        save_submission(predictions["rankings"], predictions["query_paths"], args.output,
                        valid_db_ids=set(predictions["database"]), masks=predictions["masks"] or None)
        print(f"Submission saved to: {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
