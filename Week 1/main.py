"""Retrieve paintings for QSD1 evaluation or QST1 submission.

Query directories contain JPG images and optional ground-truth correspondences.
Use --output to export validated Top-10 rankings.

Usage:
    python main.py
    python main.py --color-spaces HSV Lab --bins 16
    python main.py --method method1 --top-k-retrieval 10 --top-k-evaluation 5
    python main.py --method method1 --query_data data/qst1_w1 --output results/week1/QST1/method1/result.pkl
    python main.py --method method2 --query_data data/qst1_w1 --output results/week1/QST1/method2/result.pkl
"""
import argparse
from pathlib import Path
from typing import Any

from src.data_loading import has_ground_truth, list_images, load_ground_truth
from src.descriptors import COLOR_SPACES, compute_descriptors
from src.evaluate import evaluate_retrieval
from src.retrieval import format_results_to_lists, retrieve
from src.similarities import SIMILARITY_METRICS
from src.submission import SUBMISSION_K, save_results


PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"

DEFAULT_CORPUS_DATA_DIR: Path = DEFAULT_DATA_DIR / "BBDD"
DEFAULT_QUERY_DATA_DIR: Path = DEFAULT_DATA_DIR / "qsd1_w1"

# The two submitted methods (chosen by a step-by-step search on QSD1). Both are global 1D
# histograms of three colour spaces after gray-world and a contrast stretch of the brightness channel.
# QSD1: method1 27/30, mAP@1 0.9000, mAP@5 0.9500; method2 27/30, mAP@1 0.9000, mAP@5 0.9167.
METHODS: dict[str, dict[str, Any]] = {
    "method1": {"color_spaces": ["HSV", "Lab", "rgb"], "bins": 32, "similarity": "hellinger_kernel",
                "gray_world": True, "contrast_stretch": True, "luminance_weight": 0.25},
    "method2": {"color_spaces": ["HSV", "YCbCr", "rgb"], "bins": 32, "similarity": "chi2",
                "gray_world": True, "contrast_stretch": True, "luminance_weight": 1.0},
}


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {n}")
    return n


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and validate retrieval, evaluation, and export K."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve paintings using global color histograms and evaluate "
            "matches when ground truth is available; optionally export Top-10 rankings."
        ),
    )
    parser.add_argument("--corpus_data", type=Path, default=DEFAULT_CORPUS_DATA_DIR,
                        help="folder containing the corpus JPG images (e.g. BBDD/)")
    parser.add_argument("--query_data", type=Path, default=DEFAULT_QUERY_DATA_DIR,
                        help="folder containing query JPGs and optional ground truth (QSD1 or QST1)")
    parser.add_argument("--color-spaces", nargs="+", default=["HSV", "Lab"], choices=COLOR_SPACES)
    parser.add_argument("--bins", type=positive_int, default=16)
    parser.add_argument("--similarity", default="l1", choices=tuple(SIMILARITY_METRICS))
    parser.add_argument("--gray-world", action="store_true", help="apply gray-world colour correction")
    parser.add_argument("--contrast-stretch", action="store_true",
                        help="stretch the brightness channel to its 1st-99th percentiles")
    parser.add_argument("--luminance-weight", type=float, default=1.0,
                        help="weight of the brightness histograms (default: 1)")
    parser.add_argument(
        "--method", choices=tuple(METHODS),
        help="use a submitted method; overrides colour spaces, bins, similarity and corrections",
    )
    parser.add_argument(
        "--top-k-retrieval", type=positive_int,
        help="number of most similar images to retrieve per query (default: no limit)",
    )
    parser.add_argument(
        "--top-k-evaluation", type=positive_int,
        help="number of most similar images to consider for evaluation (default: no limit)",
    )
    parser.add_argument(
        "--output", type=Path,
        help=f"save result.pkl (always with {SUBMISSION_K} ids per query)",
    )

    args = parser.parse_args()

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

    # Only compare when both were provided
    if (args.top_k_retrieval is not None
            and args.top_k_evaluation is not None
            and args.top_k_evaluation > args.top_k_retrieval):
        parser.error("--top-k-evaluation cannot be greater than --top-k-retrieval")

    return args


def main() -> None:
    args = parse_args()
    db_paths = list_images(args.corpus_data)
    query_paths = list_images(args.query_data)
    ground_truth = (
        load_ground_truth(args.query_data)
        if has_ground_truth(args.query_data)
        else None
    )

    # The same descriptor (and the same corrections) for the database and the queries.
    options = {"gray_world_on": args.gray_world, "stretch": args.contrast_stretch,
               "luminance_weight": args.luminance_weight}
    db_descriptors = compute_descriptors(db_paths, args.color_spaces, args.bins, **options)
    query_descriptors = compute_descriptors(query_paths, args.color_spaces, args.bins, **options)

    retrieved_results = retrieve(
        query_descriptors=query_descriptors,
        db_descriptors=db_descriptors,
        top_k=args.top_k_retrieval,
        similarity_function=args.similarity,
    )

    formatted_query, formatted_results = format_results_to_lists(retrieved_results)

    db_descriptor_length = next(iter(db_descriptors.values())).size
    query_descriptor_length = next(iter(query_descriptors.values())).size

    corrections = [name for name, on in (("gray-world", args.gray_world), ("contrast stretch", args.contrast_stretch)) if on]
    print(f"Method: {args.method or 'custom'}")
    print(f"Descriptor: {' + '.join(args.color_spaces)}, {args.bins} bins per channel, "
          f"corrections: {', '.join(corrections) or 'none'}, brightness weight {args.luminance_weight:g}")
    print(f"BBDD:    {len(db_descriptors)} images -> {db_descriptor_length} values each")
    print(f"Queries: {len(query_descriptors)} images -> {query_descriptor_length} values each")
    print(f"Similarity function: {args.similarity}")
    # Without a limit, retrieval ranks the whole database and evaluation uses every retrieved image.
    k_retrieval = args.top_k_retrieval or len(db_descriptors)
    k_evaluation = args.top_k_evaluation or k_retrieval
    print(f"Top-{k_retrieval} retrieval results:")
    if ground_truth is not None:
        evaluation_results = evaluate_retrieval(
            retrieved_results, ground_truth, top_k=k_evaluation,
        )
        print(f"MAP@{k_evaluation}: {evaluation_results:.4f}")
    else:
        print("No ground truth found; evaluation skipped.")
    print("Query IDs:", formatted_query)
    print("Retrieved IDs:", formatted_results)

    if args.output is not None:
        save_results(
            formatted_results,
            args.output,
            n_queries=len(query_paths),
            valid_db_ids=set(db_descriptors),
        )
        print(f"Submission saved to: {args.output}")


if __name__ == "__main__":
    main()
