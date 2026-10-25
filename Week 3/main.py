"""Museum painting retrieval, Week 3 (Team 7): command-line entry point.

The default is the final Week 3 method, `colour_lbp_dct_wavelet`: noise detection and filtering, background
removal with up to two paintings per image (QS2), and Week 2 colour histograms fused with LBP, DCT and wavelet
texture descriptors. The other presets of src/config.py stay selectable with --method (Week 2 methods, texture
alone, the teammates' fusion, colour + wavelet).

    python main.py evaluate --set qsd1_w3     # development: mAP@1/5/10 (QSD2 sets also mask P/R/F1)
    python main.py evaluate --set qsd2_w3
    python main.py predict                    # delivery Team7/week3 (QST1 + QST2), then its format check
    python main.py validate                   # format check of Team7/week3 only

The earlier form keeps working (one query folder at a time):
    python main.py --background none --evaluate                       # QSD1-W3
    python main.py --query_data data/qst2_w3 --output Team7/week3/QST2/method1/result.pkl
"""
import argparse
import json
import sys
from pathlib import Path

from src.config import FINAL_METHOD, PRESETS
from src.data_loading import resolve_dataset
from src.evaluate import evaluate_pipeline
from src.pipeline import run_pipeline
from src.submission import SUBMISSION_K, save_submission, validate_delivery

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DEFAULT_DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_DELIVERY: Path = PROJECT_ROOT / "Team7" / "week3"
COMMANDS = ("evaluate", "predict", "validate")
DEV_SETS = ("qsd1_w3", "qsd2_w3", "qsd1_w2", "qsd2_w2")


def _common(parser: argparse.ArgumentParser, method: bool = True) -> None:
    """--data, and --method unless the command runs no method."""
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder with BBDD/ and the query sets (nested course folders are found too)")
    if method:
        parser.add_argument("--method", choices=tuple(PRESETS), default=FINAL_METHOD,
                            help=f"preset of src/config.py (default: {FINAL_METHOD})")


def _command_parser() -> argparse.ArgumentParser:
    """Parser of the subcommands evaluate / predict / validate."""
    parser = argparse.ArgumentParser(description="Painting retrieval with one Top-10 ranking per painting.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="development set with annotations: retrieval and mask metrics")
    _common(evaluate)
    evaluate.add_argument("--set", choices=DEV_SETS, default="qsd2_w3",
                          help="QSD1 sets are cropped (no background removal); QSD2 sets are segmented")
    predict = commands.add_parser("predict", help="write the QST1 / QST2 delivery and validate it")
    _common(predict)
    predict.add_argument("--out", type=Path, default=DEFAULT_DELIVERY, help="delivery folder")
    validate = commands.add_parser("validate", help="check the delivery folder against the course format")
    _common(validate, method=False)
    validate.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    return parser


def _legacy_parser() -> argparse.ArgumentParser:
    """Parser of the earlier single-folder form (python main.py [--evaluate] [--output ...])."""
    parser = argparse.ArgumentParser(description="Painting retrieval with one Top-10 ranking per painting "
                                                 f"(or use a subcommand: {', '.join(COMMANDS)}).")
    _common(parser)
    parser.add_argument("--corpus_data", type=Path, help="database JPG images (default: <data>/BBDD)")
    parser.add_argument("--query_data", type=Path,
                        help="query JPGs (default: qsd2_w3, or qsd1_w3 with --background none)")
    parser.add_argument("--background", choices=("none", "remove"), default="remove",
                        help="remove (default): segment up to two paintings; none: queries already cropped")
    parser.add_argument("--evaluate", action="store_true",
                        help="read the development annotations and print mAP@1/5/10 and mask metrics")
    parser.add_argument("--output", type=Path,
                        help=f"destination PKL ({SUBMISSION_K} ids per painting); with background removal "
                             "the PNG masks are saved next to it")
    parser.add_argument("--levels", type=int, choices=(2, 3), default=3,
                        help="PKL nesting: 3 = [[[ids] per painting] per query]; 2 = [[ids] per query]")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a subcommand, or the earlier form (command None), and fill in the dataset defaults."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in COMMANDS:
        args = _command_parser().parse_args(argv)
        if args.command == "evaluate":
            args.background = "none" if args.set.startswith("qsd1") else "remove"
            args.query_data = resolve_dataset(args.data, args.set)
        return args
    args = _legacy_parser().parse_args(argv)
    args.command = None
    args.corpus_data = args.corpus_data or resolve_dataset(args.data, "BBDD")
    if args.query_data is not None:  # unchanged unless the images sit one folder deeper
        args.query_data = resolve_dataset(args.query_data.parent, args.query_data.name)
    else:
        args.query_data = resolve_dataset(args.data, "qsd2_w3" if args.background == "remove" else "qsd1_w3")
    return args


def run(corpus: Path, queries: Path, method: str, background: str) -> dict:
    """Predictions of a preset with Top-10 rankings (src.pipeline.run_pipeline)."""
    return run_pipeline(corpus, queries, method=method, segment=background == "remove", top_k=SUBMISSION_K)


def predict(data: Path, out: Path, method: str) -> None:
    """Write the QST1 and QST2 delivery and validate it.

    Both result.pkl have 3 levels: QST1/method1 [[[ids]], ...] (50 cropped queries, one painting each) and
    QST2/method1 [[[ids], [ids]], ...] (one list per painting, left to right) with <query>.png beside it (0/255
    mask of each query). No annotations are read.
    """
    bbdd = resolve_dataset(data, "BBDD")
    qst1 = run(bbdd, resolve_dataset(data, "qst1_w3"), method, "none")
    save_submission(qst1["rankings"], qst1["query_paths"], out / "QST1" / "method1" / "result.pkl",
                    valid_db_ids=set(qst1["database"]))
    qst2 = run(bbdd, resolve_dataset(data, "qst2_w3"), method, "remove")
    save_submission(qst2["rankings"], qst2["query_paths"], out / "QST2" / "method1" / "result.pkl",
                    valid_db_ids=set(qst2["database"]), masks=qst2["masks"])
    print(f"QST1: {len(qst1['rankings'])} queries; QST2: {len(qst2['rankings'])} queries, "
          f"{sum(len(r) for r in qst2['rankings'].values())} paintings; failed: "
          f"{qst1['metadata']['failed_query_ids'] + qst2['metadata']['failed_query_ids']}")
    validate_delivery(out, data)


def main() -> None:
    """Run the chosen command (or the earlier form: pipeline, optional evaluation and export)."""
    args = parse_args()
    if args.command == "validate":
        validate_delivery(args.delivery, args.data)
        return
    if args.command == "predict":
        predict(args.data, args.out, args.method)
        return
    corpus = resolve_dataset(args.data, "BBDD") if args.command else args.corpus_data
    predictions = run(corpus, args.query_data, args.method, args.background)
    summary = predictions["metadata"]
    if args.command == "evaluate" or args.evaluate:
        summary, _, _ = evaluate_pipeline(predictions)
    if args.command is None and args.output is not None:
        save_submission(predictions["rankings"], predictions["query_paths"], args.output,
                        valid_db_ids=set(predictions["database"]), masks=predictions["masks"] or None,
                        levels=args.levels)
        print(f"Submission saved to: {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
