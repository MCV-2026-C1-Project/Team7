"""Run any submitted Week 1–4 method on an independently selected query dataset.

Commands (run from the repository root, datasets in ``data/``; see ``python main.py <command> -h``):
    evaluate  run inference, then score it with the dataset's annotations (development sets, qsd*)
    predict   run inference and write result.pkl without reading annotations (test sets, qst*)
    validate  check an existing result.pkl (format, query count, museum ids) without inference

Common flags: ``--week`` (1-4, default 4), ``--method`` (method1 / method2), ``--set`` (default qsd1 / qst1 of
``--week``), ``--data``, ``--levels``. ``evaluate`` / ``predict`` also take ``--background``, ``--threshold``
(Week 4), ``--max-paintings`` and ``--out``; ``predict`` also ``--force``; ``validate`` takes ``--delivery``.
Without ``--out``, ``predict`` writes the test sets (qst*) of the method's week to
weeks/weekN/submission/QSTn/<method>/result.pkl, its development sets (qsd*) to
weeks/weekN/results/dev_predictions/<SET>/<method>/result.pkl and any other week's set to
weeks/weekN/results/cross_week/<SET>/<method>/result.pkl. ``predict`` refuses to overwrite an existing
result.pkl under weeks/*/submission/ (a delivered file) unless ``--force`` is given. Running with no arguments
is ``evaluate`` of Week 4 on qsd1_w4.

Week 1 — global colour histograms (two methods):
    python main.py evaluate --week 1 --method method1 --set qsd1_w1
    python main.py evaluate --week 1 --method method2 --set qsd1_w1
    python main.py predict --week 1 --method method1 --set qst1_w1
    python main.py predict --week 1 --method method2 --set qst1_w1

Week 2 — spatial colour histograms (two methods):
    python main.py evaluate --week 2 --method method1 --set qsd1_w2
    python main.py evaluate --week 2 --method method2 --set qsd1_w2
    python main.py evaluate --week 2 --method method1 --set qsd2_w2
    python main.py evaluate --week 2 --method method2 --set qsd2_w2
    python main.py predict --week 2 --method method1 --set qst1_w2
    python main.py predict --week 2 --method method2 --set qst1_w2
    python main.py predict --week 2 --method method1 --set qst2_w2
    python main.py predict --week 2 --method method2 --set qst2_w2

Week 3 — colour and texture fusion (one method):
    python main.py evaluate --week 3 --method method1 --set qsd1_w3
    python main.py evaluate --week 3 --method method1 --set qsd2_w3
    python main.py predict --week 3 --method method1 --set qst1_w3
    python main.py predict --week 3 --method method1 --set qst2_w3

Week 4 — local features and unknown rejection (one method):
    python main.py evaluate --week 4 --method method1 --set qsd1_w4
    python main.py predict --week 4 --method method1 --set qst1_w4
    python main.py evaluate --week 4 --set qsd1_w4 --threshold 12     # another local acceptance score

Cross-week runs (e.g. the Week 3 method on the Week 4 set, Task 3 baseline):
    python main.py evaluate --week 3 --set qsd1_w4

Validate deliveries:
    python main.py validate --week 4
    python main.py validate --week 3 --set qst2_w3
    python main.py validate --week 1 --method method2 --set qst1_w1
    python main.py validate --week 4 --set qst1_w4 --delivery path/to/result.pkl
"""

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

from src.config import MAX_PAINTINGS, SUBMITTED_METHODS, TOP_K, UNKNOWN_THRESHOLD
from src.data_loading import get_image_id, list_images, load_ground_truth, resolve_dataset
from src.evaluate import evaluate_local, evaluate_pipeline
from src.local_pipeline import LocalMethod, run as run_pipeline
from src.pipeline import run_pipeline as run_global_pipeline
from src.submission import save_submission, validate_delivery
from src.unknowns import UNKNOWN

PROJECT_ROOT = Path(__file__).resolve().parent  # repository root (outputs go under weeks/)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"         # BBDD and the query datasets


def positive_int(value: str) -> int:
    """argparse type: parse a positive integer.

    Args:
        value: Command-line text.

    Returns:
        The integer (>= 1).

    Raises:
        argparse.ArgumentTypeError: If the integer is smaller than 1.
        ValueError: If ``value`` is not an integer (argparse reports it as an invalid value).
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def dataset_week(name: str) -> int | None:
    """Week of a course dataset read from its name, without reading any annotations.

    Args:
        name: Dataset folder name, e.g. "qsd1_w4" or "qst2_w3" (case-insensitive).

    Returns:
        The week (1-4), or None for a name that does not follow the ``qsdN_wK`` / ``qstN_wK`` pattern.
    """
    match = re.fullmatch(r"q(?:sd|st)\d+_w([1-4])", name.lower())
    return int(match[1]) if match else None


def result_path(week: int, method: str, name: str) -> Path:
    """Default result.pkl path of a method run on a dataset.

    Test sets (qst*) of the method's own week go to weeks/weekN/submission (where the delivered files are);
    its development sets (qsd*) go to weeks/weekN/results/dev_predictions, so they never land next to a
    delivery; cross-week runs go to weeks/weekN/results/cross_week, named with the full dataset name.

    Args:
        week: Week of the method.
        method: "method1" or "method2".
        name: Query dataset name, e.g. "qst1_w4".

    Returns:
        weeks/weekN/submission/QSTn/<method>/result.pkl, weeks/weekN/results/dev_predictions/<NAME>/<method>/
        result.pkl or weeks/weekN/results/cross_week/<NAME>/<method>/result.pkl.
    """
    if dataset_week(name) == week and name.lower().startswith("qst"):
        return PROJECT_ROOT / "weeks" / f"week{week}" / "submission" / name.split("_")[0].upper() / method / "result.pkl"
    folder = "dev_predictions" if dataset_week(name) == week else "cross_week"
    return PROJECT_ROOT / "weeks" / f"week{week}" / "results" / folder / name.upper() / method / "result.pkl"


def is_delivered_file(path: Path) -> bool:
    """True for an existing result.pkl under weeks/*/submission/ (a delivered file that predict must not replace).

    Args:
        path: Output path of predict.

    Returns:
        Whether the file exists inside a weeks/weekN/submission/ folder of this repository.
    """
    try:
        parts = path.resolve().relative_to((PROJECT_ROOT / "weeks").resolve()).parts
    except ValueError:
        return False
    return len(parts) > 2 and parts[1] == "submission" and path.is_file()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line and fill in the week-dependent defaults.

    Args:
        argv: Arguments without the program name; None reads ``sys.argv``. An empty list means ``evaluate``.

    Returns:
        Namespace with ``command``, ``week``, ``method``, ``data``, ``set`` and ``levels`` resolved; for
        ``validate`` also ``delivery`` (a .pkl path); otherwise ``threshold``, ``max_paintings``, ``segment``
        (bool) and ``out`` (None for evaluate without --out).

    Raises:
        SystemExit: On invalid arguments or combinations (``parser.error``).
    """
    parser = argparse.ArgumentParser(description="Run submitted Week 1–4 painting retrieval methods.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "predict", "validate"):
        command = commands.add_parser(name, help={
            "evaluate": "run inference and evaluate using development annotations",
            "predict": "run inference and export result.pkl without reading annotations",
            "validate": "check an existing result.pkl without inference",
        }[name])
        command.add_argument("--week", type=int, choices=(1, 2, 3, 4), default=4,
                             help="week of the submitted method (default: 4)")
        command.add_argument("--method", choices=("method1", "method2"), default="method1",
                             help="Weeks 1–2 have two submitted methods; Weeks 3–4 have method1")
        command.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                             help="folder containing BBDD and query datasets")
        command.add_argument("--set", help="independent query dataset (default: qsd1/qst1 of --week)")
        command.add_argument("--levels", type=int, choices=(2, 3),
                             help="PKL list levels (default: 2 for W1–2 methods on W1–2 datasets, otherwise 3)")
        if name == "validate":
            command.add_argument("--delivery", type=Path, help="delivery folder or result.pkl path")
        else:
            command.add_argument("--background", choices=("auto", "none", "remove"), default="auto",
                                 help="auto (default): Week 1 skips segmentation; other methods segment all "
                                      "datasets except qsd1/qst1 of Weeks 1-3; none: never; remove: always")
            command.add_argument("--threshold", type=positive_int,
                                 help=f"Week 4 minimum local acceptance score (default: {UNKNOWN_THRESHOLD})")
            command.add_argument("--max-paintings", type=int, choices=(1, 2, 3),
                                 help="maximum paintings per query (default: 1 for Week 1–2 methods, "
                                      "2 for Week 3, 3 for Week 4)")
            command.add_argument("--out", type=Path,
                                 help="destination result.pkl (predict default: the week's submission folder for "
                                      "its qst sets, results/dev_predictions for its qsd sets, results/cross_week "
                                      "otherwise; evaluate writes nothing without it)")
            if name == "predict":
                command.add_argument("--force", action="store_true",
                                     help="allow replacing an existing delivered result.pkl under weeks/*/submission/")
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv or ["evaluate"])
    if args.method not in SUBMITTED_METHODS[args.week]:
        parser.error(f"Week {args.week} only submitted method1.")
    # Development sets (qsd) for evaluate, test sets (qst) for predict / validate.
    args.set = args.set or f"{'qsd' if args.command == 'evaluate' else 'qst'}1_w{args.week}"
    query_week = dataset_week(args.set)
    # Weeks 1-2 were delivered flat (one painting per query); from Week 3 on, one list per painting.
    args.levels = args.levels or (2 if args.week in (1, 2) and query_week in (1, 2) else 3)
    target = result_path(args.week, args.method, args.set)
    if args.command == "validate":
        args.delivery = args.delivery or target
        if args.delivery.suffix != ".pkl":  # a delivery root: <root>/QSTn/<method>/result.pkl
            args.delivery = args.delivery / target.parent.parent.name / args.method / "result.pkl"
    else:
        if args.threshold is not None and args.week != 4:
            parser.error("--threshold applies only to the Week 4 method.")
        args.threshold = args.threshold or UNKNOWN_THRESHOLD
        args.max_paintings = args.max_paintings or MAX_PAINTINGS[args.week]
        # The frozen recipes are only run the way they were submitted.
        if args.week in (1, 2) and args.max_paintings != 1:
            parser.error("Week 1–2 methods assume one painting per query.")
        if args.week == 1 and args.background == "remove":
            parser.error("The submitted Week 1 methods do not include background removal.")
        # "auto" decides from the dataset name, so an unrecognised name needs an explicit choice.
        if query_week is None and args.background == "auto" and args.week != 1:
            parser.error("For a custom dataset name, specify --background none or remove.")
        # auto: qsd1 / qst1 of Weeks 1-3 are already cropped paintings; QS2 sets and all Week 4 sets are photos
        # with background.
        args.segment = args.week != 1 and (args.background == "remove" or (
            args.background == "auto" and (query_week == 4 or not args.set.lower().startswith(("qsd1_", "qst1_")))))
        if args.command == "predict" and args.out is None:
            args.out = target
    return args


def print_progress(done: int, total: int) -> None:
    """Progress callback of the museum descriptor computation.

    Args:
        done: Museum images described so far.
        total: Number of museum images.
    """
    print(f"Database descriptors: {done}/{total}", flush=True)


def run_query_set(data: Path, name: str, threshold: int, evaluate: bool, out: Path | None,
                  max_paintings: int = 3, *, week: int = 4, method: str = "method1",
                  segment: bool = True, levels: int = 3) -> None:
    """Run the selected frozen recipe on a query set; annotations are only used after inference.

    Prints a JSON summary (with the metrics when ``evaluate``) and, if ``out`` is given, writes the delivery.

    Args:
        data: Folder containing BBDD and the query datasets.
        name: Query dataset name.
        threshold: Week 4 minimum local acceptance score (ignored by the other weeks).
        evaluate: Score the predictions with the dataset's ground truth.
        out: Destination result.pkl, or None to write nothing.
        max_paintings: Maximum paintings per query.
        week: Week of the submitted method.
        method: "method1" or "method2".
        segment: Remove the background before describing.
        levels: List levels of the written PKL (2 or 3).

    Raises:
        FileNotFoundError: If ``evaluate`` is requested for a Week 1-3 method and the ground truth is missing.
    """
    corpus, queries = resolve_dataset(data, "BBDD"), resolve_dataset(data, name)
    preset = SUBMITTED_METHODS[week][method]
    global_result = None
    # Week 4 is the local-feature pipeline (preset None); Weeks 1-3 are global descriptor recipes.
    if week == 4:
        local_method = replace(LocalMethod(), unknown_threshold=threshold, max_paintings=max_paintings)
        result = run_pipeline(queries, corpus, local_method, unknowns=True, segment=segment, progress=print_progress)
        rankings = result["answers"]
    else:
        assert preset is not None
        global_result = run_global_pipeline(corpus, queries, segment=segment, week=week,
            descriptor=preset["descriptor"], similarity=preset["similarity"],
            top_k=TOP_K, max_paintings=max_paintings, method=method, progress=print_progress)
        rankings = global_result["rankings"]
    summary = {"week": week, "method": method, "set": name, "segment": segment,
               "max_paintings": max_paintings,
               "unknown_paintings": sum(r == [UNKNOWN] for p in rankings.values() for r in p)}
    if week == 4:
        summary["threshold"] = threshold
    else:
        summary |= {"descriptor": preset["descriptor"], "similarity": preset["similarity"]}
    if evaluate:
        # Week 4 answers (with [-1]) are scored per painting with up to three paintings; older methods are
        # scored by their own pipeline, which also handles datasets with three paintings and unknowns.
        if global_result is None:
            summary |= evaluate_local(rankings, load_ground_truth(queries), max_paintings=3)
        else:
            diagnostics, _, _ = evaluate_pipeline(global_result)
            if "retrieval" not in diagnostics:
                raise FileNotFoundError(f"Ground truth not found in: {queries}")
            summary |= diagnostics["retrieval"] | diagnostics["unknowns"]
            if "mask_metrics" in diagnostics:
                summary["mask_metrics"] = diagnostics["mask_metrics"]
    if out is not None:
        save_submission(rankings, {get_image_id(p): p for p in list_images(queries)}, out,
                        valid_db_ids={get_image_id(p) for p in list_images(corpus)},
                        masks=global_result["masks"] if global_result and segment else None,
                        max_paintings=max_paintings, levels=levels)
        print(f"Submission saved to: {out}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Command-line entry point: validate a delivery, or run evaluate / predict."""
    args = parse_args()
    if args.command == "validate":
        counts = validate_delivery(args.delivery, args.data, query_set=args.set, levels=args.levels, method=args.method)
        print(f"Delivery: {counts['queries']} queries, {counts['paintings']} paintings, {counts['unknown']} unknown, "
              f"{counts['levels']} levels - OK")
    else:
        # Delivered files are the course submissions: replacing one must be explicit.
        if args.command == "predict" and is_delivered_file(args.out) and not args.force:
            sys.exit(f"Refusing to overwrite the delivered file {args.out}; use --out elsewhere, or --force.")
        run_query_set(args.data, args.set, args.threshold, args.command == "evaluate", args.out,
                      args.max_paintings, week=args.week, method=args.method,
                      segment=args.segment, levels=args.levels)


if __name__ == "__main__":
    main()
