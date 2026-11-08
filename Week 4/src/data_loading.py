"""Data access shared by all weeks: dataset folders, image, mask and ground-truth loading.

Used by main.py, the retrieval, evaluation and submission modules and the weeks/*/experiments scripts for every
week (BBDD corpus, QSD/QST query sets of Weeks 1-4). Images are decoded with cv2.imdecode from bytes read by
numpy, so paths with non-ASCII characters work; ground truth is the course's gt_corresps.pkl.
"""

import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


GROUND_TRUTH_FILENAME = "gt_corresps.pkl"


def dataset_signature(directory: Path) -> str:
    """Identify a corpus by its location, filenames, sizes and modification times (used in cache keys).

    Args:
        directory (Path): Image folder; its JPG files are listed with `list_images`.

    Returns:
        str: First 16 hexadecimal characters of a SHA-256 digest; changes when any image is added, removed,
        resized or touched, or when the folder moves.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no usable JPG images.
    """
    digest = hashlib.sha256(str(directory.resolve()).encode())
    for path in list_images(directory):
        stat = path.stat()
        digest.update(repr((path.name, stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()[:16]


def get_file_paths(
    directory: Path,
    pattern: str,
) -> list[Path]:
    """Return matching file paths from a directory, sorted by filename.

    Args:
        directory (Path): Directory where files are searched (not recursively, unless the pattern says so).
        pattern (str): Glob pattern used to match files, e.g. "*.jpg".

    Returns:
        list[Path]: Matching file paths sorted by path.

    Raises:
        FileNotFoundError: If the directory does not exist or no files match.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = sorted(directory.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {directory}")

    return paths


def resolve_dataset(data_dir: Path, name: str) -> Path:
    """Find the folder of a dataset under data_dir, also when it is nested as the course zips unpack it.

    Candidates, in order: data_dir/name/name, data_dir/name and, for names with an underscore, the bundle
    folder data_dir/<first 3 letters>_<suffix> (e.g. qsd_w4 for qsd1_w4) as bundle/name/name and bundle/name.
    The first candidate that holds at least one JPG wins.

    Args:
        data_dir (Path): Root data folder.
        name (str): Dataset name, e.g. "BBDD", "qsd2_w3" or "qst1_w4".

    Returns:
        Path: The dataset folder, or data_dir/name when no candidate holds JPG images (so later loading reports
        the expected path).
    """
    candidates = [data_dir / name / name, data_dir / name]
    if "_" in name:  # e.g. qsd1_w4 -> qsd_w4/qsd1_w4
        bundle = data_dir / f"{name[:3]}_{name.rsplit('_', 1)[1]}" / name
        candidates.extend((bundle / name, bundle))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate
    return data_dir / name


def list_images(folder: Path) -> list[Path]:
    """Return sorted JPG paths from a corpus or query directory, without macOS "._" metadata files.

    Args:
        folder (Path): Directory containing the images.

    Returns:
        list[Path]: Image paths sorted by filename.

    Raises:
        FileNotFoundError: If the directory does not exist or no JPG images are found.
    """
    image_paths = get_file_paths(directory=folder, pattern="*.jpg")
    image_paths = [path for path in image_paths if not path.name.startswith("._")]  # AppleDouble files, not images

    if not image_paths:
        raise FileNotFoundError(f"No usable JPG images found in: {folder}")

    return image_paths


def has_ground_truth(folder: Path) -> bool:
    """Check whether a query directory contains the ground-truth pickle.

    Args:
        folder (Path): Directory containing query images and optional ground truth.

    Returns:
        bool: Whether gt_corresps.pkl exists in the folder (development sets have it, test sets do not).

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    return (folder / GROUND_TRUTH_FILENAME).is_file()


def load_image(path: Path, *, grayscale: bool = False) -> NDArray[np.uint8]:
    """Load an image using OpenCV.

    Args:
        path (Path): Path to the image file.
        grayscale (bool): Load a single channel instead of BGR.

    Returns:
        NDArray[np.uint8]: (H, W, 3) uint8 BGR image, or (H, W) uint8 if grayscale is True.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is empty or OpenCV cannot decode the image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # Read bytes with numpy and decode them: cv2.imread fails on non-ASCII paths on Windows.
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    if image_bytes.size == 0:
        raise ValueError(f"Image file is empty: {path}")

    try:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imdecode(image_bytes, flag)
    except cv2.error as error:
        raise ValueError(f"Could not decode image: {path}") from error

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def load_mask(path: Path) -> NDArray[np.uint8]:
    """Load a binary mask: 0 is background and 255 is foreground.

    Args:
        path (Path): Mask path; use image_path.with_suffix(".png") for the ground-truth mask of a query image.

    Returns:
        NDArray[np.uint8]: (H, W) uint8 mask with values 0 and 255, at the original resolution.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If the file cannot be decoded or the mask contains values other than 0 and 255.
    """
    mask = load_image(path, grayscale=True)
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError(f"Mask must contain only 0 and 255: {path}")
    return mask


def load_ground_truth(data_dir: Path) -> dict[int, list[int]]:
    """Load query-to-database correspondences from the ground-truth pickle.

    The pickle holds one list per query image in filename order; entries are matched to the sorted image paths.

    Args:
        data_dir (Path): Directory containing the query images and gt_corresps.pkl.

    Returns:
        dict[int, list[int]]: Query ID -> database IDs of its paintings in the pickle's order (-1 marks a
        painting that is not in the database in the Week 4 sets).

    Raises:
        FileNotFoundError: If the directory, ground-truth file, or JPG images are missing.
        ValueError: If the image and annotation counts differ or query IDs are duplicated.
    """
    ground_truth_path = data_dir / GROUND_TRUTH_FILENAME

    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {ground_truth_path}"
        )

    images_paths = list_images(data_dir)

    with ground_truth_path.open("rb") as file:
        ground_truth: list[list[int]] = pickle.load(file)

    return get_ground_truth_mapping(images_paths, ground_truth)


def get_image_id(path: Path) -> int:
    """Extract the numeric image ID from a filename.

    Args:
        path (Path): Path whose stem ends with a numeric ID after the last underscore, or is all digits
            (e.g. bbdd_00012.jpg or 00012.jpg).

    Returns:
        int: Numeric image ID.

    Raises:
        ValueError: If a numeric ID cannot be extracted from the filename.
    """
    image_id = path.stem.rsplit("_", maxsplit=1)[-1]

    if not image_id.isdigit():
        raise ValueError(f"Invalid image filename: {path.name}")

    return int(image_id)


def get_ground_truth_mapping(
    images: list[Path],
    ground_truth: list[list[int]],
) -> dict[int, list[int]]:
    """Associate each query image ID with its ground-truth correspondences.

    Args:
        images (list[Path]): Query image paths, in the same order as ground_truth.
        ground_truth (list[list[int]]): Ground-truth database IDs for each query.

    Returns:
        dict[int, list[int]]: Query ID (from the filename) -> its matching database IDs.

    Raises:
        ValueError: If the number of images and ground-truth entries differs
            or a query ID is duplicated.
    """
    if len(images) != len(ground_truth):
        raise ValueError(
            f"Number of images ({len(images)}) does not match "
            f"number of ground truth entries ({len(ground_truth)})."
        )

    mapping: dict[int, list[int]] = {}

    for image_path, ground_truth_ids in zip(images, ground_truth):
        query_id = get_image_id(image_path)

        if query_id in mapping:
            raise ValueError(f"Duplicate query ID: {query_id}")

        mapping[query_id] = ground_truth_ids

    return mapping
