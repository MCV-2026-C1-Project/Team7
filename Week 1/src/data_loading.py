import pickle
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


GROUND_TRUTH_FILENAME = "gt_corresps.pkl"


def get_file_paths(
    directory: Path,
    pattern: str,
) -> list[Path]:
    """Return matching file paths from a directory, sorted by filename.

    Args:
        directory: Directory where files are searched.
        pattern: Glob pattern used to match files.

    Returns:
        Matching file paths sorted by filename.

    Raises:
        FileNotFoundError: If the directory does not exist or no files match.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    paths = sorted(directory.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in: {directory}")

    return paths


def list_images(folder: Path) -> list[Path]:
    """Return sorted JPG paths from a corpus or query directory.

    Args:
        folder: Directory containing the images.

    Returns:
        Image paths sorted by filename.

    Raises:
        FileNotFoundError: If the directory does not exist or no JPG images are found.
    """
    image_paths = get_file_paths(directory=folder, pattern="*.jpg")
    image_paths = [path for path in image_paths if not path.name.startswith("._")]

    if not image_paths:
        raise FileNotFoundError(f"No usable JPG images found in: {folder}")

    return image_paths


def has_ground_truth(folder: Path) -> bool:
    """Check whether a query directory contains the ground-truth pickle.

    Args:
        folder: Directory containing query images and optional ground truth.

    Returns:
        Whether the expected ground-truth file exists.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    return (folder / GROUND_TRUTH_FILENAME).is_file()


def load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image using OpenCV.

    Args:
        path: Path to the image file.

    Returns:
        Image as a uint8 NumPy array in BGR color order.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file is empty or OpenCV cannot decode the image.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    if image_bytes.size == 0:
        raise ValueError(f"Image file is empty: {path}")

    try:
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Could not decode image: {path}") from error

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def load_ground_truth(data_dir: Path) -> dict[int, list[int]]:
    """Load query-to-database correspondences from the ground-truth pickle.

    Args:
        data_dir: Directory containing the query images and ground truth.

    Returns:
        Dictionary mapping each query ID to its list of matching database IDs.

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
        path: Path whose filename ends with a numeric image ID.

    Returns:
        Numeric image ID.

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
        images: Query image paths.
        ground_truth: Ground-truth database IDs for each query.

    Returns:
        Dictionary mapping each query ID to its matching database IDs.

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
