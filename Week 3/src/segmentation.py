"""Background removal (Week 3, Task 3): the union mask and one mask per painting (up to two, left to right).

Wall colour = a plane fitted to the border in Lab; pixels far from it are painting; closing, hole filling,
the two largest components, refined with convex quadrilaterals. The parameters validated by Albert Puga
(results/week3/task3/segmentation_parameters.csv) are `SEGMENTATION` in src/config.py."""

from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import SEGMENTATION as P


class Painting(TypedDict):
    """One detected painting: its mask and bounding box (x0, y0, x1, y1)."""
    mask: NDArray[np.uint8]
    bbox: tuple[int, int, int, int]


def remove_background(
    image: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Return the union mask and up to two paintings, ordered left to right.

    Masks are native-resolution uint8 arrays (0/255). Each painting has its own
    mask and box (x0, y0, x1, y1), with exclusive upper bounds. Input is BGR and
    is never modified. No detections returns an empty list and an empty mask.
    """
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 2):
        raise ValueError("Expected a BGR uint8 image at least 2x2 pixels.")

    distance = _background_distance(image)
    mask = (distance > P["threshold"]).astype(np.uint8) * 255
    mask = close_mask(mask, P["closing_kernel_size"])
    _, paintings = extract_paintings(fill_mask_holes(mask))
    paintings = [
        painting for painting in paintings
        if np.count_nonzero((painting["mask"] > 0) & (distance > P["strong_threshold"]))
        >= P["min_seed_fraction"] * distance.size
    ]
    evidence = cv2.GaussianBlur(distance, (0, 0), P["distance_sigma"])
    return _refine_paintings(paintings, _quadrilateral_candidates(image), evidence)


def extract_border_pixels(image: NDArray, width: int) -> NDArray:
    """Return the four border strips, counting corners only once."""
    if width <= 0 or 2 * width > min(image.shape[:2]):
        raise ValueError("Border width must be positive and at most half the short side.")
    border = np.zeros(image.shape[:2], dtype=bool)
    border[:width, :] = border[-width:, :] = True
    border[:, :width] = border[:, -width:] = True
    return image[border]


def close_mask(mask: NDArray[np.uint8], kernel_size: int = 5) -> NDArray[np.uint8]:
    """Seal small gaps with an odd square kernel; 0 disables closing."""
    if (not isinstance(kernel_size, (int, np.integer)) or kernel_size < 0
            or (kernel_size > 0 and kernel_size % 2 == 0)):
        raise ValueError("kernel_size must be zero or a positive odd integer.")
    if kernel_size <= 1:
        return mask.copy()
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))


def fill_mask_holes(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Fill background regions without an 8-connected path to an image border."""
    _, labels = cv2.connectedComponents((mask == 0).astype(np.uint8), connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]  # Label 0 is original foreground.
    return (~np.isin(labels, border_labels)).astype(np.uint8) * 255


def _background_distance(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Fit a gradual wall-color plane to the border and return weighted Lab L1."""
    smoothed = cv2.GaussianBlur(image, (0, 0), P["image_sigma"])
    lab = cv2.cvtColor(smoothed.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    lab[:, :, 0] *= P["lightness_weight"]
    height, width = lab.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x /= width - 1
    y /= height - 1
    coordinates = np.stack((np.ones_like(x), x, y), axis=2)
    border_width = min(max(1, round(P["border_fraction"] * min(height, width))), min(height, width) // 2)
    colors = extract_border_pixels(lab, border_width)
    points = extract_border_pixels(coordinates, border_width)
    step = max(1, min(10, len(colors) // 30))
    colors, points = colors[::step], points[::step]

    # Repeated fits, each retaining the border samples with the smallest residuals.
    keep = np.ones(len(colors), dtype=bool)
    for _ in range(P["fit_iterations"]):
        coefficients = np.linalg.lstsq(points[keep], colors[keep], rcond=None)[0]
        residuals = np.linalg.norm(colors - points @ coefficients, axis=1)
        keep = residuals <= np.quantile(residuals, P["retained_fraction"])
    return np.abs(lab - coordinates @ coefficients).sum(axis=2)


def extract_paintings(mask: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Keep the two largest components occupying at least `min_area_fraction` of the image."""
    if mask.ndim != 2 or mask.dtype != np.uint8 or np.any((mask != 0) & (mask != 255)):
        raise ValueError("Expected a two-dimensional binary uint8 mask.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= P["min_area_fraction"] * mask.size]
    candidates.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    paintings: list[Painting] = []
    union = np.zeros_like(mask)
    for label in candidates[:2]:
        x, y, width, height = (int(value) for value in stats[label, :4])
        painting_mask = (labels == label).astype(np.uint8) * 255
        paintings.append({"mask": painting_mask, "bbox": (x, y, x + width, y + height)})
        union |= painting_mask
    paintings.sort(key=lambda painting: (painting["bbox"][0] + painting["bbox"][2]) / 2)
    return union, paintings


def _quadrilateral_candidates(image: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    """Find convex four-corner contours independently at every Canny scale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates: list[NDArray[np.int32]] = []
    for sigma in P["canny_sigmas"]:
        smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        edges = cv2.Canny(smoothed, *P["canny_thresholds"], L2gradient=True)
        edges = close_mask(edges, P["edge_closing"])
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        scale_candidates = []
        for contour in contours:
            polygon = cv2.approxPolyDP(contour, P["polygon_epsilon"] * cv2.arcLength(contour, True), True)
            if (len(polygon) == 4 and cv2.isContourConvex(polygon)
                    and cv2.contourArea(polygon) >= P["min_area_fraction"] * edges.size):
                scale_candidates.append(polygon)
        scale_candidates.sort(key=cv2.contourArea, reverse=True)
        candidates.extend(scale_candidates)
    return candidates


def _refine_paintings(
    paintings: list[Painting],
    candidates: list[NDArray[np.int32]],
    distance: NDArray[np.float32],
) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Keep the best overlapping contour that preserves painting and limits wall."""
    union = np.zeros(distance.shape, np.uint8)
    refined: list[Painting] = []
    for index, painting in enumerate(paintings):
        original = painting["mask"] > 0
        original_area = np.count_nonzero(original)
        strong = original & (distance > P["strong_threshold"])
        strong_area = np.count_nonzero(strong)
        others = np.zeros(distance.shape, dtype=bool)
        for other_index, other in enumerate(paintings):
            if other_index != index:
                others |= other["mask"] > 0
        best_mask = painting["mask"]
        best_iou = P["min_iou"]
        occupied = (others | (union > 0)).astype(np.uint8)
        occupied_margin = cv2.dilate(occupied, np.ones((3, 3), np.uint8)) > 0
        for polygon in candidates:
            candidate = np.zeros(distance.shape, np.uint8)
            cv2.fillPoly(candidate, [polygon], 255)
            foreground = candidate > 0
            if np.any(foreground & occupied_margin):
                continue
            # A contour may remove at most max_strong_loss of the confident painting pixels.
            if np.count_nonzero(strong & ~foreground) / max(1, strong_area) > P["max_strong_loss"]:
                continue
            added = foreground & ~original
            # Larger additions may hold at most max_added_wall_fraction of confident wall.
            if np.count_nonzero(added) > P["min_added_area_fraction"] * original_area:
                if np.mean(distance[added] < P["wall_threshold"]) > P["max_added_wall_fraction"]:
                    continue
            intersection = np.count_nonzero(foreground & original)
            area_union = np.count_nonzero(foreground | original)
            iou = intersection / area_union if area_union else 0.0
            if iou >= best_iou:
                best_iou, best_mask = iou, candidate
        x, y, width, height = cv2.boundingRect(best_mask)
        refined.append({"mask": best_mask, "bbox": (x, y, x + width, y + height)})
        union |= best_mask
    refined.sort(key=lambda painting: (painting["bbox"][0] + painting["bbox"][2]) / 2)
    return union, refined
