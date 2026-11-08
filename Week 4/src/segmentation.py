"""Background removal: a union mask and individual paintings in reading order.

`remove_background` keeps one method per week so earlier deliveries can be reproduced: week=2 is the submitted
Week 2 single-painting method (median border colour); week=3 is the Week 3 Task 3 multi-painting method (wall
colour = a plane fitted to the border in Lab, pixels far from it are painting; closing, hole filling, the largest
valid components, refined with convex quadrilaterals; up to two paintings, left to right); week=4 (default, used
by the Week 4 local-feature pipeline in src/local_pipeline.py) uses the same wall model with a confident-foreground
filter and the MIN_REGION_FRACTION region filter, up to three paintings in reading order. `extract_paintings` also
splits ground-truth masks into paintings for evaluation (src/evaluate.py). The fixed parameters are
`SEGMENTATION` in src/config.py.
"""

from typing import TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import MIN_REGION_FRACTION, SEGMENTATION as P


class Painting(TypedDict):
    """One detected painting.

    Attributes:
        mask (NDArray[np.uint8]): (H, W) uint8 mask (0/255) of this painting at the image's resolution.
        bbox (tuple[int, int, int, int]): Bounding box (x0, y0, x1, y1) in pixels, upper bounds exclusive.
    """
    mask: NDArray[np.uint8]
    bbox: tuple[int, int, int, int]


def remove_background(
    image: NDArray[np.uint8],
    *, max_paintings: int = 3, week: int = 4,
) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Segment the paintings of a photo: return the union mask and up to max_paintings regions.

    The week switch keeps each submitted method unchanged so that its delivery can be regenerated: week=2 is
    the Week 2 single-painting median-border method (max_paintings is ignored); week=3 is the Week 3 method,
    which filters by confident-foreground seeds after choosing the largest components and orders paintings
    left to right; week=4 filters by seeds and MIN_REGION_FRACTION before the count limit and returns reading
    order (see `order_paintings`).

    Args:
        image (NDArray[np.uint8]): (H, W, 3) uint8 BGR photo, at least 2x2 pixels; never modified.
        max_paintings (int): Maximum number of paintings returned (weeks 3 and 4).
        week (int): Which submitted method to run: 2, 3 or 4.

    Returns:
        tuple[NDArray[np.uint8], list[Painting]]: The (H, W) uint8 union mask (0/255) at native resolution and
        the paintings, each with its own mask and box (x0, y0, x1, y1) with exclusive upper bounds. Without
        detections the list is empty and the mask is all zeros.

    Raises:
        ValueError: If the image is not a BGR uint8 image of at least 2x2 pixels, the week is not 2, 3 or 4,
            or max_paintings is not a positive integer.
    """
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or min(image.shape[:2]) < 2):
        raise ValueError("Expected a BGR uint8 image at least 2x2 pixels.")

    if week == 2:
        return _week2_background(image)
    if week not in (3, 4):
        raise ValueError("Background removal is available for weeks 2, 3 and 4.")
    distance = _background_distance(image)
    mask = (distance > P["threshold"]).astype(np.uint8) * 255
    mask = close_mask(mask, P["closing_kernel_size"])
    # Hole filling: enclosed pixels close to the wall colour (inside a painting) count as painting.
    _, paintings = extract_paintings(fill_mask_holes(mask), max_paintings=max_paintings,
                                    strong_mask=distance > P["strong_threshold"] if week == 4 else None,
                                    min_region_fraction=MIN_REGION_FRACTION if week == 4 else 0.0)
    if week == 3:
        # The submitted Week 3 code selected largest components before seed filtering, so a rejected component
        # is not replaced by the next largest one; kept as submitted to reproduce the Week 3 delivery.
        paintings = [p for p in paintings if np.count_nonzero(
            (p["mask"] > 0) & (distance > P["strong_threshold"])) >= P["min_seed_fraction"] * distance.size]
        paintings.sort(key=lambda p: p["bbox"][0] + p["bbox"][2])  # Week 3 order: left to right
    # Snap each colour region to a convex quadrilateral found on Canny edges (paintings are rectangles seen
    # in perspective), judged on the smoothed wall distance.
    evidence = cv2.GaussianBlur(distance, (0, 0), P["distance_sigma"])
    union, paintings = _refine_paintings(paintings, _quadrilateral_candidates(image), evidence)
    if week == 3:  # _refine_paintings returns reading order; Week 3 delivered left to right
        paintings.sort(key=lambda p: p["bbox"][0] + p["bbox"][2])
    return union, paintings


def _week2_background(image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Submitted Week 2 method: median border Lab, L2 > 18, closing, hole filling, largest component.

    L is weighted by 0.75; the border is 4% of the shorter side. These fixed values (not read from config) and
    the single-painting assumption belong to that submitted method and are kept to reproduce its delivery.
    """
    lab = cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2Lab).astype(np.float64)
    lab[:, :, 0] *= 0.75
    width = min(max(1, round(0.04 * min(image.shape[:2]))), min(image.shape[:2]) // 2)
    center = np.median(extract_border_pixels(lab, width), axis=0)
    mask = (np.linalg.norm(lab - center, axis=2) > 18).astype(np.uint8) * 255
    mask = fill_mask_holes(close_mask(mask, 5))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count == 1:  # only the background label: mask is all zeros
        return mask, []
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest).astype(np.uint8) * 255
    x, y, width, height = (int(value) for value in stats[largest, :4])
    return mask, [{"mask": mask, "bbox": (x, y, x + width, y + height)}]


def extract_border_pixels(image: NDArray, width: int) -> NDArray:
    """Return the pixels of the four border strips, counting corners only once.

    Args:
        image (NDArray): (H, W) or (H, W, C) array of any dtype (image, Lab values or pixel coordinates).
        width (int): Strip width in pixels; positive and at most half the shorter side.

    Returns:
        NDArray: (N,) or (N, C) array of border values in row-major order, N being the number of border pixels.

    Raises:
        ValueError: If width is not positive or exceeds half the shorter side.
    """
    if width <= 0 or 2 * width > min(image.shape[:2]):
        raise ValueError("Border width must be positive and at most half the short side.")
    border = np.zeros(image.shape[:2], dtype=bool)
    border[:width, :] = border[-width:, :] = True
    border[:, :width] = border[:, -width:] = True
    return image[border]


def close_mask(mask: NDArray[np.uint8], kernel_size: int = 5) -> NDArray[np.uint8]:
    """Seal small gaps with a morphological closing by an odd square kernel.

    Args:
        mask (NDArray[np.uint8]): (H, W) uint8 binary mask (0/255).
        kernel_size (int): Side of the square kernel in pixels; 0 or 1 disables closing.

    Returns:
        NDArray[np.uint8]: The closed (H, W) uint8 mask; a copy of the input when closing is disabled.

    Raises:
        ValueError: If kernel_size is not zero or a positive odd integer.
    """
    if (not isinstance(kernel_size, (int, np.integer)) or kernel_size < 0
            or (kernel_size > 0 and kernel_size % 2 == 0)):
        raise ValueError("kernel_size must be zero or a positive odd integer.")
    if kernel_size <= 1:
        return mask.copy()
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))


def fill_mask_holes(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Fill background regions without an 8-connected path to an image border.

    Args:
        mask (NDArray[np.uint8]): (H, W) uint8 binary mask, 0 is background and any other value foreground.

    Returns:
        NDArray[np.uint8]: (H, W) uint8 mask (0/255) where only background connected to the border stays 0.
    """
    _, labels = cv2.connectedComponents((mask == 0).astype(np.uint8), connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    border_labels = border_labels[border_labels != 0]  # Label 0 is original foreground.
    return (~np.isin(labels, border_labels)).astype(np.uint8) * 255


def _background_distance(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Fit a wall-colour plane (Lab linear in x, y) to the border and return each pixel's weighted Lab L1 distance."""
    smoothed = cv2.GaussianBlur(image, (0, 0), P["image_sigma"])
    lab = cv2.cvtColor(smoothed.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    lab[:, :, 0] *= P["lightness_weight"]
    height, width = lab.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x /= width - 1
    y /= height - 1
    coordinates = np.stack((np.ones_like(x), x, y), axis=2)  # design matrix [1, x, y] of the plane per pixel
    border_width = min(max(1, round(P["border_fraction"] * min(height, width))), min(height, width) // 2)
    colors = extract_border_pixels(lab, border_width)
    points = extract_border_pixels(coordinates, border_width)
    # Subsample the border samples: a step of at most 10, and 1 when there are fewer than 60 samples.
    step = max(1, min(10, len(colors) // 30))
    colors, points = colors[::step], points[::step]

    # Repeated fits, each retaining the border samples with the smallest residuals, so border samples that do not
    # follow the wall plane stop influencing it.
    keep = np.ones(len(colors), dtype=bool)
    for _ in range(P["fit_iterations"]):
        coefficients = np.linalg.lstsq(points[keep], colors[keep], rcond=None)[0]
        residuals = np.linalg.norm(colors - points @ coefficients, axis=1)
        keep = residuals <= np.quantile(residuals, P["retained_fraction"])
    return np.abs(lab - coordinates @ coefficients).sum(axis=2)


def extract_paintings(mask: NDArray[np.uint8], *, max_paintings: int = 3,
                      strong_mask: NDArray[np.bool_] | None = None,
                      min_region_fraction: float = 0.0) -> tuple[NDArray[np.uint8], list[Painting]]:
    """Split a binary mask into paintings: the largest 8-connected components that pass the filters.

    All filters are applied before the count limit, so a rejected component frees a slot for the next one:
    area (at least SEGMENTATION["min_area_fraction"] of the image), optional confident-foreground seeds (at
    least SEGMENTATION["min_seed_fraction"] of the image inside strong_mask) and the relative size filter.

    Args:
        mask (NDArray[np.uint8]): (H, W) uint8 binary mask (0/255), e.g. a predicted or ground-truth mask.
        max_paintings (int): Maximum number of components kept, largest first.
        strong_mask (NDArray[np.bool_] | None): (H, W) bool map of confident foreground from the wall model
            (Week 4), or None to skip the seed filter.
        min_region_fraction (float): Drop components smaller than this fraction (0 to 1) of the largest
            component that survived the previous filters; 0 disables it. Week 4 passes MIN_REGION_FRACTION
            to drop small spurious regions (labels, frame pieces).

    Returns:
        tuple[NDArray[np.uint8], list[Painting]]: The (H, W) uint8 union mask of the kept components and the
        paintings in reading order (`order_paintings`).

    Raises:
        ValueError: If max_paintings is not a positive int, the mask is not a 2-D 0/255 uint8 array, strong_mask
            has another shape or dtype, or min_region_fraction is outside [0, 1].
    """
    if type(max_paintings) is not int or max_paintings < 1:
        raise ValueError("max_paintings must be a positive integer.")
    if mask.ndim != 2 or mask.dtype != np.uint8 or np.any((mask != 0) & (mask != 255)):
        raise ValueError("Expected a two-dimensional binary uint8 mask.")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = [label for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= P["min_area_fraction"] * mask.size]
    if strong_mask is not None:
        if strong_mask.shape != mask.shape or strong_mask.dtype != np.bool_:
            raise ValueError("strong_mask must be a boolean array with the mask's shape.")
        strong_counts = np.bincount(labels[strong_mask], minlength=count)  # confident pixels per component label
        candidates = [label for label in candidates if strong_counts[label] >= P["min_seed_fraction"] * mask.size]
    if not 0.0 <= min_region_fraction <= 1.0:
        raise ValueError("min_region_fraction must be between 0 and 1.")
    if candidates and min_region_fraction > 0:
        largest = max(int(stats[label, cv2.CC_STAT_AREA]) for label in candidates)
        candidates = [label for label in candidates if stats[label, cv2.CC_STAT_AREA] >= min_region_fraction * largest]
    candidates.sort(key=lambda label: int(stats[label, cv2.CC_STAT_AREA]), reverse=True)
    paintings: list[Painting] = []
    union = np.zeros_like(mask)
    for label in candidates[:max_paintings]:
        x, y, width, height = (int(value) for value in stats[label, :4])
        painting_mask = (labels == label).astype(np.uint8) * 255
        paintings.append({"mask": painting_mask, "bbox": (x, y, x + width, y + height)})
        union |= painting_mask
    return union, order_paintings(paintings)


def _quadrilateral_candidates(image: NDArray[np.uint8]) -> list[NDArray[np.int32]]:
    """Find convex four-corner contours independently at every Canny scale (largest first within a scale)."""
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
    """Replace each painting mask by the quadrilateral of highest IoU that preserves painting and limits wall.

    A candidate is skipped if it touches another painting (3x3 dilated margin) or an already refined one, loses
    too many confident painting pixels or adds too much confident wall; a painting with no candidate reaching
    SEGMENTATION["min_iou"] keeps its colour mask.
    """
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
            if iou >= best_iou:  # ">=": on ties the later candidate wins
                best_iou, best_mask = iou, candidate
        x, y, width, height = cv2.boundingRect(best_mask)
        refined.append({"mask": best_mask, "bbox": (x, y, x + width, y + height)})
        union |= best_mask
    return union, order_paintings(refined)


def order_paintings(paintings: list[Painting]) -> list[Painting]:
    """Sort paintings in reading order: rows from top to bottom, paintings within a row from left to right.

    One or two paintings: two are stacked when their x-ranges overlap by more than half of the narrower one
    (the ground truth of QSD1-W4 then lists the upper one first) and are sorted by vertical centre; otherwise
    they are sorted by horizontal centre. Three or more: paintings, taken by top edge, join the first row whose
    every member overlaps them vertically by more than half the shorter height (requiring every member avoids
    chained rows); each row is then sorted by horizontal centre.

    Args:
        paintings (list[Painting]): Paintings with their bounding boxes (x0, y0, x1, y1) in pixels.

    Returns:
        list[Painting]: The same Painting objects in reading order (a new list; the input is not modified).
    """
    if len(paintings) == 2:
        (a0, _, a1, _), (b0, _, b1, _) = paintings[0]["bbox"], paintings[1]["bbox"]
        if min(a1, b1) - max(a0, b0) > 0.5 * min(a1 - a0, b1 - b0):
            return sorted(paintings, key=lambda painting: (painting["bbox"][1] + painting["bbox"][3]) / 2)
    if len(paintings) <= 2:
        return sorted(paintings, key=lambda painting: (painting["bbox"][0] + painting["bbox"][2]) / 2)
    rows: list[list[Painting]] = []
    for painting in sorted(paintings, key=lambda p: (p["bbox"][1], p["bbox"][0])):
        _, y0, _, y1 = painting["bbox"]
        for row in rows:
            if all(min(y1, other["bbox"][3]) - max(y0, other["bbox"][1])
                   > 0.5 * min(y1 - y0, other["bbox"][3] - other["bbox"][1]) for other in row):
                row.append(painting)
                break
        else:
            rows.append([painting])
    return [painting for row in rows
            for painting in sorted(row, key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2)]
