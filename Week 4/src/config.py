"""Project-wide settings: submitted method recipes, descriptor presets, noise and segmentation parameters.

Every week reads its frozen values from here: the Week 1-2 colour histogram recipes, the Week 3 noise
thresholds, wall-plane segmentation parameters and colour + texture fusion (``GLOBAL_PRESET``), and the Week 4
local-feature settings (``LOCAL_MATCH``, ``UNKNOWN_THRESHOLD``, ``RECOVERY_MIN_INLIERS``, ``GLOBAL_RATIO``,
``MIN_REGION_FRACTION``). ``SUBMITTED_METHODS`` maps each week to its delivered method(s) and is what
``main.py`` runs. Changing a value here changes the submitted methods.
"""

from typing import TypedDict


class DescriptorConfig(TypedDict, total=False):
    """Input of ``src.descriptors.compute_descriptor``; which keys are used depends on ``kind``."""
    kind: str                       # "1d", "2d", "3d" (colour), "texture" or "combined"
    color_spaces: list[str]         # 1d colour histograms
    space: str                      # 2d / 3d joint histograms
    bins: int                       # histogram bins per channel
    grid: int                       # N x N blocks (1 = global)
    pyramid_levels: int             # spatial pyramid levels (grids 1, 2, 4, ...); absent = no pyramid
    pyramid_weights: str            # pyramid level weights: "equal" or "lazebnik"
    gray_world: bool                # gray-world colour correction before the histograms
    contrast_stretch: bool          # stretch the luminance channels to their full range
    luminance_weight: float         # weight of the brightness histograms
    texture: str                    # "lbp", "dct" or "wavelet" (kind "texture")
    params: dict                    # keyword arguments of the texture descriptor
    parts: list["DescriptorConfig"]  # kind "combined": the descriptors to fuse
    weights: list[float]            # kind "combined": weight of each part
    scales: list[float]             # kind "combined": each part is divided by its scale
    denoise_method: str             # "auto" (detect the noise type) or a fixed filter for impulse-noisy images


class RetrievalConfig(TypedDict):
    """A preset: descriptor configuration and similarity measure."""
    descriptor: DescriptorConfig    # what to compute per painting
    similarity: str                 # name of the measure in src.retrieval (e.g. "l1", "chi2", "hellinger_kernel")


# ---- Global colour histogram presets, reused for fallback and comparisons.
# Week 2 method1 colour descriptor (16 bins, 4x4 blocks; weeks/week2/README.md); also the colour part of the W3 fusion.
COLOUR_HISTOGRAM: DescriptorConfig = {"kind": "1d", "color_spaces": ["HSV", "Lab", "rgb"], "bins": 16, "grid": 4,
                                  "gray_world": True, "contrast_stretch": True, "luminance_weight": 0.25}
# The two submitted Week 2 methods (W2 Tasks 1-2 descriptor search,
# weeks/week2/results/records/descriptors/comparison.md).
COLOUR_PRESETS: dict[str, RetrievalConfig] = {
    "method1": {"descriptor": COLOUR_HISTOGRAM, "similarity": "l1"},
    "method2": {"descriptor": {"kind": "1d", "color_spaces": ["HSV", "YCbCr", "rgb"], "bins": 32, "grid": 4,
                               "gray_world": True, "contrast_stretch": True, "luminance_weight": 1.0},
                "similarity": "l1"},
}

# ---- Noise detection and filtering on painting crops (Week 3 Task 1, weeks/week3/results/comparison.md).
# Impulse: grey levels from the 3x3 median for a pixel to be an outlier. Set by Mitali Raj with the Week 3 impulse
# detector (commit 3b2591c); no tuning experiment is recorded.
OUTLIER_THRESHOLD: int = 40
NOISE_THRESHOLD: float = 0.01          # impulse: block-median density of isolated outliers (detector kept as is)
SIGMA_THRESHOLD: float = 6.67          # grain: largest noise_sigma among the 287 clean BBDD images
GRAIN_FILTER: dict = {"method": "bilateral"}  # filter of grain-flagged images (src.denoise.remove_noise), W3 Task 1

# ---- Background removal parameters (Week 3 Task 3, validated one at a time on QSD2-W3:
# weeks/week3/results/task3/segmentation_parameters.csv). Week 4 reuses them.
SEGMENTATION: dict = {
    "image_sigma": 1.0,                # Gaussian smoothing of the image before the Lab conversion
    "lightness_weight": 0.5,           # weight of L in the Lab distance to the wall
    "border_fraction": 0.02,           # border strip (fraction of the short side) the wall plane is fitted on
    "retained_fraction": 0.8,          # border samples kept after each fit (smallest residuals)
    "fit_iterations": 3,               # fits of the wall-colour plane
    "threshold": 16,                   # L1 Lab distance above which a pixel is painting
    "closing_kernel_size": 5,          # closing of the colour mask (pixels)
    "min_area_fraction": 0.01,         # smallest painting / quadrilateral, fraction of the image
    "strong_threshold": 24,            # distance of a confident painting pixel
    "min_seed_fraction": 0.005,        # confident pixels a painting needs, fraction of the image
    "distance_sigma": 1.0,             # smoothing of the distance map for the refinement (fraction 0, floor 1.0)
    "canny_sigmas": (0.5, 1.0, 2.0, 4.0),  # Gaussian scales of the Canny quadrilateral search
    "canny_thresholds": (60, 180),     # low / high hysteresis thresholds of Canny
    "edge_closing": 3,                 # closing of the Canny edges (pixels)
    "polygon_epsilon": 0.01,           # polygon tolerance, fraction of the perimeter
    "min_iou": 0.6,                    # smallest IoU of a quadrilateral with the colour mask
    "max_strong_loss": 0.05,           # confident painting pixels a quadrilateral may remove
    "min_added_area_fraction": 0.01,   # additions larger than this (fraction of the painting) are checked
    "wall_threshold": 8,               # distance of a confident wall pixel
    "max_added_wall_fraction": 0.5,    # confident wall allowed in such an addition
}

# ---- Colour and texture components. Noise "auto" selects the filter per painting.
# Texture parameters chosen on QSD1-W3 in Week 3 Task 2 (weeks/week3/results/comparison.md).
FUSION_PARTS: dict[str, DescriptorConfig] = {
    "colour": {**COLOUR_HISTOGRAM, "denoise_method": "auto"},
    "lbp": {"kind": "texture", "texture": "lbp", "denoise_method": "auto",
            "params": {"grid": 4, "radii": (2,), "points": 8, "mapping": "none"}},
    "dct": {"kind": "texture", "texture": "dct", "denoise_method": "auto",
            "params": {"grid": 8, "n_coefs": 5, "normalize": "l2"}},
    "wavelet": {"kind": "texture", "texture": "wavelet", "denoise_method": "auto",
                "params": {"family": "db2", "levels": 4, "grid": 4, "normalize": "l2"}},
}
# Mean L1 distance between different database descriptors, used to balance each component
# (measured on BBDD in Week 3 Task 4, weeks/week3/results/comparison.md).
FUSION_SCALES: dict[str, float] = {"colour": 0.8998, "lbp": 10.7636, "dct": 5.8191, "wavelet": 7.7616}

# Name of the Week 3 submitted fusion (W3 Task 4 final method, weeks/week3/results/comparison.md).
GLOBAL_METHOD = "colour_lbp_dct_wavelet"
# The Week 3 method: four parts, equal weights, each divided by its scale, L1. Also the Week 4 global fallback.
GLOBAL_PRESET: RetrievalConfig = {
    "descriptor": {
        "kind": "combined", "weights": [1.0, 1.0, 1.0, 1.0],
        "scales": [FUSION_SCALES[p] for p in ("colour", "lbp", "dct", "wavelet")],
        "parts": [FUSION_PARTS[p] for p in ("colour", "lbp", "dct", "wavelet")],
    },
    "similarity": "l1",
}

# Global descriptor presets for the fallback and historical baseline comparisons.
PRESETS: dict[str, RetrievalConfig] = {**COLOUR_PRESETS, GLOBAL_METHOD: GLOBAL_PRESET}

# Frozen submitted recipes, week -> method name -> preset. Dataset selection never changes these descriptors.
# Week 1 recipes: weeks/week1/design_choices.md; Week 2: COLOUR_PRESETS; Week 3: GLOBAL_PRESET.
# None selects LocalMethod (DoG + RootSIFT, geometry and unknown rejection).
SUBMITTED_METHODS: dict[int, dict[str, RetrievalConfig | None]] = {
    1: {
        "method1": {"descriptor": {"kind": "1d", "color_spaces": ["HSV", "Lab", "rgb"],
                    "bins": 32, "grid": 1, "gray_world": True, "contrast_stretch": True,
                    "luminance_weight": 0.25}, "similarity": "hellinger_kernel"},
        "method2": {"descriptor": {"kind": "1d", "color_spaces": ["HSV", "YCbCr", "rgb"],
                    "bins": 32, "grid": 1, "gray_world": True, "contrast_stretch": True,
                    "luminance_weight": 1.0}, "similarity": "chi2"},
    },
    2: COLOUR_PRESETS,
    3: {"method1": GLOBAL_PRESET},
    4: {"method1": None},
}

# ---- Delivery and Week 4 local features (src/matching.py, src/local_pipeline.py, main.py).
TOP_K: int = 10                                    # ranking depth of every delivered painting (delivery format)
# Default paintings per query of each week's method (the weekly task statements: W3 up to two, W4 up to three).
MAX_PAINTINGS: dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 3}
# Week 4 segmentation: regions below this fraction of the photo's largest region are dropped before MAX_PAINTINGS.
MIN_REGION_FRACTION: float = 0.10  # drops labels/frame pieces; smallest real painting in dev sets is 20 % of the largest
# Generic MatchConfig defaults (the Task 2 comparisons start from them, weeks/week4/experiments/task2_metrics.py).
# Set by Mitali Raj in the first Week 4 matching module (commit 861446c); no tuning experiment is recorded.
# max_side / max_points fix the museum feature cache (longest side 900 px, 1000 strongest keypoints).
MATCH_DEFAULTS: dict = {"ratio": 0.8, "mutual": True, "reprojection_px": 5.0, "max_side": 900,
                        "max_points": 1000, "metric": "l2", "score": "inliers"}
# Submitted Week 4 method (LocalMethod): RootSIFT, ratio 0.75, cross-check, distinct inliers
# (W4 Task 2-I comparison, weeks/week4/results/task2/task2_results.md); unset keys come from MATCH_DEFAULTS.
LOCAL_MATCH: dict = {"ratio": 0.75, "mutual": True, "metric": "hellinger", "score": "distinct"}
UNKNOWN_THRESHOLD: int = 10                        # minimum local score (distinct inliers) to accept; chosen on QSD1-W4
# Inliers to recover an extra painting inside a merged region. Set by Albert Puga with the merged-region recovery
# (commit 5b65d07); no tuning experiment is recorded.
RECOVERY_MIN_INLIERS: int = 10
GLOBAL_RATIO: float = 0.85                         # fallback: best / second global distance must be below; QSD1-W4
