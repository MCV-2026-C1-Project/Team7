"""Retrieval presets (descriptor + similarity), selectable with `main.py --method <name>`.

- Week 1: global colour histograms (`WEEK1_METHODS`; not a preset, only the compatibility test uses them).
- Week 2: 4x4 block colour histograms (`method1`, `method2`).
- Week 3: texture alone (`texture_lbp`, `texture_dct`, `texture_wavelet`, Task 2), the teammates' fusion
  (`colour_lbp_dct`), colour + wavelet (`colour_wavelet`, the first choice, kept for the tables) and the
  **final method `colour_lbp_dct_wavelet`** (the default).
How every Week 3 parameter was chosen, and on which data: results/week3/comparison.md.
Every value chosen by experiment lives here (noise thresholds, segmentation parameters, fusion scales); the
internal constants of each algorithm stay next to its code.
"""

from typing import TypedDict


class DescriptorConfig(TypedDict, total=False):
    """Input of src.descriptors.compute_descriptor."""
    kind: str                       # "1d", "2d", "3d" (colour), "texture" or "combined"
    color_spaces: list[str]         # 1d colour histograms
    space: str                      # 2d / 3d joint histograms
    bins: int
    grid: int                       # N x N blocks (1 = global)
    pyramid_levels: int
    pyramid_weights: str
    gray_world: bool
    contrast_stretch: bool
    luminance_weight: float         # weight of the brightness histograms
    texture: str                    # "lbp", "dct" or "wavelet" (kind "texture")
    params: dict                    # keyword arguments of the texture descriptor
    parts: list["DescriptorConfig"]  # kind "combined": the descriptors to fuse
    weights: list[float]            # kind "combined": weight of each part
    scales: list[float]             # kind "combined": each part is divided by its scale
    denoise_method: str             # "auto" (detect the noise type) or a fixed filter for impulse-noisy images


class RetrievalConfig(TypedDict):
    """A preset: descriptor configuration and similarity measure."""
    descriptor: DescriptorConfig
    similarity: str


class Week1Config(TypedDict):
    """A Week 1 global-histogram method."""
    color_spaces: list[str]
    bins: int
    similarity: str
    gray_world: bool
    contrast_stretch: bool
    luminance_weight: float


# ---- Week 1: global 1D colour histograms (whole image, no blocks); tests/test_descriptors.py checks that
# compute_descriptor still gives the Week 1 descriptors. Not selectable in main.py.
WEEK1_METHODS: dict[str, Week1Config] = {
    "week1_method1": {"color_spaces": ["HSV", "Lab", "rgb"], "bins": 32, "similarity": "hellinger_kernel",
                      "gray_world": True, "contrast_stretch": True, "luminance_weight": 0.25},
    "week1_method2": {"color_spaces": ["HSV", "YCbCr", "rgb"], "bins": 32, "similarity": "chi2",
                      "gray_world": True, "contrast_stretch": True, "luminance_weight": 1.0},
}

# ---- Week 2: the Week 1 histograms in 4x4 blocks (method1 is the colour part of the fusions)
WEEK2_COLOUR: DescriptorConfig = {"kind": "1d", "color_spaces": ["HSV", "Lab", "rgb"], "bins": 16, "grid": 4,
                                  "gray_world": True, "contrast_stretch": True, "luminance_weight": 0.25}
WEEK2_METHODS: dict[str, RetrievalConfig] = {
    "method1": {"descriptor": WEEK2_COLOUR, "similarity": "l1"},
    "method2": {"descriptor": {"kind": "1d", "color_spaces": ["HSV", "YCbCr", "rgb"], "bins": 32, "grid": 4,
                               "gray_world": True, "contrast_stretch": True, "luminance_weight": 1.0},
                "similarity": "l1"},
}

# ---- Week 3, Task 1: noise detection (src/denoise.py). Chosen on QSD1-W3 / QSD2-W3 and the clean BBDD.
OUTLIER_THRESHOLD: int = 40            # impulse: grey levels from the 3x3 median for a pixel to be an outlier
NOISE_THRESHOLD: float = 0.01          # impulse: block-median density of isolated outliers
SIGMA_THRESHOLD: float = 6.67          # grain: largest noise_sigma among the 287 clean BBDD images
GRAIN_FILTER: dict = {"method": "bilateral"}  # filter of the images flagged as grain (src.denoise.remove_noise)

# ---- Week 3, Task 3: background removal (src/segmentation.py). Validated one by one by Albert Puga on the
# development sets (results/week3/task3/segmentation_parameters.csv, column "final").
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
    "canny_thresholds": (60, 180),
    "edge_closing": 3,                 # closing of the Canny edges (pixels)
    "polygon_epsilon": 0.01,           # polygon tolerance, fraction of the perimeter
    "min_iou": 0.6,                    # smallest IoU of a quadrilateral with the colour mask
    "max_strong_loss": 0.05,           # confident painting pixels a quadrilateral may remove
    "min_added_area_fraction": 0.01,   # additions larger than this (fraction of the painting) are checked
    "wall_threshold": 8,               # distance of a confident wall pixel
    "max_added_wall_fraction": 0.5,    # confident wall allowed in such an addition
}

# ---- Week 3: parts of the colour + texture fusion. Texture parameters chosen on QSD1-W3 and checked on
# QSD2-W3 (experiments/task2_texture.py). Noise "auto": 3x3 median for impulse noise, bilateral for grain,
# nothing for clean images (src/denoise.py).
FUSION_PARTS: dict[str, DescriptorConfig] = {
    "colour": {**WEEK2_COLOUR, "denoise_method": "auto"},
    "colour_nlm": {**WEEK2_COLOUR, "denoise_method": "median_nlm"},
    "lbp": {"kind": "texture", "texture": "lbp", "denoise_method": "auto",
            "params": {"grid": 4, "radii": (2,), "points": 8, "mapping": "none"}},
    "dct": {"kind": "texture", "texture": "dct", "denoise_method": "auto",
            "params": {"grid": 8, "n_coefs": 5, "normalize": "l2"}},
    "wavelet": {"kind": "texture", "texture": "wavelet", "denoise_method": "auto",
                "params": {"family": "db2", "levels": 4, "grid": 4, "normalize": "l2"}},
}
# Mean L1 distance between two BBDD descriptors of each part (results/week3/task4/fusion_scales.csv)
FUSION_SCALES: dict[str, float] = {"colour": 0.8998, "lbp": 10.7636, "dct": 5.8191, "wavelet": 7.7616}

WEEK3_METHODS: dict[str, RetrievalConfig] = {
    # Task 2: texture alone (no colour), the parameters chosen in the texture search
    "texture_lbp": {"descriptor": FUSION_PARTS["lbp"], "similarity": "l1"},
    "texture_dct": {"descriptor": FUSION_PARTS["dct"], "similarity": "l1"},
    "texture_wavelet": {"descriptor": FUSION_PARTS["wavelet"], "similarity": "l1"},
    # The teammates' fusion (branch Mitali_W3) with its own parameters, kept only for the comparison tables
    "colour_lbp_dct": {
        "descriptor": {"kind": "combined", "weights": [1.0, 1.0, 1.0], "scales": [0.8998, 32.32, 239982.0],
                       "parts": [{**WEEK2_COLOUR, "denoise_method": "median_nlm"},
                                 {"kind": "texture", "texture": "lbp", "denoise_method": "median"},
                                 {"kind": "texture", "texture": "dct", "denoise_method": "median"}]},
        "similarity": "l1"},
    # First choice: Week 2 colour + db2 wavelet, equal weights, each part divided by its scale, L1. Chosen by
    # the selection rule (QSD1-W3 / QSD2-W3 with GT masks, fewest parts among ties, tie broken on QSD1-W2 /
    # QSD2-W2; experiments/task4_retrieval.py); kept as an alternative for the tables.
    "colour_wavelet": {
        "descriptor": {"kind": "combined", "weights": [1.0, 1.0],
                       "scales": [FUSION_SCALES["colour"], FUSION_SCALES["wavelet"]],
                       "parts": [FUSION_PARTS["colour"], FUSION_PARTS["wavelet"]]},
        "similarity": "l1"},
    # FINAL METHOD: Week 2 colour + LBP + DCT + wavelet, equal weights, each part divided by its scale, L1.
    # It tied with colour_wavelet on the selection sets (QSD1-W3, QSD2-W3 with GT masks) and on QSD1-W2, and
    # with our own masks (the real delivery situation) it gets 39/39 on QSD2-W3 against 38/39. Adopted after
    # seeing that result (results/week3/comparison.md).
    "colour_lbp_dct_wavelet": {
        "descriptor": {"kind": "combined", "weights": [1.0, 1.0, 1.0, 1.0],
                       "scales": [FUSION_SCALES[p] for p in ("colour", "lbp", "dct", "wavelet")],
                       "parts": [FUSION_PARTS[p] for p in ("colour", "lbp", "dct", "wavelet")]},
        "similarity": "l1"},
}

# Every preset of main.py --method (one ranking per painting, Top-10 export).
PRESETS: dict[str, RetrievalConfig] = {**WEEK2_METHODS, **WEEK3_METHODS}
FINAL_METHOD = "colour_lbp_dct_wavelet"
