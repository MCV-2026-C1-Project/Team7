"""Retrieval presets; segmentation defaults live in segmentation.py.

Week 2 delivery uses method1 and method2 (both L1, fixed across datasets).
The non-submitted Hellinger comparison is explicitly named week2_hellinger.
The original Week 1 presets are preserved as week1_method1 / week1_method2.
"""

from typing import TypedDict


class DescriptorConfig(TypedDict, total=False):
    kind: str
    color_spaces: list[str]
    space: str
    bins: int
    grid: int
    pyramid_levels: int
    pyramid_weights: str
    gray_world: bool
    contrast_stretch: bool
    luminance_weight: float


class RetrievalConfig(TypedDict):
    descriptor: DescriptorConfig
    similarity: str


class Week1Config(TypedDict):
    color_spaces: list[str]
    bins: int
    similarity: str
    gray_world: bool
    contrast_stretch: bool
    luminance_weight: float


METHODS: dict[str, Week1Config] = {
    "week1_method1": {"color_spaces": ["HSV", "Lab", "rgb"], "bins": 32, "similarity": "hellinger_kernel",
                "gray_world": True, "contrast_stretch": True, "luminance_weight": 0.25},
    "week1_method2": {"color_spaces": ["HSV", "YCbCr", "rgb"], "bins": 32, "similarity": "chi2",
                "gray_world": True, "contrast_stretch": True, "luminance_weight": 1.0},
}

WEEK2_METHODS: dict[str, RetrievalConfig] = {
    # Final Week 2 submissions. Descriptor parameters are fixed for both query sets.
    "method1": {
        "descriptor": {"kind": "1d", "color_spaces": ["HSV", "Lab", "rgb"],
                       "bins": 16, "grid": 4, "gray_world": True,
                       "contrast_stretch": True, "luminance_weight": 0.25},
        "similarity": "l1",
    },
    "method2": {
        "descriptor": {"kind": "1d", "color_spaces": ["HSV", "YCbCr", "rgb"],
                       "bins": 32, "grid": 4, "gray_world": True,
                       "contrast_stretch": True, "luminance_weight": 1.0},
        "similarity": "l1",
    },
    # Non-submitted Week 2 Hellinger finalist, kept for comparison only.
    "week2_hellinger": {
        "descriptor": {"kind": "1d", "color_spaces": ["HSV", "Lab", "rgb"],
                       "bins": 16, "grid": 4, "gray_world": True,
                       "contrast_stretch": True, "luminance_weight": 0.25},
        "similarity": "hellinger_kernel",
    },
}

DEFAULT_WEEK2_METHOD = "method1"
