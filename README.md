# Museum Painting Retrieval — Team 7

<p align="center">
  <img src="https://raw.githubusercontent.com/MCV-2026-C1-Project/Team7/submission/Week%203/assets/week3_retrieval.jpg" alt="Week 3 example: a query photo with two paintings found, each painting crop and its rank-1 match in the database" width="820">
</p>

<p align="center">
  <strong>Content-Based Image Retrieval · Computer Vision · MCV C1</strong>
  <br>
  Week 1 baseline, Week 2 and Week 3 submissions
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Task-Image%20Retrieval-0A7B83" alt="Image retrieval">
  <img src="https://img.shields.io/badge/Delivery-Top--10%20Rankings-2E8B57" alt="Top 10 rankings">
</p>

A content-based image retrieval pipeline for matching museum painting photographs against a reference database. The project explores global and spatial color descriptors, background segmentation, histogram distances, and, from Week 3, noise filtering, texture descriptors, and queries with more than one painting, with reproducible retrieval evaluation.

The Week 2 submission provides two fixed retrieval methods, optional development evaluation, validated Top-10 ranking exports, and predicted foreground masks for queries containing background. The Week 3 submission, in [`Week 3/`](Week%203), adds noise detection and filtering, LBP, DCT and wavelet texture descriptors fused with the Week 2 colour descriptor, and one Top-10 ranking for each of up to two paintings per query.

---

## Navigation

- [Week 3](#week-3)
- [Overview](#overview)
- [Methods](#methods)
- [Results](#development-results)
- [Installation](#installation)
- [Dataset setup](#dataset-setup)
- [Run Week 2](#run-week-2)
- [Run Week 3](#run-week-3)
- [Evaluate on development data](#development-evaluation)
- [Week 1 compatibility](#week-1-compatibility)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Reproducibility and submission format](#reproducibility-and-submission-format)

## Week 3

Week 3 lives in the [`Week 3/`](Week%203) folder, with its own README, entry point (`python main.py evaluate | predict | validate`) and sources. The default method, `colour_lbp_dct_wavelet`, works in four steps:

1. **Noise:** two detectors decide, once per image, whether it is noisy. Impulse noise gets a 3 × 3 median, additive grain a bilateral filter, and clean images are left untouched.
2. **Segmentation (queries with background):** the wall colour is estimated from the image border, and up to two paintings are kept, ordered from left to right.
3. **Descriptor:** the Week 2 colour histograms are fused with LBP, DCT and db2-wavelet texture descriptors computed on the grey painting. Each part is divided by its mean L1 distance between two database images, then the parts are compared with equal weights and the L1 distance.
4. **Output:** one Top-10 ranking per painting.

Texture ignores the random hue changes that break the colour histograms, and colour and texture fail on different paintings, so their fusion is stronger than either one alone.

| Development set (Week 3) | Week 2 colour alone | Week 3 final method |
|---|---:|---:|
| QSD1-W3, cropped queries (30), top-1 | 26/30 | **30/30** |
| QSD2-W3, our masks (39 paintings), top-1 | 32/39 | **39/39** |
| QSD1-W2 / QSD2-W2 (30 + 30), top-1 | 30/30, 26/30 | **30/30, 30/30** |

Segmentation on QSD2-W3 (union mask against the ground truth): precision 0.969, recall 0.987, F1 0.978, with the right number of paintings in all 30 images. These are development results that the settings were chosen on, not estimates of test accuracy, and one painting is worth about 3 points. Details, parameters and commands: [`Week 3/README.md`](Week%203/README.md).

## Overview

The system retrieves the most visually similar reference paintings for each query image using handcrafted color and texture descriptors.

**Main components**

- **Color descriptors:** HSV, Lab, YCbCr, normalized RGB, global histograms, spatial blocks, and pyramids.
- **Texture descriptors (Week 3):** block LBP, DCT, and wavelet descriptors on the grey painting, fused with the colour descriptor.
- **Noise filtering (Week 3):** impulse and grain detectors; only images that look noisy are filtered.
- **Preprocessing:** gray-world color correction and percentile contrast stretching.
- **Background segmentation:** color-distance-based foreground extraction for queries containing background; from Week 3, up to two paintings per query.
- **Retrieval:** histogram comparison and ranked database matching.
- **Evaluation:** retrieval mAP@1, mAP@5, mAP@10, and available segmentation metrics.
- **Submission:** validated Python pickle rankings (one list per painting from Week 3) and native-resolution binary PNG masks.

The command-line interface runs inference and optional evaluation directly from source code. Development notebooks and cached experiment outputs are not required.

## Methods

The two selected Week 2 methods use the same preprocessing and retrieval pipeline structure, with different descriptor configurations.

| Configuration | Method 1 | Method 2 |
|---|---|---|
| Descriptor | HSV + Lab + normalized RGB | HSV + YCbCr + normalized RGB |
| Spatial layout | 4 × 4 blocks | 4 × 4 blocks |
| Histogram bins | 16 per channel | 32 per channel |
| Distance | L1 | L1 |
| Luminance weight | 0.25 | 1.0 |
| Descriptor dimension | 2,304 | 4,608 |

Both methods use gray-world correction and percentile contrast stretching during descriptor preprocessing. Both use the same segmentation configuration for QS2.

The historical Week 2 Hellinger configuration remains available for comparison, but it is not one of the two submitted methods. Week 1 presets are explicitly named `week1_method1` and `week1_method2`.

<details>
<summary><strong>How does retrieval work?</strong></summary>

1. Load the query painting and reference database.
2. Apply the selected query preprocessing mode.
3. Compute the configured descriptor for the query and database images.
4. Compare the query descriptor against the database descriptors using the selected distance.
5. Rank database images from best to worst match.
6. Validate the ranking and export the requested artifacts.

Each query is assumed to contain one painting. Database and query images are processed using the same descriptor configuration.

</details>

<details>
<summary><strong>How does background segmentation work?</strong></summary>

Segmentation uses color distance in Lab space, channel-wise border medians, L2 distance, and lightness weighting.

The default configuration uses:

- Lightness weight: `0.75`
- Border width: `4%` of the shorter image dimension
- Distance threshold: `18`
- Morphological closing: `5 × 5`
- Hole filling and largest connected-component selection

Images are not resized. Gray-world correction is applied during descriptor preprocessing, not during segmentation.

</details>

## Development results

The following results are from the reported development experiments and are not estimates of performance on unseen test images.

| Task | Reported result |
|---|---|
| Descriptor families | Blocks and pyramids achieve QSD1 mAP@5 of 1.0000 in the reported experiments |
| Cropped retrieval — Method 1 | 30/30 top-1 matches; mAP@5 = 1.0000 |
| Cropped retrieval — Method 2 | 29/30 top-1 matches; mAP@5 = 0.9778 |
| Background segmentation | Mean precision = 0.9646; mean recall = 0.9833; mean F1 = 0.9733 |
| Full QS2 pipeline — Method 1 | 27/30 top-1 matches; mAP@5 = 0.9361 |
| Full QS2 pipeline — Method 2 | 28/30 top-1 matches; mAP@5 = 0.9500 |

### Selected method comparison

| Finalist | QSD1 top-1 | QSD1 mAP@5 | QSD2 top-1 | QSD2 mAP@5 | Mean mAP@5 |
|---|---:|---:|---:|---:|---:|
| HSV + Lab + RGB, 16 bins, L1 | 30/30 | 1.0000 | 27/30 | 0.9361 | **0.9681** |
| HSV + YCbCr + RGB, 32 bins, L1 | 29/30 | 0.9778 | 28/30 | 0.9500 | 0.9639 |
| HSV + Lab + RGB, 16 bins, Hellinger | 29/30 | 0.9667 | 27/30 | 0.9344 | 0.9506 |

The finalists were ranked by equal-weight mean mAP@5 across QSD1 and QSD2, with worst-set mAP@5, mean mAP@1, and descriptor size used as tie-breakers. QSD2 selection uses predicted masks.

Method 1 provides the highest mean mAP@5 and the smaller descriptor. Method 2 performs better on QSD2. Both are retained as delivery methods.

<details>
<summary><strong>Additional evaluation notes</strong></summary>

- QSD2 retrieval with ground-truth masks is a separate diagnostic, not the full pipeline score.
- The reported QSD2 diagnostic mAP@5 is 0.9556 for Method 1 and 0.9667 for Method 2.
- The segmentation evaluation reports a minimum F1 of 0.9013, with no images below 0.90 in the evaluated development set.
- The experiments compare selected configurations; they do not establish global optimality.

</details>

## Installation

**Requirements:** Python 3.10 or newer.

Clone the repository and enter the directory of the week you want to run (Week 2 shown; use `"Week 3"` for Week 3):

```bash
git clone https://github.com/MCV-2026-C1-Project/Team7.git
cd Team7/"Week 2"
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
python -m pip install -e .
```

In `Week 3` you can also use `python -m pip install -r requirements.txt`. Inference and evaluation require NumPy and OpenCV. The project configuration declares the installation dependencies.

To inspect the available command-line options:

```bash
python main.py --help
```

The installed command is `cbir`, equivalent to invoking `python main.py`.

> **Dataset note:** The dataset is distributed separately and is not included in Git. Download or obtain the course dataset before running the commands below.

## Dataset setup

The default expected layout is:

```text
data/
├── BBDD/       # Reference database: bbdd_00000.jpg, ...
├── qsd1_w1/    # Week 1 development queries and annotations
├── qsd1_w2/    # Cropped Week 2 development queries
├── qsd2_w2/    # Week 2 queries with background and binary masks
├── qst1_w1/    # Week 1 test queries
├── qst1_w2/    # Cropped Week 2 test queries
├── qst2_w2/    # Week 2 test queries with background
├── qsd1_w3/    # Week 3 cropped development queries, augmentations and clean originals
├── qsd2_w3/    # Week 3 queries with up to two paintings and binary masks
├── qst1_w3/    # Week 3 cropped test queries
└── qst2_w3/    # Week 3 test queries with background
```

Development datasets include the required annotations, such as `gt_corresps.pkl`, and segmentation masks where applicable.

For custom locations, use the supported path arguments:

```bash
python main.py --help
```

Consult the CLI help for the exact options available in your installed version, including `--corpus_data` and `--query_data`.

## Run Week 2

The background mode is explicit and must match the input type:

- **QS1:** Queries are already cropped. Use `--background none`.
- **QS2:** Queries contain background. Use `--background remove`.

The pipeline does not infer the mode from the directory name.

### Method 1

```bash
# QS1: already-cropped paintings
python main.py \
  --method method1 \
  --background none \
  --query_data data/qst1_w2 \
  --output results/week2/QST1/method1/result.pkl

# QS2: paintings with background
python main.py \
  --method method1 \
  --background remove \
  --query_data data/qst2_w2 \
  --output results/week2/QST2/method1/result.pkl
```

### Method 2

```bash
# QS1: already-cropped paintings
python main.py \
  --method method2 \
  --background none \
  --query_data data/qst1_w2 \
  --output results/week2/QST1/method2/result.pkl

# QS2: paintings with background
python main.py \
  --method method2 \
  --background remove \
  --query_data data/qst2_w2 \
  --output results/week2/QST2/method2/result.pkl
```

These four commands recreate the Week 2 delivery artifacts. Each method retains the same descriptor and retrieval parameters across the relevant datasets; only the input background mode changes.

## Run Week 3

Week 3 has its own entry point with three commands. The default method is `colour_lbp_dct_wavelet`.

```bash
cd "Week 3"

# Development evaluation (retrieval mAP@1/5/10; QSD2 sets also mask precision / recall / F1)
python main.py evaluate --set qsd1_w3            # cropped queries, no background removal
python main.py evaluate --set qsd2_w3            # background removal, up to two paintings
python main.py evaluate --set qsd1_w2            # also qsd2_w2

# Test delivery (QST1 and QST2) and its format check
python main.py predict --out results/week3
python main.py validate --delivery results/week3
```

```text
results/week3/
├── QST1/method1/result.pkl
└── QST2/method1/
    ├── result.pkl
    ├── 00000.png
    └── ...
```

Other presets (`--method`): `method1` / `method2` (Week 2), `texture_lbp` / `texture_dct` / `texture_wavelet` (texture alone), `colour_lbp_dct` and `colour_wavelet` (earlier fusions).

## Development evaluation

Use `--evaluate` to evaluate on annotated Week 1 and Week 2 development datasets (Week 3 uses `python main.py evaluate`, above).

```bash
# Method 1: cropped development queries
python main.py --method method1 \
  --background none \
  --query_data data/qsd1_w2 \
  --evaluate

# Method 1: queries with background
python main.py --method method1 \
  --background remove \
  --query_data data/qsd2_w2 \
  --evaluate

# Method 2: cropped development queries
python main.py --method method2 \
  --background none \
  --query_data data/qsd1_w2 \
  --evaluate

# Method 2: queries with background
python main.py --method method2 \
  --background remove \
  --query_data data/qsd2_w2 \
  --evaluate

# Week 1 development baseline
python main.py --method week1_method1 \
  --query_data data/qsd1_w1 \
  --evaluate
```

Evaluation reports retrieval mAP@1, mAP@5, mAP@10, and available segmentation metrics.

Without `--evaluate`, inference does not read ground-truth annotations.

## Week 1 compatibility

The original Week 1 presets remain available from the same CLI:

```bash
python main.py \
  --method week1_method1 \
  --query_data data/qst1_w1 \
  --output results/week1/QST1/method1/result.pkl

python main.py \
  --method week1_method2 \
  --query_data data/qst1_w1 \
  --output results/week1/QST1/method2/result.pkl
```

The original Week 1 preset parameters are unchanged; only the CLI names include the `week1_` prefix. Custom global histograms are also supported.

The Week 2 project directory contains its own entry point and source modules, so it can be run from its directory. Keep the root repository's Week 1 and Week 2 folders as siblings.

## Project structure

```text
Week 2/
├── main.py                 # CLI and submission export
├── README.md               # Documentation and run instructions
├── pyproject.toml           # Dependencies and CLI installation
├── .gitignore
├── assets/
│   ├── unified_selection.png
│   └── unified_selection.csv
└── src/
    ├── __init__.py
    ├── config.py            # Descriptor presets
    ├── data_loading.py      # Image and annotation loading
    ├── segmentation.py      # Foreground mask prediction
    ├── descriptors.py       # Color and spatial descriptors
    ├── similarities.py      # Histogram distances
    ├── retrieval.py         # Database ranking
    ├── pipeline.py          # Inference orchestration
    ├── evaluate.py          # Retrieval and mask metrics
    └── submission.py        # Validated PKL and PNG export
```

```text
Week 3/
├── main.py                 # CLI: evaluate / predict / validate
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── assets/
│   └── week3_retrieval.jpg  # example shown at the top of this README
└── src/
    ├── config.py           # presets, noise thresholds, fusion scales, segmentation parameters
    ├── denoise.py          # noise detection (impulse / grain) and filters
    ├── colour.py           # colour histograms of Weeks 1 and 2
    ├── texture.py          # LBP, DCT and wavelet descriptors
    ├── descriptors.py      # crop + noise filter, then colour, texture or fused descriptor
    ├── segmentation.py     # background removal, up to two paintings
    ├── pipeline.py         # images -> masks -> descriptors -> rankings
    └── data_loading.py, similarities.py, retrieval.py, evaluate.py, submission.py
```

The source code separates descriptor computation, segmentation, retrieval, evaluation, and submission generation. Supplementary assets are optional and are not loaded by the inference pipeline.

## Reproducibility and submission format

In Weeks 1 and 2 the pipeline assumes one painting per query.

**Retrieval output**

- A Python `list[list[int]]` serialized as a pickle file.
- One row per query in sorted filename order.
- Exactly ten distinct database IDs per row, ranked from best to worst.

**Week 3 retrieval output**

- A Python `list[list[list[int]]]` with one extra list level so a query can hold more than one painting: queries in sorted filename order, then one list per painting from left to right, then exactly ten distinct database IDs.
- QST1 has exactly one painting per query (`[[[ids]], ...]`); QST2 has one or two (`[[[ids], [ids]], ...]`).

**Segmentation output**

- QS2 masks are saved alongside `result.pkl`.
- Each mask uses the original query filename stem with a `.png` extension.
- Masks retain the input image dimensions and contain only `uint8` values `0` and `255`.
- With two paintings, the mask of a query is the union of both.

Predictions are validated before export. The saved pickle is reopened and checked. Empty or too-small masks count as evaluation failures and prevent an incomplete submission from being exported.

Inference writes only explicitly requested submission artifacts. It does not automatically generate CSV reports, figures, or development reports. The `results/` directory is created when predictions are exported.

## Design decisions

- **Two fixed delivery methods:** Each method uses identical retrieval parameters across the relevant development and test datasets.
- **Explicit segmentation mode:** Background removal is enabled only when requested.
- **Consistent descriptors:** Query and reference images use the same descriptor configuration.
- **Development-only selection:** The two delivery methods are selected using development results, not test accuracy.
- **Validated exports:** Ranking format, uniqueness, and segmentation-mask constraints are checked before delivery.
- **Filter only what is noisy (Week 3):** the noise detectors decide per image; clean images are never filtered.
- **Equal-weight fusion (Week 3):** each descriptor is divided by its typical distance in the database, so no part dominates.
- **Separated concerns:** The CLI delegates algorithms to modules under `src/`.

---

<p align="center">
  <strong>Team 7 · Museum Painting Retrieval</strong>
  <br>
  <sub>MCV C1 — Weeks 1 to 3 · Colour and texture image retrieval</sub>
</p>
