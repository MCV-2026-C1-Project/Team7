# Museum Painting Retrieval — Team 7

<p align="center">
  <img src="assets/unified_selection.png" alt="Museum painting retrieval — method selection overview" width="820">
</p>

<p align="center">
  <strong>Content-Based Image Retrieval · Computer Vision · MCV C1</strong>
  <br>
  Week 1 baseline and Week 2 submission
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Task-Image%20Retrieval-0A7B83" alt="Image retrieval">
  <img src="https://img.shields.io/badge/Delivery-Top--10%20Rankings-2E8B57" alt="Top 10 rankings">
</p>

A color-based image retrieval pipeline for matching museum painting photographs against a reference database. The project explores global and spatial color descriptors, background segmentation, histogram distances, and reproducible retrieval evaluation.

The Week 2 submission provides two fixed retrieval methods, optional development evaluation, validated Top-10 ranking exports, and predicted foreground masks for queries containing background.

---

## Navigation

- [Overview](#overview)
- [Methods](#methods)
- [Results](#development-results)
- [Installation](#installation)
- [Dataset setup](#dataset-setup)
- [Run Week 2](#run-week-2)
- [Evaluate on development data](#development-evaluation)
- [Week 1 compatibility](#week-1-compatibility)
- [Project structure](#project-structure)
- [Design decisions](#design-decisions)
- [Reproducibility and submission format](#reproducibility-and-submission-format)

## Overview

The system retrieves the most visually similar reference paintings for each query image using handcrafted color descriptors.

**Main components**

- **Color descriptors:** HSV, Lab, YCbCr, normalized RGB, global histograms, spatial blocks, and pyramids.
- **Preprocessing:** gray-world color correction and percentile contrast stretching.
- **Background segmentation:** color-distance-based foreground extraction for queries containing background.
- **Retrieval:** histogram comparison and ranked database matching.
- **Evaluation:** retrieval mAP@1, mAP@5, mAP@10, and available segmentation metrics.
- **Submission:** validated Python pickle rankings and native-resolution binary PNG masks.

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

Clone the repository and enter the Week 2 directory:

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

Inference and evaluation require NumPy and OpenCV. The project configuration declares the installation dependencies.

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
└── qst2_w2/    # Week 2 test queries with background
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

## Development evaluation

Use `--evaluate` to evaluate on annotated development datasets.

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

The source code separates descriptor computation, segmentation, retrieval, evaluation, and submission generation. Supplementary assets are optional and are not loaded by the inference pipeline.

## Reproducibility and submission format

The pipeline assumes one painting per query.

**Retrieval output**

- A Python `list[list[int]]` serialized as a pickle file.
- One row per query in sorted filename order.
- Exactly ten distinct database IDs per row, ranked from best to worst.

**Segmentation output**

- QS2 masks are saved alongside `result.pkl`.
- Each mask uses the original query filename stem with a `.png` extension.
- Masks retain the input image dimensions and contain only `uint8` values `0` and `255`.

Predictions are validated before export. The saved pickle is reopened and checked. Empty or too-small masks count as evaluation failures and prevent an incomplete submission from being exported.

Inference writes only explicitly requested submission artifacts. It does not automatically generate CSV reports, figures, or development reports. The `results/` directory is created when predictions are exported.

## Design decisions

- **Two fixed delivery methods:** Each method uses identical retrieval parameters across the relevant development and test datasets.
- **Explicit segmentation mode:** Background removal is enabled only when requested.
- **Consistent descriptors:** Query and reference images use the same descriptor configuration.
- **Development-only selection:** The two delivery methods are selected using development results, not test accuracy.
- **Validated exports:** Ranking format, uniqueness, and segmentation-mask constraints are checked before delivery.
- **Separated concerns:** The CLI delegates algorithms to modules under `src/`.

---

<p align="center">
  <strong>Team 7 · Museum Painting Retrieval</strong>
  <br>
  <sub>MCV C1 — Weeks 1 and 2 · Color-based image retrieval</sub>
</p>
