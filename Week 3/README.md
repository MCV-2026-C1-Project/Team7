# Museum painting retrieval — Team 7, Week 3

Painting retrieval for MCV C1. Week 3 adds noisy queries, random hue changes and up to two paintings per query.
Queries are denoised only when they look noisy, the background is removed painting by painting, and each painting is
described by its Week 2 colour histograms fused with three texture descriptors (LBP, DCT, wavelet). Every painting
gets its own Top-10 ranking, ordered left to right.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/MCV-2026-C1-Project/Team7.git
cd Team7/"Week 3"
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Only NumPy and OpenCV are needed. `python -m pip install -e .` also installs the `cbir` command (same as `python main.py`).

## Data

The dataset is not included in Git. Commands read `data/` next to `main.py` (`--data <folder>` for another place).
Course folders that unpack one level deeper (`data/qsd1_w3/qsd1_w3/`) are found automatically.

```text
data/
├── BBDD/        # bbdd_00000.jpg ...
├── qsd1_w3/     # 30 cropped development queries, gt_corresps.pkl, augmentations.pkl, non_augmented/
├── qsd2_w3/     # 30 development queries with background, GT masks <id>.png, gt_corresps.pkl
├── qsd1_w2/     # Week 2 cropped development queries (optional, extra check)
├── qsd2_w2/     # Week 2 queries with background and masks (optional, extra check)
├── qst1_w3/     # 50 cropped test queries
└── qst2_w3/     # 30 test queries with background
```

## Final method (`--method colour_lbp_dct_wavelet`)

| Step | What | Code |
| --- | --- | --- |
| Noise (unknown model) | two detectors, once per image: impulse (isolated outlier pixels) → 3×3 median; additive grain (Donoho MAD of the Haar diagonal band) → bilateral filter; clean images untouched | `src/denoise.py` |
| Segmentation (QS2) | Lab distance to a wall-colour plane fitted on the image border, closing, hole filling, up to two components of at least 1% of the image, refined with convex quadrilaterals; paintings ordered left to right | `src/segmentation.py` |
| Colour | Week 2 `method1`: HSV + Lab + normalised rgb, 4×4 blocks, 16 bins, gray-world, contrast stretch, brightness weight 0.25, statistics from the painting mask | `src/colour.py` |
| Texture | on the 256×256 grey painting: LBP (radius 2, 8 neighbours, 4×4 blocks); DCT (8×8 blocks, 5 zig-zag coefficients, unit norm); db2 wavelet (4 levels, 4×4 blocks, mean absolute value and std of each detail band, unit norm) | `src/texture.py` |
| Fusion | each part divided by its mean L1 distance between two database images (`FUSION_SCALES`), equal weights, L1 distance | `src/config.py` |

The noise detectors and the values of the fusion are fixed in `src/config.py`. How each value was chosen (experiment scripts,
tests and result tables) is in the group repository, `PolPerezPrades/D1-C1-Group7`, branch `week3-final`; comments in
`src/` that name `experiments/`, `tests/` or `results/week3/` refer to it. The retrieval query is always one
painting: the cropped image for QS1, or each detected painting for QS2.

## Development results

Not estimates of test accuracy: the settings were chosen on these sets.

| Set | Top-1 | mAP@1 | mAP@5 |
| --- | --- | --- | --- |
| QSD1-W3, cropped queries (30) | 30/30 | 1.000 | 1.000 |
| QSD2-W3, our masks (39 paintings) | 39/39 | 1.000 | 1.000 |
| QSD1-W2 (30) | 30/30 | 1.000 | 1.000 |
| QSD2-W2, our masks (30) | 30/30 | 1.000 | 1.000 |

Segmentation on QSD2-W3 (union mask against the ground truth): precision 0.969, recall 0.987, F1 0.978, the right
number of paintings in all 30 images. The Week 2 colour descriptor alone reaches 26/30 on QSD1-W3 and 32/39 on QSD2-W3.

## Run

`colour_lbp_dct_wavelet` is the default method, so `--method` can be omitted.

```bash
# Development evaluation (retrieval mAP@1/5/10; QSD2 sets also mask precision / recall / F1)
python main.py evaluate --set qsd1_w3            # cropped queries, no background removal
python main.py evaluate --set qsd2_w3            # background removal, up to two paintings
python main.py evaluate --set qsd1_w2            # also qsd2_w2

# Test delivery (QST1 and QST2) and its format check
python main.py predict --out results/week3
python main.py validate --delivery results/week3
```

This creates:

```text
results/week3/
├── QST1/method1/result.pkl
└── QST2/method1/
    ├── result.pkl
    ├── 00000.png
    └── ...
```

Both `result.pkl` files have three levels: queries in sorted filename order, then one list per painting (left to right),
then ten distinct Python-int database ids from best to worst match. QST1 has exactly one painting per query
(`[[[ids]], ...]`); QST2 has one or two (`[[[ids], [ids]], ...]`). The QST2 masks are `<query>.png` beside the PKL,
at the image size, with values 0 and 255 (the union of all the paintings of the query). `validate` checks every file.

Other presets (`--method`): `method1` / `method2` (Week 2), `texture_lbp` / `texture_dct` / `texture_wavelet`
(texture alone), `colour_lbp_dct` (earlier fusion), `colour_wavelet` (colour + wavelet).

## Project structure

```text
Week 3/
├── main.py                 # CLI: evaluate / predict / validate
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── assets/
│   └── week3_retrieval.jpg  # example: two paintings found in one photo and their rank-1 matches
└── src/
    ├── config.py           # presets, noise thresholds, fusion scales, segmentation parameters
    ├── data_loading.py     # images, masks, ground truth, dataset folders
    ├── denoise.py          # noise detection (impulse / grain) and filters
    ├── colour.py           # Weeks 1-2 colour histograms (global, blocks, pyramid)
    ├── texture.py          # LBP, DCT and wavelet descriptors
    ├── descriptors.py      # crop + noise filter, then colour, texture or fused descriptor
    ├── segmentation.py     # background removal, up to two paintings
    ├── pipeline.py         # images -> masks -> descriptors -> rankings
    ├── similarities.py     # histogram / vector distances
    ├── retrieval.py        # database ranking
    ├── evaluate.py         # retrieval and mask metrics
    └── submission.py       # result.pkl / PNG export and the delivery format check
```

