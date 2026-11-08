# Week 4 detector–descriptor study

Open `weeks/week4/experiments/local_descriptors.ipynb` and run all cells. The notebook contains the complete diagnostic study: matching, RANSAC, fusion, evaluation and result export. Missing or outdated measurements are recomputed automatically; set `RERUN_STUDY=True` to force a fresh run.

`src/local_descriptors.py` contains only descriptor extraction and point-frame preparation. Production extraction tests are in `tests/test_local_descriptors.py`; experiment-specific checks are embedded in the notebook.

The study uses the Harris, Harris-Laplace and DoG detectors with SIFT, ORB and BRISK. All 30 QSD1-W4 images are tested under a known 90-degree rotation and a resize to 65%. Known query–museum pairs and one unrelated control per known query provide additional geometric diagnostics. There is no database search, unknown rejection, threshold optimization or production-pipeline integration.

Settings and a fingerprint of the extraction code and notebook helper cells are stored in `metadata.json`. The notebook checks the fingerprint before loading the saved results.

- `extraction.csv`: detected/described point counts, detection and description time, and descriptor storage. Description includes point-frame preparation. Timing is one measurement per image, aggregated across images.
- `synthetic.csv`: tentative matches, correct endpoints within 3 pixels of the known transformation, precision, and the fraction of unique source points with at least one correct match. Multiple correct target pairs for one source point count once in this coverage measure. Rows retain the query ID and transformation.
- `pairs.csv`: tentative matches and homography RANSAC consensus for annotated reference pairs and unrelated controls. Inlier counts are not independent ground-truth correctness labels.

All comparisons use a fixed ratio threshold of 0.75. ORB receives external points and uses a fixed original-image-scale patch. Harris and Harris-Laplace orientations are estimated by a common adapter; Harris scales are mapped into the SIFT pyramid convention. BRISK estimates its own orientation. No ORB detector is used.

The combination study includes two constructions from existing SIFT and BRISK descriptors:

- **Paired:** the intersection of surviving input points, with aligned SIFT and BRISK blocks. A fixed 50/50 normalized distance is compared with each block alone on that same subset.
- **Union:** all points described by either method; independent accepted correspondences are merged and identical original point-ID pairs are counted once. Conflicting targets are retained, so additional correct and false matches can both increase. No mixed descriptor distance is assigned.

The point cap is an equal upper bound rather than a guarantee of equal point counts. Raw counts, precision and normalized yield must be interpreted together. The study evaluates these baseline configurations, not the globally best settings for each algorithm.
