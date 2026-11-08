# Week 4: local-feature matching and unknown decisions

Keypoints and descriptors are extracted by `src/keypoints.py` and `src/local_descriptors.py`;
matching is implemented in `src/matching.py`. Results and figures: `weeks/week4/results/task2/`
(`task2_results.md` has the tables for the slides).

## Method (default of `src/local_pipeline.py`)

1. Background removal (up to `LocalMethod.max_paintings`, three by default), paintings in reading order
   (rows top to bottom, left to right within rows; the original two-painting rule is retained), each painting cropped
   to its mask's bounding box and filtered only if the noise detector flags the crop.
2. DoG keypoints (at most 1000 strongest, image resized to 900 px longest side), only inside the painting mask;
   SIFT descriptors compared as RootSIFT (Hellinger distance).
3. Tentative matches: Lowe ratio 0.75 and cross-check (mutual best match).
4. Verification: RANSAC homography (5 px); score of a museum image = inliers counted once per distinct museum
   keypoint (stops degenerate many-to-one homographies, see `degenerate_00005_qsd2w3.png`).
5. Ranking: museum images by decreasing score, top 10 per painting.
6. Additional known paintings in merged masks can be recovered by matching the unexplained keypoints,
   up to the query's painting limit. Separately segmented unknown regions retain their slots. The final regions
   are kept with their scores and reused by the global fallback instead of segmenting again.

The comparison numbers below and in `weeks/week4/results/task2/task2_results.md` describe the earlier local-only
study, before support for three paintings and the final global fallback. Rerun the studies for updated comparisons.

Known-painting mAP@1: QSD1-W4 0.828 (24/29), QSD2-W3 0.949 (37/39). With cross-check on, the RANSAC verification and
the distinct-inlier count do not change mAP on these sets (cross-check already makes the matches one-to-one);
they are kept because the score is also the evidence for the unknown decision (a geometric consensus, not a raw
match count), at about 2 s more per query. Not adopted: L1 / L2 distances, ORB,
BRISK, and the homography plausibility check (it zeroes correct matches because museum images are larger
than the query crops). That check lives only in `weeks/week4/experiments/task2_check.py`; both experimental variants
share the same initially segmented regions, features and RANSAC results, without matching-based recovery.

## API for Task 2 (II), unknown paintings

```python
from pathlib import Path
from src.local_pipeline import LocalMethod, run

out = run(Path("data/qsd1_w4"), Path("data/BBDD"), LocalMethod())   # default method above
out["rankings"]    # {query id: [[10 museum ids], [10 museum ids]]}  one list per painting, reading order
out["scores"]      # {query id: [{museum id: score}, {museum id: score}]}  score of EVERY museum image per painting
out["seconds"]     # {query id: seconds}
```

- A score is an `int`: distinct RANSAC inliers (0 when fewer than 4 tentative matches or RANSAC fails). The
  best score of a painting is `max(scores[q][i].values())`; `src.matching.rank_scores(s, k)` gives the top k ids.
- Query ids come from the file names (`00012.jpg` -> 12); the painting lists follow the order of the GT.
- For one image: `rank_query(image, matching_database(bbdd_dir, LocalMethod()), LocalMethod())` returns
  `(rankings, scores)` for its paintings (`matching_database` loads the cached museum features).
- Museum features are cached in `.cache/bbdd_local/local-<hash>.npz`. The key includes the corpus files,
  extraction settings, relevant source and NumPy/OpenCV versions. The global fallback descriptors have their
  own corpus-aware cache under `.cache/global_descriptors/`.
- `run(..., unknowns=True)` also returns `out["answers"]`: one Top-10 ranking or `[-1]` per painting.
  `main.py evaluate` and `main.py predict` enables this. The final rule uses threshold 10 and a global colour/texture fallback
  (best/second distance ratio below 0.85). The threshold-only study in `task2_unknowns.py` is a separate diagnostic.
- `out["paintings"]` contains the ordered native-image regions used for both local matching and the fallback.
- `out["decisions"]` contains the final ranking, reason and global ratio when `unknowns=True`.
- `process_query(image, database, method, global_database=global_db, debug=True)` also returns `matches` with
  the actual input features, crop and `MatchResult`. Recovery and visualization reuse the RANSAC mask used
  for scoring. The inspection notebook calls this public API without patching production functions.
- Recovery uses `LocalMethod.recovery_min_inliers`; rejection uses `unknown_threshold`. Changing the latter
  does not alter the recovered region count. `global_ratio` is the separate global-confidence setting.
- Query-score study caches include the complete method, query and corpus file signatures and source fingerprint.
- Threshold-search helpers and the experimental local-score ratio rule are defined in
  `weeks/week4/experiments/task2_unknowns.py`. Production rejection uses the fixed local threshold and global fallback.

## Rerun (from the repository root, `data/` with BBDD, qsd1_w4, qsd2_w3)

```bash
python -m pytest tests/test_matching.py                       # matching sanity test (needs data/BBDD)
python weeks/week4/experiments/task2_sanity.py --set qsd1_w4           # one configuration, failures, time per query
python weeks/week4/experiments/task2_metrics.py --set qsd1_w4          # 90-configuration grid (about 15 min, 8 processes)
python weeks/week4/experiments/task2_metrics.py --set qsd2_w3
python weeks/week4/experiments/task2_check.py                          # homography plausibility check, with / without
python weeks/week4/experiments/task2_failures.py                       # QSD2-W3 misses, degenerate-homography figure
python weeks/week4/experiments/task2_tables.py                         # final tables (CSV + task2_results.md)
```
