# Multi-seed Strategy A runner

`runner.py` runs one independent `skopt.BayesSearchCV` for one retained dataset
and one canonical patient split. It reads
`agent_log/multi_seed_splits/patient_assignments.csv` and only loads the matching
already-declustered input:

- `CD4_CSF` → `paper/ML/Store/CD4_CSF_SAMU/datasetDeclustered.h5ad`
- `CSF_BCELLS` → `paper/ML/Store/CSF_BCELLS/datasetDeclustered.h5ad`
- `PBMC_BCELLS` → `paper/ML/Store/PBMC_BCELLS_2DATASET/datasetDeclustered.h5ad`

## CLI

Dry-run (validation only; no sampler, scaler, XGBoost, or search fitting):

```bash
micromamba run -n ms-rebuttal python paper/ML/multiseed/runner.py \
  --dataset CD4_CSF --split-seed 101 --search-iterations 2 \
  --output-dir /path/to/results/cd4_seed101 --dry-run
```

A real short or full run omits `--dry-run`. Strategy A's notebook-sized
searches use `--search-iterations 400` for `CD4_CSF` and `CSF_BCELLS`, and
`--search-iterations 500` for `PBMC_BCELLS`; a smaller positive value is useful
for a smoke run. `--search-iterations` is required and must be positive for a
real run.

A real run creates the supplied output directory only when it is absent (or
empty), and refuses to overwrite a non-empty directory. Results are written
atomically as `summary.json`, `metrics.csv`, `best_parameters.json`,
`bayes_search.joblib`, and `run.log`.

## Validation and data boundaries

The manifest is authoritative. The runner checks that each selected seed has
exactly train/validation/test assignments, all h5ad patients are covered once,
there is no patient overlap, every partition contains both `CTRL` and `MS`,
manifest cell counts match the h5ad, and each patient has one consistent
condition in both sources. Rows in the matrices are cells, while partition
membership is patient-level, matching the notebooks. The test matrix is not
passed to `BayesSearchCV`; it is evaluated only after the search's `refit="f1"`
estimator has been selected.

## Notebook compatibility notes

- All three notebooks define the same scoring dictionary: accuracy, precision
  and recall for classes 0/1, and macro-F1 under the key `f1`; Bayesian refit
  uses `refit="f1"` and a `PredefinedSplit` with train folds `-1` and validation
  fold `0`.
- `CD4_CSF` follows the visible narrow CD4/Samuele BayesSearchCV branch:
  `n_estimators` 50–600, learning rate 0.001–0.7, gamma 0.0001–1, and
  regularization 0.001–100 (all log-uniform where shown), with fixed
  `scale_pos_weight=5263/8000`. That branch searches the original `x_train`
  without resampling. The notebook also contains an earlier, conflicting
  RandomUnderSampler branch; it is not silently combined with the selected
  search branch.
- `CSF_BCELLS` follows the visible `MajorUpperbound` search: broad
  `n_estimators` 50–3000, learning rate 1e-7–1, gamma 1e-6–1,
  `scale_pos_weight` in `{1, 400/510}`, and regularization 1e-6–100. Its
  training matrix uses `SMOTETomek(random_state=42)`; validation is untouched.
- `PBMC_BCELLS` follows its broad search: `n_estimators` 50–3000, learning
  rate 1e-7–1, gamma 1e-6–1, `scale_pos_weight` in `{1, 439/1077, 4}`, and
  regularization 1e-6–100. Its training matrix uses
  `SMOTETomek(random_state=42)`; validation is untouched.
- The notebooks hard-code different patient lists and random seeds. Those are
  intentionally replaced by the supplied seed-101/202 manifest. Notebook
  preprocessing uses `MinMaxScaler` followed by XGBoost; the runner preserves
  that pipeline and explicitly sets XGBoost `device="cuda"` and
  `tree_method="hist"`.
- The manifest `split_seed` controls only patient assignment. To make the two
  searches differ only because of their split (and preserve notebook behavior),
  sampler, XGBoost, and Bayesian-search RNGs are fixed at `42`; this is recorded
  in the dry-run plan and `summary.json`.
