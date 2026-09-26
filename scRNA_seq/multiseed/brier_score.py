#!/usr/bin/env python3
"""Recompute test metrics (including the Brier score) for the reported runs.

Two run families are covered:

- ``reference``: the original notebook run of a retained single-cell dataset.
  The patient split is regenerated exactly as the notebook does
  (``RandomState(notebook_seed).shuffle(obs["sample"].unique())`` then a
  60/20/20 floor split), and the classifier of the notebook's final test cell
  is refitted on train+validation before scoring the test set.
- ``manifest``: one Strategy A run (``results/strategy_a/<dataset>/seed_<seed>``).
  The saved ``bayes_search.joblib`` is reloaded and its refitted
  ``best_estimator_`` is scored on the canonical test split of
  ``agent_log/multi_seed_splits/patient_assignments.csv``.

Both families score the same test quantities, so the numbers can be compared
across runs:

- accuracy, precision, recall, F1 (positive class = MS), macro F1, ROC-AUC
- Brier score of the MS probability, ``mean((p - y) ** 2)``

Predictions are dumped as ``.npz`` next to the metrics so the numbers can be
re-derived without reloading a multi-GB ``.h5ad``, and each reference run is
checked against the confusion matrix printed by the notebook.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

# The classifier of each notebook's final test cell.  ``params`` are the
# hard-coded hyperparameters of that cell; CD4_CSF instead reuses the
# ``best_estimator_`` of the search it saved, exactly like the notebook.
REFERENCE_RUNS: dict[str, dict[str, Any]] = {
    "CSF_BCELLS": {
        "h5ad": ROOT / "paper/ML/Store/CSF_BCELLS/datasetDeclustered.h5ad",
        "notebook_seed": 59,
        "source": "paper/ML/CSF BCELLS .ipynb cell 22",
        "params": {
            "gamma": 1e-06,
            "learning_rate": 0.050020916043633644,
            "max_depth": 15,
            "min_child_weight": 3,
            "n_estimators": 3000,
            "reg_alpha": 0.00010871970763546638,
            "reg_lambda": 100.0,
            "scale_pos_weight": 0.7843137254901961,
        },
        # Confusion matrix printed by the notebook for this test set.
        "notebook_confusion": [[215, 13], [11, 119]],
    },
    "CD4_CSF": {
        "h5ad": ROOT / "paper/ML/Store/CD4_CSF_SAMU/CD4TNONAIVE_csf_datasetDeclustered.h5ad",
        "notebook_seed": 53,
        # The CD4 CSF notebook does not shuffle; it hard-codes these lists
        # (cell 14).  The other two notebooks use the 60/20/20 shuffle.
        "patients": {
            "train": ["19270", "49131", "58637", "74594", "YYW_CSF", "KJS_CSF", "41540", "45044", "83775"],
            "validation": ["71658", "JYJ_CSF", "85037", "95809"],
            "test": ["60249", "32190"],
        },
        # The notebook refits the search's best_estimator_ on train+validation.
        # The pickled search cannot be reused here (its pipelines were written by
        # scikit-learn 1.5 and its scorers live in the notebook's __main__), so
        # the parameters printed by that notebook (cell 21 output) are used.
        "source": "paper/ML/CD4 CSF SAMU.ipynb cell 23 (search best_params_ refit)",
        # The notebook's stored output corresponds to a fit on the
        # RandomUnderSampler-resampled training set (cell 15), not on the raw
        # one: refitting raw train+val gives accuracy 0.719, the printed value
        # is 0.7156.
        "resample": {"sampling_strategy": 5263 / 8000, "random_state": 42},
        "params": {
            "gamma": 0.0001,
            "learning_rate": 0.044980648663355424,
            "max_depth": 15,
            "min_child_weight": 8,
            "n_estimators": 600,
            "reg_alpha": 0.001,
            "reg_lambda": 0.001,
            "scale_pos_weight": 0.657875,
        },
        "notebook_confusion": [[519, 944], [362, 2767]],
    },
    "PBMC_BCELLS": {
        "h5ad": ROOT / "paper/ML/Store/PBMC_BCELLS_2DATASET/datasetDeclustered.h5ad",
        "notebook_seed": 46,
        "source": "paper/ML/PBMC BCELLS 2DATASET.ipynb cell 23",
        "params": {
            "gamma": 0.00010424606377892094,
            "learning_rate": 0.08395998719982567,
            "max_depth": 3,
            "min_child_weight": 8,
            "n_estimators": 1095,
            "reg_alpha": 100.0,
            "reg_lambda": 100.0,
            "scale_pos_weight": 1,
        },
        "notebook_confusion": [[298, 249], [69, 807]],
    },
    "CD4_PBMC": {
        # The CD4 PBMC notebook reads the 17 483-gene / 15-patient store; the
        # sibling PBMC_CD4TNONAIVE_2DATASET store has 14 patients and can not
        # reproduce the notebook's printed split (9/3/3 patients).
        "h5ad": ROOT / "paper/ML/Store/PBMC_CD4THANDPICK/datasetDeclustered.h5ad",
        "notebook_seed": 47,
        "source": "paper/ML/CD4NONAIVE MODEL.ipynb cell 22 (no stored output)",
        "params": {
            "gamma": 1e-06,
            "learning_rate": 4.222975850408766,
            "max_depth": 2,
            "min_child_weight": 1,
            "n_estimators": 1357,
            "reg_alpha": 1e-06,
            "reg_lambda": 1e-06,
            "scale_pos_weight": 0.41007194244604317,
        },
        "notebook_confusion": None,
    },
}


def _runner() -> Any:
    """Import the sibling multiseed runner so split handling stays in one place."""
    spec = importlib.util.spec_from_file_location("multiseed_runner", Path(__file__).with_name("runner.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(y: np.ndarray, probability: np.ndarray, split: str) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = (probability > 0.5).astype(int)
    return {
        "split": split,
        "n_obs": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def _confusion(y: np.ndarray, probability: np.ndarray) -> list[list[int]]:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y, (probability > 0.5).astype(int)).tolist()


def _install_runner_scorers() -> None:
    """Expose the runner's earlier private scorers so old ``joblib`` files load.

    ``results/strategy_a/CD4_CSF/seed_202/bayes_search.joblib`` predates the
    switch to sklearn-native scorers and references ``_precision_class_*`` /
    ``_recall_class_*`` from ``__main__``.  They only appear in the stored
    ``cv_results_``; the estimator that is scored is unaffected.
    """
    import __main__

    from sklearn.metrics import precision_score, recall_score

    def _precision_class_0(y_true: Any, y_pred: Any) -> Any:
        return precision_score(y_true, y_pred, average=None)[0]

    def _precision_class_1(y_true: Any, y_pred: Any) -> Any:
        return precision_score(y_true, y_pred, average=None)[1]

    def _recall_class_0(y_true: Any, y_pred: Any) -> Any:
        return recall_score(y_true, y_pred, average=None)[0]

    def _recall_class_1(y_true: Any, y_pred: Any) -> Any:
        return recall_score(y_true, y_pred, average=None)[1]

    for name, scorer in (
        ("_precision_class_0", _precision_class_0),
        ("_precision_class_1", _precision_class_1),
        ("_recall_class_0", _recall_class_0),
        ("_recall_class_1", _recall_class_1),
    ):
        setattr(__main__, name, scorer)


def _probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x))
    if proba.shape[1] != 2:
        raise ValueError(f"Expected binary probabilities, got shape {proba.shape}")
    return proba[:, 1]


def _classifier(params: dict[str, Any]) -> Any:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBClassifier

    settings: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
        "tree_method": "hist",
        "device": "cuda",
    }
    settings.update(params)
    try:
        classifier = XGBClassifier(use_label_encoder=False, **settings)
    except TypeError:  # xgboost >= 2 removed the deprecated flag
        classifier = XGBClassifier(**settings)
    return Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])


def _notebook_split(adata: ad.AnnData, seed: int, proportions: tuple[float, float]) -> dict[str, list[str]]:
    import math

    patients = adata.obs["sample"].unique().tolist()
    np.random.RandomState(seed).shuffle(patients)
    n_train = math.floor(len(patients) * proportions[0])
    n_validation = math.floor(len(patients) * proportions[1])
    return {
        "train": patients[:n_train],
        "validation": patients[n_train : n_train + n_validation],
        "test": patients[n_train + n_validation :],
    }


def run_reference(dataset: str, output: Path) -> dict[str, Any]:
    module = _runner()
    recipe = REFERENCE_RUNS[dataset]
    adata = ad.read_h5ad(recipe["h5ad"])
    patients = recipe.get("patients") or _notebook_split(adata, recipe["notebook_seed"], (0.6, 0.2))
    matrices = {split: module._dense(module._matrix_for(adata, ids)) for split, ids in patients.items()}
    labels = {
        split: (adata.obs.loc[adata.obs["sample"].astype(str).isin(ids), "condition"] == "MS").astype(int).to_numpy()
        for split, ids in patients.items()
    }
    model = _classifier(recipe["params"])
    fitted_params = recipe["params"]
    if recipe.get("resample"):
        from imblearn.under_sampling import RandomUnderSampler

        matrices["train"], labels["train"] = RandomUnderSampler(**recipe["resample"]).fit_resample(
            matrices["train"], labels["train"]
        )
    x_full = np.concatenate([matrices["train"], matrices["validation"]], axis=0)
    y_full = np.concatenate([labels["train"], labels["validation"]], axis=0)
    model.fit(x_full, y_full)

    metrics = []
    predictions: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        probability = _probabilities(model, matrices[split])
        predictions[f"y_{split}"] = labels[split]
        predictions[f"p_{split}"] = probability
        metrics.append(_score(labels[split], probability, split))
    recorded = recipe["notebook_confusion"]
    observed = _confusion(labels["test"], predictions["p_test"])

    result = {
        "run": f"reference/{dataset}",
        "dataset": dataset,
        "run_type": "reference",
        "source": recipe["source"],
        "notebook_seed": recipe["notebook_seed"],
        "h5ad": str(recipe["h5ad"]),
        "partitions": {split: {"patients": ids, "cells": int(len(labels[split]))} for split, ids in patients.items()},
        "fitted_params": {str(k): (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in fitted_params.items()},
        "test_confusion": observed,
        "notebook_confusion": recorded,
        "matches_notebook_confusion": None if recorded is None else observed == recorded,
        "metrics": metrics,
    }
    _write(output, result, predictions)
    return result


def run_manifest(dataset: str, seed: int, output: Path) -> dict[str, Any]:
    import joblib

    module = _runner()
    adata = ad.read_h5ad(module.DATASET_PATHS[dataset])
    manifest = module._manifest_frame(module.MANIFEST_PATH)
    plan = module._validate_and_plan(manifest, dataset, seed, adata)
    partitions = plan["partitions"]
    matrices = {split: module._dense(module._matrix_for(adata, partitions[split]["patients"])) for split in ("train", "test")}
    labels = {
        split: (adata.obs.loc[adata.obs["sample"].astype(str).isin(partitions[split]["patients"]), "condition"] == "MS")
        .astype(int)
        .to_numpy()
        for split in ("train", "test")
    }
    _install_runner_scorers()
    search = joblib.load(ROOT / f"results/strategy_a/{dataset}/seed_{seed}/bayes_search.joblib")
    model = search.best_estimator_

    metrics = []
    predictions: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        probability = _probabilities(model, matrices[split])
        predictions[f"y_{split}"] = labels[split]
        predictions[f"p_{split}"] = probability
        metrics.append(_score(labels[split], probability, split))
    recorded = _recorded_test_metrics(dataset, seed)
    observed = {key: value for key, value in metrics[-1].items() if key in recorded}
    diffs = {key: abs(recorded[key] - observed[key]) for key in observed}
    result = {
        "run": f"manifest/{dataset}/seed_{seed}",
        "dataset": dataset,
        "run_type": "manifest",
        "source": f"results/strategy_a/{dataset}/seed_{seed}/bayes_search.joblib",
        "split_seed": seed,
        "fitted_params": {str(k): (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in search.best_params_.items()},
        "recorded_test_metrics": recorded,
        "max_metric_difference": max(diffs.values()),
        "matches_recorded_metrics": max(diffs.values()) <= 1e-9,
        "metrics": metrics,
    }
    _write(output, result, predictions)
    return result


def _recorded_test_metrics(dataset: str, seed: int) -> dict[str, float]:
    """Read the test row the Strategy A runner already wrote for this run."""
    path = ROOT / f"results/strategy_a/{dataset}/seed_{seed}/metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                return {key: float(value) for key, value in row.items() if key not in {"split", "n_obs"}}
    raise ValueError(f"No test row in {path}")


def _write(output: Path, result: dict[str, Any], predictions: dict[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    name = result["run"].replace("/", "_")
    np.savez(output / f"{name}_predictions.npz", **predictions)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "partitions"}, indent=2, sort_keys=True))


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    test = next(metric for metric in result["metrics"] if metric["split"] == "test")
    return {
        "run": result["run"],
        "dataset": result["dataset"],
        "run_type": result["run_type"],
        "seed": result.get("split_seed", result.get("notebook_seed")),
        **{key: test[key] for key in ("n_obs", "accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "brier")},
    }


METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc", "brier")


def _group_summary(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    """Mean and sample standard deviation of the test metrics per dataset."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    summary = []
    for dataset, items in sorted(grouped.items()):
        entry: dict[str, Any] = {
            "dataset": dataset,
            "n_runs": len(items),
            "runs": "; ".join(str(item["run"]) for item in items),
        }
        for key in METRIC_KEYS:
            values = np.array([float(item[key]) for item in items])
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.append(entry)
    with (output / "aggregated_with_brier.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "manifest", "summary"), required=True)
    parser.add_argument("--dataset", choices=sorted(REFERENCE_RUNS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/brier")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()

    if args.mode == "summary":
        rows = [_csv_row(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(output.glob("*.json"))]
        with (output / "test_metrics_with_brier.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = _group_summary(rows, output)
        header = "| dataset | n | acc | precision | recall | F1 | macro-F1 | AUC | Brier |"
        print(header)
        print("|---|---|---|---|---|---|---|---|---|")
        for entry in summary:
            cells = " | ".join(
                f"{entry[f'{key}_mean']:.3f} ± {entry[f'{key}_sd']:.3f}" for key in METRIC_KEYS
            )
            print(f"| {entry['dataset']} | {entry['n_runs']} | {cells} |")
        return 0

    if not args.dataset:
        parser.error("--dataset is required")
    if args.mode == "reference":
        run_reference(args.dataset, output)
    else:
        if args.seed is None:
            parser.error("--seed is required for manifest runs")
        run_manifest(args.dataset, args.seed, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
