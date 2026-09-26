#!/usr/bin/env python3
"""Compute notebook ROC-AUC from predicted probabilities instead of labels.

``roc_auc_score(y, hard_predictions)`` is not an AUC: with a two-valued score it
returns the trapezoid through ``(0, 0) -> (FPR, TPR) -> (1, 1)``, i.e. the mean of
sensitivity and specificity.  This script rewrites the affected lines so that the
AUC used in the manuscript comes from the predicted MS-class probability.

Each edit is applied inside one specific cell of one specific notebook, and the
notebook JSON is never re-serialised wholesale: only the target line inside that
cell's ``source`` array is replaced, so the surrounding file stays byte-identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# (notebook, cell index, estimator variable, split) -> the offending call becomes
# roc_auc_score(<y>, <estimator>.predict_proba(<x>)[:, 1]).
EDITS: dict[str, list[tuple[int, str, str, str]]] = {
    "paper/ML/CSF BCELLS .ipynb": [
        (21, "y_val", "x_val", "pipeline"),
        (22, "y_test", "x_test", "pipeline"),
        (24, "y_test", "x_test", "ModelloEnorme"),
    ],
    "paper/ML/CD4 CSF SAMU.ipynb": [
        (22, "y_val", "x_val", "pipeline"),
        (23, "y_test", "x_test", "pipeline"),
    ],
    "paper/ML/PBMC BCELLS 2DATASET.ipynb": [
        (22, "y_val", "x_val", "pipeline"),
        (23, "y_test", "x_test", "pipeline"),
    ],
    "paper/ML/CD4NONAIVE MODEL.ipynb": [
        (21, "y_val", "x_val", "pipeline"),
        (22, "y_test", "x_test", "pipeline"),
    ],
}


def _source_block(raw: str, cell_index: int) -> tuple[int, int]:
    """Byte range of the ``source`` array of the n-th cell as serialised."""
    starts: list[int] = []
    cursor = 0
    while True:
        found = raw.find('"source": [', cursor)
        if found == -1:
            break
        starts.append(found)
        cursor = found + 1
    if cell_index >= len(starts):
        raise IndexError(f"notebook has {len(starts)} cells, requested {cell_index}")
    start = starts[cell_index]
    end = raw.find("\n   ]", start)
    if end == -1:
        raise ValueError("could not find the end of the source array")
    return start, end


def patch(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    notebook = json.loads(raw)
    changes: list[str] = []
    # Highest cell index first so earlier byte offsets stay valid.
    for cell_index, y, x, estimator in sorted(EDITS[str(path.relative_to(ROOT))], reverse=True):
        source = notebook["cells"][cell_index]["source"]
        old_line = next((line for line in source if "roc_auc_score(" in line and "predict_proba" not in line), None)
        if old_line is None:
            raise ValueError(f"{path}: cell {cell_index} has no label-based roc_auc_score call")
        new_line = f"roc_auc = roc_auc_score({y}, {estimator}.predict_proba({x})[:, 1])"
        start, end = _source_block(raw, cell_index)
        old_token = json.dumps(old_line if old_line.endswith("\n") else old_line + "\n")
        new_token = json.dumps(new_line + "\n")
        block = raw[start:end]
        if block.count(old_token) != 1:
            raise ValueError(f"{path}: cell {cell_index} does not contain {old_token!r} exactly once")
        raw = raw[:start] + block.replace(old_token, new_token) + raw[end:]
        changes.append(f"{path.name} cell {cell_index}: {old_line.strip()}  ->  {new_line}")
    path.write_text(raw, encoding="utf-8")
    # Re-read to confirm the notebook is still valid and the edits landed.
    updated = json.loads(path.read_text(encoding="utf-8"))
    for cell_index, y, x, estimator in EDITS[str(path.relative_to(ROOT))]:
        joined = "".join(updated["cells"][cell_index]["source"])
        expected = f"roc_auc_score({y}, {estimator}.predict_proba({x})[:, 1])"
        if expected not in joined:
            raise ValueError(f"{path}: edit did not land in cell {cell_index}")
        if f"roc_auc_score({y}, y_pred_" in joined:
            raise ValueError(f"{path}: label-based call still present in cell {cell_index}")
        compile(joined, f"{path.name}:cell{cell_index}", "exec")  # syntax check
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    args = parser.parse_args(argv)
    for relative in EDITS:
        path = ROOT / relative
        if args.apply:
            for change in patch(path):
                print(change)
        else:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell_index, y, x, estimator in EDITS[relative]:
                source = notebook["cells"][cell_index]["source"]
                line = next((each for each in source if "roc_auc_score(" in each and "predict_proba" not in each), None)
                print(f"{relative} cell {cell_index}: {line.strip() if line else 'NO LABEL-BASED CALL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
