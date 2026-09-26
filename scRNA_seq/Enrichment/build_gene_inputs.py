#!/usr/bin/env python3
"""Build the three single-cell gene lists used for Table 8 enrichment.

The selection reproduces paper/ML/XAI/GeniCutoff.ipynb:
1. keep the first 1,000 positive MeanAbsoluteSHAP features;
2. by default, do not reinsert genes from declustering clusters;
3. optionally reinsert every gene in a cluster whose representative is selected
   with --include-cluster-genes for the comparison condition.
Only small feature-importance and cluster pickles are read. No .h5ad or raw
expression matrix is loaded.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib


DATASETS = (
    {
        "name": "CD4_CSF",
        "store_dir": "CD4_CSF_SAMU",
        "importance": "xgbCSFBCELLS_feature_importance.pkl",
    },
    {
        "name": "BCELLS_CSF",
        "store_dir": "CSF_BCELLS",
        "importance": "xgbCSFBCELLS2_feature_importance.pkl",
    },
    {
        "name": "BCELLS_PBMC",
        "store_dir": "PBMC_BCELLS_2DATASET",
        "importance": "xgbPBMCBCELLS_feature_importance.pkl",
    },
)


def build_gene_list(
    importance_path: Path,
    clusters_path: Path,
    cutoff: int,
    include_cluster_genes: bool,
) -> tuple[list[str], int, int]:
    importance = joblib.load(importance_path)
    positive = importance.loc[importance["MeanAbsoluteShap"] > 0, "Feature"].astype(str).tolist()
    selected = positive[:cutoff]
    if not include_cluster_genes:
        return selected, len(positive), len(selected)

    selected_set = set(selected)
    clusters = joblib.load(clusters_path)
    additions: set[str] = set()
    for cluster_genes, representative in clusters.items():
        if str(representative) in selected_set:
            additions.update(str(gene) for gene in cluster_genes)

    genes = selected + sorted(additions.difference(selected_set))
    return genes, len(positive), len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Store",
        help="Directory containing the Store/<dataset> artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs",
        help="Directory for one-gene-per-line input files and manifest.csv.",
    )
    parser.add_argument("--cutoff", type=int, default=1000)
    parser.add_argument(
        "--include-cluster-genes",
        action="store_true",
        help="Reinsert declustering cluster members for the comparison condition.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for config in DATASETS:
        store_dir = args.store_root / config["store_dir"]
        importance_path = store_dir / config["importance"]
        clusters_path = store_dir / "uniqueClusters.pkl"
        genes, positive_count, cutoff_count = build_gene_list(
            importance_path,
            clusters_path,
            args.cutoff,
            args.include_cluster_genes,
        )

        output_path = args.output_dir / f"{config['name']}.txt"
        output_path.write_text("\n".join(genes) + "\n", encoding="utf-8")
        rows.append(
            {
                "dataset": config["name"],
                "input_genes": len(genes),
                "positive_shap_features": positive_count,
                "shap_cutoff_features": cutoff_count,
                "importance_file": str(importance_path),
                "clusters_file": str(clusters_path),
                "selection_rule": (
                    f"top {cutoff_count} positive SHAP features plus their declustering clusters"
                    if args.include_cluster_genes
                    else f"top {cutoff_count} positive SHAP features without declustering reinsertion"
                ),
                "gene_file": str(output_path),
            }
        )
        print(f"{config['name']}: {len(genes)} genes")

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
