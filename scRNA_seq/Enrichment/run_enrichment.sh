#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STORE_ROOT="${1:-$ROOT/../Store}"
OUTPUT_DIR="${2:-$ROOT/results}"
INPUT_DIR="$ROOT/inputs"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-micromamba}"
ENV_NAME="${ENV_NAME:-ms-enrichment}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
INCLUDE_CLUSTER_GENES="${INCLUDE_CLUSTER_GENES:-0}"
STRINGDB_CACHE_DIR="${STRINGDB_CACHE_DIR:-$OUTPUT_DIR/stringdb_cache}"

BUILD_ARGS=(
  --store-root "$STORE_ROOT"
  --output-dir "$INPUT_DIR"
)
if [[ "$INCLUDE_CLUSTER_GENES" == "1" ]]; then
  BUILD_ARGS+=(--include-cluster-genes)
fi

MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$PYTHON_BIN" "$ROOT/build_gene_inputs.py" \
  "${BUILD_ARGS[@]}"

MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MICROMAMBA_BIN" run -n "$ENV_NAME" Rscript \
  "$ROOT/run_stringdb_enrichment.R" \
  "$INPUT_DIR" \
  "$OUTPUT_DIR" \
  400 \
  "$STRINGDB_CACHE_DIR"
