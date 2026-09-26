#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk1/f.massafra"
MAMBA_ROOT_PREFIX="$ROOT/.mamba"
MICROMAMBA="$MAMBA_ROOT_PREFIX/bin/micromamba"
ENV_NAME="ms-rebuttal"
RUNNER="$ROOT/paper/ML/multiseed/runner.py"
RESULT_ROOT="${STRATEGY_A_RESULT_ROOT:-$ROOT/results/strategy_a}"
QUEUE_LOG="$RESULT_ROOT/queue.log"
STATUS="$RESULT_ROOT/queue_status.tsv"

mkdir -p "$RESULT_ROOT"
if [[ -e "$STATUS" ]]; then
    echo "Refusing to overwrite existing queue status: $STATUS" >&2
    exit 2
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1
printf 'QUEUE_START %s\n' "$(date --iso-8601=seconds)"
printf 'dataset\tseed\titerations\tstatus\tstarted\tended\toutput\n' > "$STATUS"

run_one() {
    local dataset="$1"
    local seed="$2"
    local iterations="$3"
    local output="$RESULT_ROOT/$dataset/seed_$seed"
    local started ended rc

    if [[ -e "$output" ]]; then
        echo "Refusing to reuse existing output directory: $output" >&2
        exit 2
    fi
    started="$(date --iso-8601=seconds)"
    printf '%s\t%s\t%s\trunning\t%s\t\t%s\n' "$dataset" "$seed" "$iterations" "$started" "$output" >> "$STATUS"
    echo "JOB_START dataset=$dataset seed=$seed iterations=$iterations output=$output"
    set +e
    "$MICROMAMBA" run -n "$ENV_NAME" python "$RUNNER" \
        --dataset "$dataset" \
        --split-seed "$seed" \
        --search-iterations "$iterations" \
        --output-dir "$output"
    rc=$?
    set -e
    ended="$(date --iso-8601=seconds)"
    if [[ "$rc" -eq 0 ]]; then
        printf '%s\t%s\t%s\tdone\t%s\t%s\t%s\n' "$dataset" "$seed" "$iterations" "$started" "$ended" "$output" >> "$STATUS"
        echo "JOB_DONE dataset=$dataset seed=$seed output=$output"
    else
        printf '%s\t%s\t%s\tfailed(rc=%s)\t%s\t%s\t%s\n' "$dataset" "$seed" "$iterations" "$rc" "$started" "$ended" "$output" >> "$STATUS"
        echo "JOB_FAILED dataset=$dataset seed=$seed rc=$rc output=$output" >&2
        exit "$rc"
    fi
}

run_one CD4_CSF 101 400
run_one CD4_CSF 202 400
run_one CSF_BCELLS 101 400
run_one CSF_BCELLS 202 400
run_one PBMC_BCELLS 101 500
run_one PBMC_BCELLS 202 500

printf 'QUEUE_DONE %s\n' "$(date --iso-8601=seconds)"
