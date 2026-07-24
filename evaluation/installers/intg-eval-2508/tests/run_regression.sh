#!/bin/bash
#
# Golden (snapshot) regression tests for the job-script generators.
#
# Each case in cases.txt runs `python3 ../scripts/<script> <args> --dry-run`
# under a fixed dummy environment and compares stdout+stderr+exit code against
# the recorded snapshot in golden/. Any change to the generated job scripts
# (or to argument validation) shows up as a diff, so unintended changes are
# caught before deployment; intended changes are reviewed via the golden diff.
#
# Usage:
#   bash run_regression.sh           # compare against golden/
#   bash run_regression.sh --update  # (re)generate the golden files
#
# No GPU, cluster or network access is required (--dry-run only renders the
# job script and exits before touching the filesystem).

set -u

SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
cd "$SCRIPT_DIR"

# Deterministic dummy environment. These values are embedded into the golden
# files; never export real secrets here.
export HF_HOME=/groups/gcg51557/experiments/dummy/.cache/huggingface
export HF_TOKEN=dummy-hf-token
export OPENAI_API_KEY=dummy-openai-key
export AZURE_OPENAI_ENDPOINT=https://dummy.openai.azure.com/
export AZURE_OPENAI_API_KEY=dummy-azure-key
export OPENAI_API_VERSION=2025-04-01-preview
unset OPENAI_BASE_URL AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION \
    INTG_EVAL_EXPERIMENT_DIR 2> /dev/null || true

UPDATE=false
if [ "${1:-}" = "--update" ]; then UPDATE=true; fi

mkdir -p golden

# Strip run-dependent noise so snapshots stay stable:
# - logging timestamps ("2026-07-24 12:00:00,123 - WARNING - ...")
normalize_output() {
    sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+ - /TIMESTAMP - /'
}

# For failing cases, additionally drop traceback locations (file/line/code
# context); only the final "SomeError: message" line is meaningful, and line
# numbers would churn on every unrelated edit.
normalize_traceback() {
    sed -E -e '/^Traceback \(most recent call last\):$/d' \
        -e '/^  File "/d' \
        -e '/^    /d'
}

fail=0
total=0
seen=""
while read -r name script args; do
    case $name in ''|'#'*) continue ;; esac
    case " $seen " in *" $name "*) echo "DUPLICATE CASE NAME: $name"; fail=1; continue ;; esac
    seen="$seen $name"
    total=$((total + 1))

    # shellcheck disable=SC2086  # word splitting of $args is intended
    out=$(python3 "../scripts/${script}" $args --dry-run 2>&1)
    status=$?
    out=$(printf '%s\n' "$out" | normalize_output)
    if [ $status -ne 0 ]; then
        out=$(printf '%s\n' "$out" | normalize_traceback)
    fi
    out="${out}
exit_code: ${status}"

    golden_file="golden/${name}.txt"
    if [ "$UPDATE" = true ]; then
        printf '%s\n' "$out" > "$golden_file"
        echo "updated: $name"
    elif [ ! -f "$golden_file" ]; then
        echo "MISSING GOLDEN: $name (record it with 'bash run_regression.sh --update')"
        fail=1
    elif ! diff -u "$golden_file" <(printf '%s\n' "$out"); then
        echo "REGRESSION: $name"
        fail=1
    fi
done < cases.txt

if [ "$UPDATE" = true ]; then
    # Detect golden files whose case was removed or renamed.
    for f in golden/*.txt; do
        name=$(basename "$f" .txt)
        case " $seen " in *" $name "*) ;; *) echo "STALE GOLDEN (no matching case): $f" ;; esac
    done
    echo "Golden files regenerated (${total} cases). Review the diff before committing."
    exit 0
fi

if [ $fail -ne 0 ]; then
    echo "FAILED (${total} cases)"
    exit 1
fi
echo "OK (${total} cases)"
