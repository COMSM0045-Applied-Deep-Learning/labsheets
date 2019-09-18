#!/usr/bin/env bash
set -eux
HERE="$(dirname "$(readlink -f $0)")"

[[ $# -ne 1 ]] && { echo "USAGE: $0 notebook.ipynb"; exit 1; }
NOTEBOOK="$1"; shift

ARGS=""
if grep "$(basename "$NOTEBOOK")" "$HERE/.error-causing-notebooks"; then
    ARGS+="--allow-errors"
fi
jupyter nbconvert  \
    --ExecutePreprocessor.kernel_name=python3 \
    --execute \
    --inplace \
    $ARGS \
    "$NOTEBOOK"
