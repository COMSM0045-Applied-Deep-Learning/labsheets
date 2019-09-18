#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f $0)")"
set -ex

for f in "$HERE"/**/*.ipynb; do
    "$HERE/render-notebook.sh" "$f" &
done
wait
