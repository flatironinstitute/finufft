#!/usr/bin/env bash
# Run sigma analysis: build training binary, generate CSV, plot fits.
# Usage (from repo root): bash devel/run_sigma_analysis.sh [build_dir] [prec] [type] [dim]
# Defaults: build_dir=build, prec=d, type=1, dim=1

set -e

BUILD=${1:-build}
PREC=${2:-d}
TYPE=${3:-1}
DIM=${4:-1}
CSV=/tmp/sigma_${PREC}_t${TYPE}_d${DIM}.csv
FIT=/tmp/sigma_${PREC}_t${TYPE}_d${DIM}_fit.txt
PLOT=/tmp/sigma_${PREC}_t${TYPE}_d${DIM}.png

echo "=== Building find_sigma_bound ==="
cmake --build "$BUILD" -j --target find_sigma_bound

echo "=== Running training (prec=$PREC type=$TYPE dim=$DIM) ==="
"$BUILD"/devel/find_sigma_bound --prec "$PREC" --type "$TYPE" --dim "$DIM" --ntol 300 \
	>"$CSV" 2>"$FIT"

echo "--- Fit results ---"
cat "$FIT"
echo "--- CSV: $CSV ($(wc -l <"$CSV") rows) ---"

echo "=== Plotting ==="
python3 devel/analyse_sigma.py "$CSV" "$PLOT"
echo "Plot saved to $PLOT"
