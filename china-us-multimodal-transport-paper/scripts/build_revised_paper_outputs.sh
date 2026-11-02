#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python -u scripts/build_revised_paper_analysis.py
python -u paper/render_figures.py

echo "Revised manuscript tables and figures completed."
echo "Tables: analysis/paper_revised"
echo "Figures: paper/figures"
