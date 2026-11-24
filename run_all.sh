#!/usr/bin/env bash
# Reproduce every result and figure end to end (Git Bash / Linux / macOS).
# Total wall-clock on a laptop (RTX 5060 + 16-core CPU): about 25-35 min.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  uv venv .venv --python 3.12
  uv pip install --python .venv -r requirements.txt
fi
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
export PYTHONUNBUFFERED=1
mkdir -p logs results figures

rm -f results/runlog.csv
$PY scripts/00_sanity_reproductions.py | tee logs/00.log      # Ch10 'abc' primer + Ch7 SIR NIO reproductions
$PY scripts/01_baselines_malmem.py     | tee logs/01.log      # Part A1 (+ detector scores for Part C)
$PY scripts/03_primer_malmem.py        | tee logs/03.log      # Part B1 NIO primer + leakage audit
$PY scripts/05_primer_defense.py       | tee logs/05.log      # Part B3 defences
$PY scripts/06_policy_malware.py       | tee logs/06.log      # Part C1 detector-in-the-loop policy NIO
$PY scripts/08_make_figures.py         | tee logs/08.log      # all figures from results/
echo "done: see RESULTS.md, results/, figures/"
