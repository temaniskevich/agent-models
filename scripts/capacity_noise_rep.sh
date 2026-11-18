#!/bin/bash
# Один повтор проверки шума (поправка 19.11.2026 к docs/CAPACITY_PROTOCOL.md): bash scripts/capacity_noise_rep.sh K
set -e
K=$1
cd "$(dirname "$0")/.."
PY=venv/bin/python
OUT=outputs/capacity
$PY scripts/capacity_make_panels.py --theta B --rep "$K" --jobs 3
$PY scripts/capacity_make_panels.py --theta A --rep "$K" --jobs 3
$PY scripts/capacity_m12.py --windows B A E --rep "$K" > "$OUT/m12_r$K.log" 2>&1 &
$PY scripts/capacity_m3.py --windows B E --jobs 6 --rep "$K" > "$OUT/m3_r$K.log" 2>&1
wait
echo "повтор $K завершён"
