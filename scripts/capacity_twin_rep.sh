#!/bin/bash
# Варианты twin@w для запуска K (поправка 18.11.2026 к docs/CAPACITY_PROTOCOL.md, раздел 12): bash scripts/capacity_twin_rep.sh K
# Запуск 0: M1, M2, M3 в окнах B, A, E; запуски 1–5: M1, M2 в B, A, E, M3 в B, E (как в проверке шума, раздел 11).
set -e
K=$1
cd "$(dirname "$0")/.."
PY=venv/bin/python
OUT=outputs/capacity
$PY scripts/capacity_make_twin.py --theta B --rep "$K"
$PY scripts/capacity_make_twin.py --theta A --rep "$K"
if [ "$K" = "0" ]; then M3W="B A E"; else M3W="B E"; fi
$PY scripts/capacity_m12.py --twin --windows B A E --rep "$K" > "$OUT/twin_m12_r$K.log" 2>&1 &
$PY scripts/capacity_m3.py --twin --windows $M3W --jobs 3 --rep "$K" > "$OUT/twin_m3_r$K.log" 2>&1
wait
echo "двойник, запуск $K завершён"
