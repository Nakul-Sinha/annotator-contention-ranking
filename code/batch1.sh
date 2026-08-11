#!/usr/bin/env bash
cd /home/nakul/sv || exit 1
mkdir -p logs out
export OMP_NUM_THREADS=4 NTHREADS=4
laneA() {
  python3 code/expcv.py --tag base   --epochs 30 >> logs/lane_a.log 2>&1
  python3 code/expcv.py --tag ep60   --epochs 60 >> logs/lane_a.log 2>&1
  python3 code/expcv.py --tag drop50 --epochs 30 --drop 0.50 >> logs/lane_a.log 2>&1
}
laneB() {
  python3 code/expcv.py --tag size160 --epochs 30 --size 160 >> logs/lane_b.log 2>&1
  python3 code/expcv.py --tag noaux   --epochs 30 --noaux >> logs/lane_b.log 2>&1
  python3 code/expcv.py --tag scratch --epochs 60 --nopre --lr 4e-3 >> logs/lane_b.log 2>&1
}
laneA & laneB &
wait
echo BATCH1_DONE
