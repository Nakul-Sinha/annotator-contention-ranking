#!/usr/bin/env bash
# Config-variant sweep. Epoch count set from the epoch-response probe.
cd /home/nakul/sv || exit 1
mkdir -p logs out
EP=${EP:-40}
export OMP_NUM_THREADS=4 NTHREADS=4
laneA() {
  python3 code/expcv.py --tag v_size160 --epochs $EP --size 160 --seed 3 >> logs/lane_a.log 2>&1
  python3 code/expcv.py --tag v_r34     --epochs $EP --backbone resnet34 --lr 2e-3 --seed 4 >> logs/lane_a.log 2>&1
  python3 code/expcv.py --tag v_lrtm10  --epochs $EP --lrtm 0.10 --lr 3e-3 --seed 7 >> logs/lane_a.log 2>&1
}
laneB() {
  python3 code/expcv.py --tag v_effb0   --epochs $EP --backbone efficientnet_b0 --lr 2e-3 --drop 0.35 --seed 5 >> logs/lane_b.log 2>&1
  python3 code/expcv.py --tag v_drop45  --epochs $EP --drop 0.45 --w w_rank=1.4 --seed 6 >> logs/lane_b.log 2>&1
  python3 code/expcv.py --tag v_noaux18 --epochs $EP --noaux --bins 18 --seed 8 >> logs/lane_b.log 2>&1
}
laneA & laneB &
wait
python3 code/handmodel.py > logs/hand2.log 2>&1
echo BATCH2_DONE
