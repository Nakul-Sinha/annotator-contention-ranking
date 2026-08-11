#!/usr/bin/env bash
cd /home/nakul/sv || exit 1
export OMP_NUM_THREADS=3 NTHREADS=3
python3 code/handmodel.py
python3 code/leak_audit.py
echo AUDIT_DONE
