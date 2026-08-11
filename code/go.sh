#!/usr/bin/env bash
# go.sh <tag> <extra args...>   -- launch exp1 detached, log to logs/<tag>.log
cd /home/nakul/sv || exit 1
mkdir -p logs out
TAG="$1"; shift
setsid nohup python3 code/exp1.py --tag "$TAG" "$@" > "logs/$TAG.log" 2>&1 < /dev/null &
echo "STARTED $TAG pid=$!"
