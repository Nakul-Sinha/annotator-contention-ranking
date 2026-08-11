#!/usr/bin/env bash
# run.sh <tag> <script.py> <extra args...>
cd /home/nakul/sv || exit 1
mkdir -p logs out
TAG="$1"; SCRIPT="$2"; shift 2
setsid nohup python3 "code/$SCRIPT" "$@" > "logs/$TAG.log" 2>&1 < /dev/null &
echo "STARTED $TAG ($SCRIPT) pid=$!"
