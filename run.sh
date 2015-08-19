#!/bin/bash

tasks="1 2 3 4 5 6 9 10 11 12 13 14 15 16 17 18 20"
for task in $tasks; do
  for run in `jot 10`; do
    python -u main.py --task $task > results/${task}_${run}.txt &
  done
  wait
done

for task in $tasks; do
  best=`grep -H TRAIN_ERROR results/${task}_*.txt | sort -n -k2 | head -n 1 | awk -F ":" '{print $1}'`
  echo "TASK $task: `grep TEST_ERROR $best | tail -n 1`"
done

