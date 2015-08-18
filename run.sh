for task in 1 2 3 4 5 6 9 10 11 12 13 14 15 16 17 18 20; do
  echo $task
  for run in `jot 10`; do
    python main.py --task $task > results/${task}_${run}.txt &
  done
  wait
done
