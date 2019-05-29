#!/usr/bin/env bash


NUM_WORKERS=4
CONFIG_FILE="configs/BipedalWalkerHardcore.gin"

# parse command line args
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -c|--config)
      CONFIG_FILE=$2
      shift 2
      ;;
    -n|--num-workers)
      NUM_WORKERS=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS --$1"
      shift
      ;;
  esac
done

declare -A worker_ids

for ((i = 0; i < $NUM_WORKERS; i++))
do
    python es_worker.py --config=$CONFIG_FILE --worker-id=$i &
    worker_ids[$i]=$!
    echo "Worker $i started."
done

python es_master.py --config=$CONFIG_FILE --num-workers=$NUM_WORKERS

echo "Terminate all workers ..."
for wid in "${worker_ids[@]}"
do
    kill $wid
done
echo "Done"
