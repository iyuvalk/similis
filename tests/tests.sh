#!/usr/bin/env bash

mkdir -p /tmp/similis 2> /dev/null
process_no=0
while read -r domain_info ; do
  domain_info=$(echo "${domain_info}" | tr -d '\r')
  ./test-worker.sh "${domain_info}" &
  process_no=$(( process_no + 1 ))
  if (( process_no % 10 == 0 )) && (( process_no > 1000 )); then
    # echo "DEBUG: Processes launched is ${process_no}. Going to sleep for 5 seconds..."
    sleep 5
  fi
done < test_set.lst
