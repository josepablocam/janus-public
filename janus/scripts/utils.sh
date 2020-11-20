#!/usr/bin/env bash


function tsp-tasks-remaining() {
  tsp -l | awk '$2 == "queued" || $2 == "running"' | wc -l
}

function block-until-done() {
  if [ "$#" -ne 1 ]
  then
    echo "Usage: <cmd> <seconds-to-sleep>"
    return 1
  fi

  sleep_seconds=$1
  while [[ $(tsp-tasks-remaining) -ne 0 ]]
  do
    echo "tsp still has running/queued jobs, sleeping..."
    # use seconds so portable between mac and linux
    sleep ${sleep_seconds}
  done
}


function block-until-done-check-deadlocks() {
  if [ "$#" -lt 1 ]
  then
    echo "Usage: <cmd> <seconds-to-sleep> [deadlock-time]"
    return 1
  fi

  sleep_seconds=$1
  deadlock_time=0
  shift
  if [ "$#" -eq 1 ]
  then
    deadlock_time=$1
    shift
  fi


  while [[ $(tsp-tasks-remaining) -ne 0 ]]
  do
    echo "Checking tsp for possible deadlocks"
    if [ ${deadlock_time} -eq 0 ]
    then
      python scripts/utils.py --cmd monitor_tsp
    else
      python scripts/utils.py --cmd monitor_tsp --arg1 ${deadlock_time}
    fi
    echo "tsp still has running/queued jobs, sleeping..."
    # use seconds so portable between mac and linux
    sleep ${sleep_seconds}
  done
}
