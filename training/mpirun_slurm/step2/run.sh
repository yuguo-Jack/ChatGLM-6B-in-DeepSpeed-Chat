#/bin/bash

mkdir -p log 
#rm -rf log/*
mkdir -p step2_output
mkdir -p hostfile

sbatch run-step2.sh
