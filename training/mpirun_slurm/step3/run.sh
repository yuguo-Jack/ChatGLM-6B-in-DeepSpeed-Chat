#/bin/bash

mkdir -p log 
#rm -rf log/*
mkdir -p step3_output
mkdir -p hostfile

sbatch run-step3.sh
