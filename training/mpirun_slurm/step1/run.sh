#/bin/bash

mkdir -p log 
#rm -rf log/*
mkdir -p step1_output
mkdir -p hostfile

sbatch run-step1.sh
