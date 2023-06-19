#!/bin/bash
#SBATCH -p kshdnormal
#SBATCH -N 8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=dcu:4
#SBATCH -J step2
#SBATCH -o ./log/chatglm-%j.out
#SBATCH -e ./log/chatglm-%j.out

source /public/home/yuguo960516yuguo/torch/env3.7_2.sh
rm -rf ./hostfile/*

echo "START TIME: $(date)"
hostfile=./hostfile/$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)
np=$(($np*4))

echo $np
nodename=$(cat $hostfile |sed -n "1p")
dist_url=`echo $nodename | awk '{print $1}'`

mpirun -np $np --allow-run-as-root --hostfile hostfile/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/single-step2.sh
