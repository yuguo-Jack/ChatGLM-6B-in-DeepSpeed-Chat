#!/bin/bash

source /public/home/yuguo960516yuguo/torch/env3.7_2.sh

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

export HIP_VISIBLE_DEVICES=0,1,2,3

OUTPUT=./step1_output
ZERO_STAGE=3

# RM_STATIC=/public/home/hepj/data/DS-C/Dahoas/rm-static
# FULL=/public/home/hepj/data/DS-C/Dahoas/full-hh-rlhf
# SYNTHETIC=/public/home/hepj/data/DS-C/Dahoas/synthetic-instruct-gptj-pairwise
# RLHF=/public/home/hepj/data/DS-C/yitingxie/rlhf-reward-datasets

# OPT=/public/home/hepj/model_source/DS-C/opt-1.3b
# LLAMA=/public/home/hepj/torch/Linly-main/llama-hf
CHATGLM=/public/home/yuguo960516yuguo/chatglm-6b

APP="python main.py \
    --data_path Dahoas/rm-static \
    --data_split 2,4,4 \
    --data_output_path /public/home/yuguo960516yuguo/torch/data_tmp_files
    --model_name_or_path $CHATGLM \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --max_seq_len 1024 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage $ZERO_STAGE \
    --gradient_checkpointing \
    --deepspeed \
    --local_rank $lrank \
    --output_dir $OUTPUT "

echo ${APP}

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_2:1
  export UCX_IB_PCI_BW=mlx5_2:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=0,1,2,3
  export UCX_NET_DEVICES=mlx5_3:1
  export UCX_IB_PCI_BW=mlx5_3:50Gbs
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac
