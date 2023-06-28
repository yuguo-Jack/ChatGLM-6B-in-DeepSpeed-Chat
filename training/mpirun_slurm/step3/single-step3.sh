#!/bin/bash

source /public/home/yuguo960516yuguo/torch/env3.7_2.sh

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$OMPI_COMM_WORLD_SIZE

export HIP_VISIBLE_DEVICES=0,1,2,3

OUTPUT=./step3_output
ACTOR_ZERO_STAGE=3
CRITIC_ZERO_STAGE=3
Actor_Lr=1e-6
Critic_Lr=1e-6

# RM_STATIC=/public/home/hepj/data/DS-C/Dahoas/rm-static
# FULL=/public/home/hepj/data/DS-C/Dahoas/full-hh-rlhf
# SYNTHETIC=/public/home/hepj/data/DS-C/Dahoas/synthetic-instruct-gptj-pairwise
# RLHF=/public/home/hepj/data/DS-C/yitingxie/rlhf-reward-datasets

# OPT=/public/home/hepj/model_source/DS-C/opt-1.3b
# LLAMA=/public/home/hepj/torch/Linly-main/llama-hf
CHATGLM=/public/home/yuguo960516yuguo/chatglm-6b
ACTOR_MODEL_PATH=/public/home/yuguo960516yuguo/torch/ChatGLM-6B-in-DeepSpeed-Chat/training/step1_supervised_finetuning/step1_output
CRITIC_MODEL_PATH=/public/home/yuguo960516yuguo/torch/ChatGLM-6B-in-DeepSpeed-Chat/training/step2_reward_model_finetuning/step2_output

APP="python main.py \
    --data_path Dahoas/rm-static \
    --data_split 2,4,4 \
    --data_output_path /public/home/yuguo960516yuguo/torch/data_tmp_files  \
    --actor_model_name_or_path $ACTOR_MODEL_PATH \
    --critic_model_name_or_path $CRITIC_MODEL_PATH \
    --num_padding_at_beginning 0 \
    --per_device_train_batch_size 1 \
    --per_device_mini_train_batch_size 1 \
    --generation_batch_numbers 1 \
    --ppo_epochs 1 \
    --max_answer_seq_len 512 \
    --max_prompt_seq_len 512 \
    --actor_learning_rate ${Actor_Lr} \
    --critic_learning_rate ${Critic_Lr} \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --actor_gradient_checkpointing \
    --critic_gradient_checkpointing \
    --disable_actor_dropout \
    --num_warmup_steps 100 \
    --deepspeed --seed 1234 \
    --offload \
    --offload_reference_model \
    --actor_zero_stage $ACTOR_ZERO_STAGE \
    --critic_zero_stage $CRITIC_ZERO_STAGE \
    --enable_ema \
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
