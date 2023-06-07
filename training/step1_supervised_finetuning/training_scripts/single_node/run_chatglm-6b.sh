#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/data/deepspeed-chat/chatglm-6b-actor
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
printf "Hello, Shell"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

HIP_VISIBLE_DEVICES=4,5,6,7 deepspeed --num_gpus=4 --master_port $MASTER_PORT main.py \
   --data_path Dahoas/rm-static \
   --model_name_or_path /zhaoy/chatglm-6b-moel \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --offload \
   --deepspeed \
   --output_dir $OUTPUT 
   #&> $OUTPUT/training.log
