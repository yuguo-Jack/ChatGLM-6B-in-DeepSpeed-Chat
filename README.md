# ChatGLM-6B in DeepSpeed-Chat for DCU

在DCU上利用DeepSpeed-Chat的强化学习方案进行ChatGLM-6B全参微调。

## ChatGLM-6B

ChatGLM-6B 是清华大学开源的开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

## DeepSpeed-Chat

一个完整的端到端三阶段 OpenAI InstructGPT 训练策略，带有强化学习人类反馈（RLHF），从用户最喜欢的预训练大型语言模型检查点生成高质量的 ChatGPT 风格模型。

## 数据集

[Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)

## 环境配置

单节点需要8张Z100L。

推荐使用docker方式运行，提供[光源](https://www.sourcefind.cn/#/service-details)torch的docker镜像：image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.10.0-centos7.6-dtk-22.10.1-py37-latest

进入docker:

```plaintext
cd /opt/dtk/.hip
source replace_origin.sh
```

然后需要卸载torch1.10，安装dtk22.10.1对应的 Deepspeed0.8.2与torch1.13，可从开发者社区[AI生态包](https://developer.hpccube.com/tool/)下载安装。

[模型目录](https://huggingface.co/THUDM/chatglm-6b)，需要修改config.json中auto_map：

"AutoModel": "modeling_chatglm.ChatGLMModel"

"AutoModelForCausalLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

## step1

阶段1可以采取任意一个支持ChatGLM-6B全参微调的项目进行，但是使用的数据集尽量和step2和step3保证分布一致（[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)）

### 训练

该微调脚本运行环境为1节点，8张DCU-Z100L-32G

微调训练命令：

```plaintext
# Move into the first step of the pipeline
cd training/step1_supervised_finetuning/

# Run the training script
bash training_scripts/single_node/run_chatglm-6b.sh
```

## step2

### 训练

该微调脚本运行环境为1节点，8张DCU-Z100L-32G

微调训练命令：

```plaintext
# Move into the second step of the pipeline
cd training/step2_reward_model_finetuning/

# Run the training script
bash training_scripts/single_node/run_chatglm-6b.sh
```

## step3

当前环境可以承载的负载有限，阶段3需要加载step1、2的输出模型，所以打开尽可能多的显存内存优化策略，参考step3的main.py，如果在超算上运行可以适当放宽限制提高性能。

### 训练

该微调脚本运行环境为1节点，8张DCU-Z100L-32G

微调训练命令：

```plaintext
# Move into the third step of the pipeline
cd training/step3_rlhf_finetuning/

# Run the training script
bash training_scripts/single_node/run_chatglm-6b.sh actor_model_path critic_model_path
```

## 参考

[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

[THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/tree/main)

[DeepSpeed-Chat-ChatGLM](https://github.com/yangzhipeng1108/DeepSpeed-Chat-ChatGLM)

[DeepSpeed-Chat源码详解](https://blog.csdn.net/remixa/category_12325075.html)

## 备注

可能模型路径跟用户环境有所不同，脚本中需要注意。
