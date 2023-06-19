# ChatGLM-6B in DeepSpeed-Chat for DCU

在DCU上利用DeepSpeed-Chat的强化学习方案进行ChatGLM-6B全参微调。

## ChatGLM-6B

ChatGLM-6B 是清华大学开源的开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。

## DeepSpeed-Chat

一个完整的端到端三阶段 OpenAI InstructGPT 训练策略，带有强化学习人类反馈（RLHF），从用户最喜欢的预训练大型语言模型检查点生成高质量的 ChatGPT 风格模型。

## 数据集

[Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) 或自定义数据集

## 环境配置

单节点需要8张Z100L。

推荐使用docker方式运行，提供[光源](https://www.sourcefind.cn/#/service-details)torch的docker镜像：image.sourcefind.cn:5000/dcu/admin/base/pytorch:1.10.0-centos7.6-dtk-22.10.1-py37-latest。（如果使用超算集群环境，建议使用Python虚拟环境）

进入docker:

```plaintext
cd /opt/dtk/.hip
source replace_origin.sh
```

然后需要卸载torch1.10，安装dtk22.10.1对应的 Deepspeed0.9.2与torch1.13.1，可从开发者社区[AI生态包](https://developer.hpccube.com/tool/)下载安装。（dtk-23.04已经发布，可以dtk也替换成23.04版本）

[模型目录](https://huggingface.co/THUDM/chatglm-6b)，需要修改config.json中auto_map：

"AutoModel": "modeling_chatglm.ChatGLMModel"

"AutoModelForCausalLM": "modeling_chatglm.ChatGLMForConditionalGeneration"

## step1

阶段1可以采取任意一个支持ChatGLM-6B全参微调的项目进行，但是使用的数据集尽量和step2和step3保证分布一致（[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)）。

### 训练

该微调脚本运行环境为1节点，8张DCU-Z100L-32G。

如果在超算集群上测试，可以参考 training/mpirun_slurm 中相关脚本，需要建个Python虚拟环境venv_torch3.7，将step1下的脚本放置到 training/step1_supervised_finetuning 下，执行 run.sh 即可。

以下只演示如何在本地节点启动。

微调训练命令：

```plaintext
# Move into the first step of the pipeline
cd training/step1_supervised_finetuning/

# Run the training script
bash training_scripts/single_node/run_chatglm-6b.sh
```

### 收敛性

在超算集群上采用**自定义数据集**训练**10 epoch loss 可以从4左右降至0.02左右，ppl从72降至1.01**，推理输出正常，学的比较过拟合。

## step2

### 训练

该微调脚本运行环境为1节点，8张DCU-Z100L-32G。

如果在超算集群上测试，可以参考 training/mpirun_slurm 中相关脚本，需要建个Python虚拟环境venv_torch3.7，将step2下的脚本放置到 training/step2_reward_model_finetuning 下，执行 run.sh 即可。

以下只演示如何在本地节点启动。

微调训练命令：

```plaintext
# Move into the second step of the pipeline
cd training/step2_reward_model_finetuning/

# Run the training script
bash training_scripts/single_node/run_chatglm-6b.sh
```

### 收敛性

在超算集群上采用**自定义数据集**只训练**1 epoch，loss 从1.7左右降至0.25左右**

chosen scores, rejected scores, acc如下所示：

before train：

```
chosen_last_scores (higher is better) : 1.8644356727600098, reject_last_scores (lower is better) : 1.9873332977294922, acc (higher is better) : 0.33432674408
```

after train：

```
chosen_last_scores (higher is better) : 12.223048210144043, reject_last_scores (lower is better) : 4.411005973815918, acc (higher is better) : 0.98
```

可以看出 scores diff 增大，acc 增大。由于训练数据不多，评估时 acc 较高。

## step3

当前环境可以承载的负载有限，阶段3需要加载step1、2的输出模型，所以打开尽可能多的显存内存优化策略，参考step3的main.py，如果在超算上运行可以适当放宽限制提高性能。

### 训练

该微调脚本运行环境为1节点，8张DCU-Z100L-32G。

如果在超算集群上测试，可以参考 training/mpirun_slurm 中相关脚本，需要建个Python虚拟环境venv_torch3.7，将step3下的脚本放置到 training/step3_rlhf_finetuning 下，执行 run.sh 即可。

以下只演示如何在本地节点启动。

微调训练命令：

```plaintext
# Move into the third step of the pipeline
cd training/step3_rlhf_finetuning/

# Run the training script
bash training_scripts/single_node/run_chatglm-6b.sh actor_model_path critic_model_path 
```

稍后更新在自定义数据集上的训练效果。

### 推理

```
bash inference.sh
```

## 参考

[DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

[THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B/tree/main)

[DeepSpeed-Chat-ChatGLM](https://github.com/yangzhipeng1108/DeepSpeed-Chat-ChatGLM)

[DeepSpeed-Chat源码详解](https://blog.csdn.net/remixa/category_12325075.html)

## 备注

可能模型路径跟用户环境有所不同，脚本中需要注意。
