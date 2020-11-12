### Original Competition Starter Code: https://github.com/AIcrowd/neurips2020-procgen-starter-kit

# Overview

For this competition, I focused on making large changes to the learning algorithm to improve generalization and sample efficiency on the ProcGen benchmark, instead of tweaking the CNN architecture or tuning hyperparameters. I thought this was the fastest way for me to get better at reinforcement learning, even though it meant ignoring the simple and small optimizations that could've increased my score.

For reference, here is how standard PPO performs in the BigFish environment. I'll use the BigFish environment for all subsequent graphs. For this PPO agent, the only changes I made from the starter code were switching Tensorflow to PyTorch and changing the epsilon hyperparameter in the Adam Optimizer (RLLib doesn't use the same value as OpenAI Baselines, and it performed better when I changed it to the OpenAI Baselines value).

![PPO](ppo-bigfish.png)

## Approach 1: Automatic Data Augmentation

**Code is in the [ucbfinal branch](https://github.com/wkwan/procgen-competition/tree/ucbfinal).**

My first idea was to add data augmentation on top of the RLLib PPO implementation, since the competition restricts the agent to 8M frames of training on each game. Looking at the results from this paper, [Automatic Data Augmentation for Generalization in Deep Reinforcement Learning](https://arxiv.org/pdf/2006.12862.pdf), it seems that different ProcGen games work best with different data augmentations (although crop usually works best), so I followed the authors' approach and implemented the Upper Confidence Bounds algorithm to automatically choose 1 of 8 data augmentations (crop, grayscale, random convolution, color jitter, cutout color, cutout, rotate, and flip) during training. I also changed the loss function according to the authors' approach. The idea is to add two regularization terms (one for the policy function, and the other for the value function), so that the policy and value functions produce similar results with and without the data augmentation.

My implementation initially performed worse than PPO:




However, the number of stochastic gradient descent iterations was reduced from 3 to 2 to allow for more training time on the augmented frames, and the training would still hit the 2 hour training time limit on the test servers before finishing 8M frames, so I think there would be minor improvements over default PPO if I improved efficiency and tweaked hyperparameters. But I wanted to focus on making bigger changes to the learning algorithm than simply adding data augmentation, so I scrapped this approach.

## Approach 2: Phasic Policy Gradient

**Code is in the default [ppg branch](https://github.com/wkwan/procgen-competition).**

Two weeks before the deadline, I found OpenAI's new [Phasic Policy Gradient](https://arxiv.org/pdf/2009.04416.pdf) paper for improving generalization and sample efficiency on the ProcGen games. Their models were trained on 100M timesteps for each game and looking at their results, most games only had minor improvements in the first 8M frames. However, I thought this was an important paper so I wanted to implement it myself and see if I could make further improvements to the sample efficiency.

The main problem I encountered were sudden drops in the mean reward during training time.

I found that reducing the number of auxiliary phases reduced these drops.

I then added more iterations of value loss training during the policy phase to make up for the reduced value function training by reducing the auxiliary phases.

This was as far as I got before the competition deadline. The performance was worse than default PPO, and I believe the key issue is to fix are the sudden performance drops during training.

# Setup

The Dockerfile is used for my official competition submission but isn't necessary.

I tested my code on an **AWS p3.2xlarge instance**, using this AMI:

**Deep Learning AMI (Ubuntu 18.04) Version 36.0 - ami-063585f0e06d22308**

First, activate this default Conda environment on the image (I actually use PyTorch, but that gets installed with pip):

```
source activate tensorflow2_latest_p37
```

Then install the Python packages:

```
pip install -r requirements.txt --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html
```

# Training

Change these environment variables in [run.sh](run.sh) depending on your available resources:

```
  export RAY_MEMORY_LIMIT=60129542144
  export RAY_CPUS=8
  export RAY_STORE_MEMORY=30000000000
```

To change the ProcGen game environment, change this line in [experiments/ppg.yaml](experiments/ppg.yaml):
```
env_name: coinrun
```

To start training:

```
./run.sh --train
```

Your model and checkpoints will be saved in ~/ray_results/procgen-ppg

# Rollout

Uncomment and modify these environment variables in [run.sh](run.sh):

```
  # export EPISODES=5
  # replace with your own checkpoint path
  # export CHECKPOINT=~/ray_results/procgen-ppg/PPG_procgen_env_wrapper_0_2020-11-10_18-16-19qlw86nzo/checkpoint_447/checkpoint-447
```

To rollout the agent:

```
./run.sh --rollout
```


