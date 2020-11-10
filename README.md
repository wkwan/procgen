### Original Competition Starter Code: https://github.com/AIcrowd/neurips2020-procgen-starter-kit

# Overview

For this competition, I focused on making large changes to the learning algorithm to improve generalization and sample efficiency on the ProcGen benchmark, instead of tweaking the CNN architecture or tuning hyperparameters. I thought this was the fastest way for me to get better at reinforcement learning, even though it meant ignoring the simple and small optimizations that could've increase my score.

My first approach was to add data augmentation on top of the RLLib PPO implementation, since the competition restricts the agent to 8M frames of training on each game. Looking at the results from this paper, [Automatic Data Augmentation for Generalization in Deep Reinforcement Learning](https://arxiv.org/pdf/2006.12862.pdf), it seems that different ProcGen games work best with different data augmentations (although crop usually works best), so I followed the authors' approach and implemented the Upper Confidence Bounds algorithm to automatically choose 1 of 8 data augmentations (crop, grayscale, random convolution, color jitter, cutout color, cutout, rotate, and flip) during training. I also changed the loss function according to the authors' approach. Intuitively, we want to 

Code can be found in the [ucbfinal branch](https://github.com/wkwan/procgen-competition/tree/ucbfinal).


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

Uncomment these environment variables in [run.sh](run.sh) and set your saved model and the number of episodes you want to rollout:

```
  # export CHECKPOINT=
  # export EPISODES=5
```

To rollout the agent:

```
./run.sh --rollout
```


