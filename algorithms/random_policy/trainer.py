#!/usr/bin/env python

from ray.rllib.agents.trainer_template import build_trainer
# from .policy import RandomPolicy
from ray.rllib.agents import ppo

DEFAULT_CONFIG = (
    {}
)  # Default config parameters that can be overriden by experiments YAML.

RandomPolicyTrainer = build_trainer(
    name="RandomPolicyTrainer",
    default_policy=ppo.policy,
    default_config=DEFAULT_CONFIG,
)
