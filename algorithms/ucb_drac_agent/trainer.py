import logging

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.trainer import Trainer

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
})
# __sphinx_doc_end__
# yapf: enable

def update_kl(trainer, fetches):
    # Single-agent.
    if "kl" in fetches:
        trainer.workers.local_worker().for_policy(
            lambda pi: pi.update_kl(fetches["kl"]))

    # Multi-agent.
    else:

        def update(pi, pi_id):
            if pi_id in fetches:
                pi.update_kl(fetches[pi_id]["kl"])
            else:
                logger.debug("No data for {}, not updating kl".format(pi_id))

        trainer.workers.local_worker().foreach_trainable_policy(update)


def warn_about_bad_reward_scales(trainer, result):
    if result["policy_reward_mean"]:
        return  # Punt on handling multiagent case.

    # Warn about excessively high VF loss.
    learner_stats = result["info"]["learner"]
    if "default_policy" in learner_stats:
        scaled_vf_loss = (trainer.config["vf_loss_coeff"] *
                          learner_stats["default_policy"]["vf_loss"])
        policy_loss = learner_stats["default_policy"]["policy_loss"]
        if trainer.config["vf_share_layers"] and scaled_vf_loss > 100:
            logger.warning(
                "The magnitude of your value function loss is extremely large "
                "({}) compared to the policy loss ({}). This can prevent the "
                "policy from learning. Consider scaling down the VF loss by "
                "reducing vf_loss_coeff, or disabling vf_share_layers.".format(
                    scaled_vf_loss, policy_loss))

    # Warn about bad clipping configs
    if trainer.config["vf_clip_param"] <= 0:
        rew_scale = float("inf")
    else:
        rew_scale = round(
            abs(result["episode_reward_mean"]) /
            trainer.config["vf_clip_param"], 0)
    if rew_scale > 200:
        logger.warning(
            "The magnitude of your environment rewards are more than "
            "{}x the scale of `vf_clip_param`. ".format(rew_scale) +
            "This means that it will take more than "
            "{} iterations for your value ".format(rew_scale) +
            "function to converge. If this is not intended, consider "
            "increasing `vf_clip_param`.")


def validate_config(config):
    if config["entropy_coeff"] < 0:
        raise DeprecationWarning("entropy_coeff must be >= 0")
    if isinstance(config["entropy_coeff"], int):
        config["entropy_coeff"] = float(config["entropy_coeff"])
    if config["sgd_minibatch_size"] > config["train_batch_size"]:
        raise ValueError(
            "Minibatch size {} must be <= train batch size {}.".format(
                config["sgd_minibatch_size"], config["train_batch_size"]))
    if config["batch_mode"] == "truncate_episodes" and not config["use_gae"]:
        raise ValueError(
            "Episode truncation is not supported without a value "
            "function. Consider setting batch_mode=complete_episodes.")
    if config["multiagent"]["policies"] and not config["simple_optimizer"]:
        logger.info(
            "In multi-agent mode, policies will be optimized sequentially "
            "by the multi-GPU optimizer. Consider setting "
            "simple_optimizer=True if this doesn't work for you.")
    if config["simple_optimizer"]:
        logger.warning(
            "Using the simple minibatch optimizer. This will significantly "
            "reduce performance, consider simple_optimizer=False.")
    elif config["use_pytorch"] or (tf and tf.executing_eagerly()):
        config["simple_optimizer"] = True  # multi-gpu not supported


def get_policy_class(config):
    if config["use_pytorch"]:
        return PPOTorchPolicy
    else:
        from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
        return PPOTFPolicy


UcbDracTrainer = build_trainer(
    name="UcbDracAgent",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTorchPolicy,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales)
