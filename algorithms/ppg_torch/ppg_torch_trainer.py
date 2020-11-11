import logging

from .ppg_torch_policy import PPGTorchPolicy, DEFAULT_CONFIG

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, \
    StandardizeFields, SelectExperiences
from .custom_train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.utils import try_import_tf

tf = try_import_tf()

logger = logging.getLogger(__name__)

def warn_about_bad_reward_scales(config, result):
    # if result["policy_reward_mean"]:
    #     return result  # Punt on handling multiagent case.

    # # Warn about excessively high VF loss.
    # learner_stats = result["info"]["learner"]
    # if "default_policy" in learner_stats:
    #     scaled_vf_loss = (config["vf_loss_coeff"] *
    #                       learner_stats["default_policy"]["vf_loss"])
    #     policy_loss = learner_stats["default_policy"]["policy_loss"]
    #     if config["vf_share_layers"] and scaled_vf_loss > 100:
    #         logger.warning(
    #             "The magnitude of your value function loss is extremely large "
    #             "({}) compared to the policy loss ({}). This can prevent the "
    #             "policy from learning. Consider scaling down the VF loss by "
    #             "reducing vf_loss_coeff, or disabling vf_share_layers.".format(
    #                 scaled_vf_loss, policy_loss))

    # # Warn about bad clipping configs
    # if config["vf_clip_param"] <= 0:
    #     rew_scale = float("inf")
    # else:
    #     rew_scale = round(
    #         abs(result["episode_reward_mean"]) / config["vf_clip_param"], 0)
    # if rew_scale > 200:
    #     logger.warning(
    #         "The magnitude of your environment rewards are more than "
    #         "{}x the scale of `vf_clip_param`. ".format(rew_scale) +
    #         "This means that it will take more than "
    #         "{} iterations for your value ".format(rew_scale) +
    #         "function to converge. If this is not intended, consider "
    #         "increasing `vf_clip_param`.")

    return result


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
    # Multi-gpu not supported for PyTorch and tf-eager.
    elif config["framework"] in ["tfe", "torch"]:
        config["simple_optimizer"] = True


def get_policy_class(config):
    return PPGTorchPolicy

class UpdateKL:
    """Callback to update the KL based on optimization info."""

    def __init__(self, workers):
        self.workers = workers

    def __call__(self, fetches):
        pass
        # def update(pi, pi_id):
        #     assert "kl" not in fetches, (
        #         "kl should be nested under policy id key", fetches)
        #     if pi_id in fetches:
        #         assert "kl" in fetches[pi_id], (fetches, pi_id)
        #         pi.update_kl(fetches[pi_id]["kl"])
        #     else:
        #         logger.warning("No data for {}, not updating kl".format(pi_id))

        # self.workers.local_worker().foreach_trainable_policy(update)


def execution_plan(workers, config):
    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # Collect large batches of relevant experiences & standardize.
    rollouts = rollouts.for_each(
        SelectExperiences(workers.trainable_policies()))
    rollouts = rollouts.combine(
        ConcatBatches(min_batch_size=config["train_batch_size"]))
    rollouts = rollouts.for_each(StandardizeFields(["advantages"]))

    if config["simple_optimizer"]:
        train_op = rollouts.for_each(
            TrainOneStep(
                workers,
                num_sgd_iter=config["num_sgd_iter"],
                sgd_minibatch_size=config["sgd_minibatch_size"]))
    else:
        train_op = rollouts.for_each(
            TrainTFMultiGPU(
                workers,
                sgd_minibatch_size=config["sgd_minibatch_size"],
                num_sgd_iter=config["num_sgd_iter"],
                num_gpus=config["num_gpus"],
                rollout_fragment_length=config["rollout_fragment_length"],
                num_envs_per_worker=config["num_envs_per_worker"],
                train_batch_size=config["train_batch_size"],
                shuffle_sequences=config["shuffle_sequences"],
                _fake_gpus=config["_fake_gpus"]))

    # Update KL after each round of training.
    train_op = train_op.for_each(lambda t: t[1]).for_each(UpdateKL(workers))

    return StandardMetricsReporting(train_op, workers, config) \
        .for_each(lambda result: warn_about_bad_reward_scales(config, result))


PPGTrainer = build_trainer(
    name="PPG",
    default_config=DEFAULT_CONFIG,
    default_policy=PPGTorchPolicy,
    get_policy_class=get_policy_class,
    execution_plan=execution_plan,
    validate_config=validate_config)