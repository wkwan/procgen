import logging

import ray
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from .custom_postprocessing import Postprocessing, compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from .custom_torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from .custom_torch_policy_template import build_torch_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.torch_ops import sequence_mask
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

def setup_config(policy, obs_space, action_space, config):
    # auto set the model option for layer sharing
    config["model"]["vf_share_layers"] = config["vf_share_layers"]


def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append([sample_batch["state_out_{}".format(i)][-1]])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    # print("post process ppo gae", sample_batch[SampleBatch.REWARDS].shape)

    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch


class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:
            num_valid = torch.sum(valid_mask)

            def reduce_mean_valid(t):
                return torch.sum(t * valid_mask) / num_valid

        else:

            def reduce_mean_valid(t):
                return torch.mean(t)

        # prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = torch.exp(
            curr_action_dist.logp(actions) - prev_actions_logp)
        # action_kl = prev_dist.kl(curr_action_dist)
        # self.mean_kl = 0.5*reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        # advantages = (advantages - reduce_mean_valid(advantages)) / (torch.std(advantages) + 1e-8)
        surrogate_loss = torch.max(
            -advantages * logp_ratio,
            -advantages * torch.clamp(logp_ratio, 1 - clip_param,
                                     1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(surrogate_loss)

        if use_gae:
            # vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
            # vf_clipped = vf_preds + torch.clamp(value_fn - vf_preds,
            #                                     -vf_clip_param, vf_clip_param)
            # vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
            # vf_loss = torch.max(vf_loss1, vf_loss2)
            # self.mean_vf_loss = reduce_mean_valid(vf_loss)
            self.mean_vf_loss = vf_loss_coeff * reduce_mean_valid(torch.pow(vf_preds - value_targets, 2.0))

            # loss = reduce_mean_valid(
            #     -surrogate_loss + cur_kl_coeff * action_kl +
            #     vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = 0.0
            # loss = reduce_mean_valid(-surrogate_loss +
            #                          cur_kl_coeff * action_kl -
            #                          entropy_coeff * curr_entropy)

        # self.loss = torch.stack((self.mean_policy_loss, self.mean_vf_loss))
        print("ent coef", entropy_coeff)
        print("vf coef", vf_loss_coeff)
        print("negent", -entropy_coeff * self.mean_entropy)
        print("mean pl", self.mean_policy_loss)
        print("mean vl", self.mean_vf_loss)
        self.loss = torch.stack((-entropy_coeff * self.mean_entropy + self.mean_policy_loss, self.mean_vf_loss))
        # print("mean policy loss and vf loss", self.loss)

        # self.loss = loss


def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)
    # print("action dist in surrogate loss fn", action_dist)
    # train_batch['oldpd'] = action_dist
    # print("state", len(state), state)
    # print("the old pd we just put in", train_batch['oldpd'])

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def kl_and_loss_stats(policy, train_batch):
    return {
        # "cur_kl_coeff": policy.kl_coeff,
        # "cur_lr": policy.cur_lr,
        # "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        # "vf_explained_var": explained_variance(
        #     train_batch[Postprocessing.VALUE_TARGETS],
        #     policy.model.value_function(),
        #     framework="torch"),
        # "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        # "entropy_coeff": policy.entropy_coeff,
    }


def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient.
        self.kl_coeff = config["kl_coeff"]
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        return self.kl_coeff


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: self._convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: self._convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: self._convert_to_tensor(
                        [prev_reward]),
                    "is_training": False,
                }, [self._convert_to_tensor(s) for s in state],
                                          self._convert_to_tensor([1]))
                # print("value mixin", self.model.value_function()[0])
                return self.model.value_function()[0]

        else:

            def value(ob, prev_action, prev_reward, *state):
                return 0.0

        self._value = value


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def choose_optimizer(policy, config):
    return torch.optim.Adam(policy.model.parameters(), lr=config["lr"], eps=1e-5)

PPOTorchPolicy = build_torch_policy(
    name="PPOTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    optimizer_fn=choose_optimizer,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])