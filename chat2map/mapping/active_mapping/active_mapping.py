# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from chat2map.common.loss_n_evalMetrics import compute_loss_n_evalMetrics
from chat2map.mapping.active_mapping.ddppo_utils import distributed_mean_and_var

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        config,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,

    ):
        """
        Creates object to train the chat2map active mapping policy with PPO and train chat2map mapper with supervised learning
        :param actor_critic: full chat2map active mapping model
        :param config: config
        :param use_clipped_value_loss: flag saying if value loss of PPO to be clipped or not
        :param use_normalized_advantage: flag saying if advantage to be normalized in generalized advantage estimation
                                        (GAE) (https://arxiv.org/abs/1506.02438) of PPO
        """
        super().__init__()

        self.actor_critic = actor_critic

        self.sim_cfg = config.TASK_CONFIG.SIMULATOR

        self.ppo_cfg = config.RL.PPO
        self.passive_mapping_cfg = config.PassiveMapping

        self.clip_param = self.ppo_cfg.clip_param
        self.ppo_epoch = self.ppo_cfg.ppo_epoch
        self.num_mini_batch = self.ppo_cfg.num_mini_batch
        self.value_loss_coef = self.ppo_cfg.value_loss_coef
        self.entropy_coef = self.ppo_cfg.entropy_coef
        self.lr_pol = self.ppo_cfg.lr
        self.max_grad_norm_pol = self.ppo_cfg.max_grad_norm
        self.eps_pol = self.ppo_cfg.eps
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_normalized_advantage = use_normalized_advantage

        self.lr_mapper = self.passive_mapping_cfg.lr
        self.betas_mapper = self.passive_mapping_cfg.betas
        self.eps_mapper = self.passive_mapping_cfg.eps
        self.wd_mapper = self.passive_mapping_cfg.weight_decay
        self.max_grad_norm_mapper = self.passive_mapping_cfg.max_grad_norm
        self.mapper_epoch = self.passive_mapping_cfg.num_epochs
        self.num_mini_batch_mapper = self.passive_mapping_cfg.num_mini_batch

        self.optimizer_pol = None
        if self.ppo_cfg.agent_type == "chat2map_activeMapper":
            pol_params = list(actor_critic.policy.parameters()) + list(actor_critic.action_dist.parameters()) +\
                         list(actor_critic.critic.parameters())
            self.optimizer_pol = optim.Adam(pol_params, lr=self.lr_pol, eps=self.eps_pol)

        mapper_params = list(actor_critic.mapper.parameters())

        self.optimizer_mapper = torch.optim.Adam(
            filter(lambda p: p.requires_grad, mapper_params),
            lr=self.lr_mapper,
            betas=tuple(self.betas_mapper),
            eps=self.eps_mapper,
            weight_decay=self.wd_mapper,
        )

        self.device = next(actor_critic.parameters()).device

    def get_advantages(self, rollouts_pol):
        """get advantages from policy rollout storage"""
        advantages = rollouts_pol.returns[:-1] - rollouts_pol.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update_pol(self, rollouts_pol):
        """update policy"""
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        advantages = self.get_advantages(rollouts_pol)

        for e in range(self.ppo_epoch):
            data_generator = rollouts_pol.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample


                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer_pol.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step_pol()
                self.optimizer_pol.step()
                self.after_step()

                action_loss_epoch += action_loss.item()
                value_loss_epoch += value_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        action_loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def update_mapper(self, rollouts_mapper):
        """update mapper"""
        mapper_loss_epoch = 0.

        for e in range(self.mapper_epoch):
            data_generator = rollouts_mapper.recurrent_generator(self.num_mini_batch_mapper)

            for sample in data_generator:
                (
                    obs_batch,
                ) = sample

                mapper_step_observation, query_views_gt, query_views_evalMasks = self.build_mapper_step_observation(
                    {
                        k: v for k, v in obs_batch.items()
                    }
                )

                preds =\
                    self.actor_critic.mapper_forward(
                        mapper_step_observation,
                    )

                mapper_loss = compute_loss_n_evalMetrics(
                    loss_or_metric_types=self.passive_mapping_cfg.TrainLosses.types,
                    loss_or_metric_weights=self.passive_mapping_cfg.TrainLosses.weights,
                    gt_occMap=query_views_gt.view(-1, *query_views_gt.size()[2:]),
                    pred_occMap=preds.view(-1, *preds.size()[2:]),
                    mask=mapper_step_observation["query_views_mask"].view(-1),
                )

                self.optimizer_mapper.zero_grad()

                self.before_backward(mapper_loss)
                self.before_step_mapper()

                mapper_loss.backward()
                self.after_backward(mapper_loss)

                self.optimizer_mapper.step()
                self.after_step()

                mapper_loss_epoch += mapper_loss.item()

        num_updates = self.mapper_epoch * self.num_mini_batch_mapper

        mapper_loss_epoch /= num_updates

        return mapper_loss_epoch

    def build_mapper_step_observation(self, initial_mapper_step_observation):
        """build observation for mapper from data storred in rollout storage"""
        mapper_step_observation = dict()
        for k, v in initial_mapper_step_observation.items():
            if k not in ["query_globCanMapEgoCrop_gt", "query_globCanMapEgoCrop_gt_exploredPartMask"]:
                if k in ["query_views_mask"]:
                    mapper_step_observation[k] = v.reshape((v.size(0), v.size(1) * v.size(2), *v.size()[3:]))
                else:
                    mapper_step_observation[k] = v

        mapper_step_observation["context_selfAudio_pose"] = mapper_step_observation["context_views_pose"]
        mapper_step_observation["context_otherAudio_mask"] = mapper_step_observation["context_selfAudio_mask"][:, [1, 0], :]

        mapper_step_observation["query_views_pose"] = mapper_step_observation["context_views_pose"].reshape(
            mapper_step_observation["context_views_pose"].size(0),
            mapper_step_observation["context_views_pose"].size(1) * mapper_step_observation["context_views_pose"].size(2),
            *mapper_step_observation["context_views_pose"].size()[3:]
        )

        query_maps_gt = initial_mapper_step_observation["query_globCanMapEgoCrop_gt"]
        query_maps_gt = query_maps_gt.reshape(
            query_maps_gt.size(0),
            query_maps_gt.size(1) * query_maps_gt.size(2),
            *query_maps_gt.size()[3:]
        )

        query_maps_exploredMasks = initial_mapper_step_observation["query_globCanMapEgoCrop_gt_exploredPartMask"]
        query_maps_exploredMasks = query_maps_exploredMasks.reshape(
            query_maps_exploredMasks.size(0),
            query_maps_exploredMasks.size(1) * query_maps_exploredMasks.size(2),
            *query_maps_exploredMasks.size()[3:]
        )

        return mapper_step_observation,\
               query_maps_gt,\
               query_maps_exploredMasks

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step_pol(self):
        pol_params = list(self.actor_critic.policy.parameters()) +\
                     list(self.actor_critic.action_dist.parameters()) +\
                     list(self.actor_critic.critic.parameters())
        nn.utils.clip_grad_norm_(
            pol_params, self.max_grad_norm_pol,
        )

    def before_step_mapper(self):
        if self.max_grad_norm_mapper is not None:
            nn.utils.clip_grad_norm_(
                self.actor_critic.mapper.parameters(), self.max_grad_norm_mapper,
            )

    def after_step(self):
        pass


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(
        self, rollouts_pol
    ) -> torch.Tensor:
        advantages = rollouts_pol.returns[:-1] - rollouts_pol.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        mean, var = distributed_mean_and_var(advantages)

        return (advantages - mean) / (var.sqrt() + EPS_PPO)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        """
        Initializes distributed training for the model
        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model
        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work cormappertly.
        :return: None
        """

        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self.actor_critic, self.device)
        self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        super().before_backward(loss)

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class DDPPO(DecentralizedDistributedMixin, PPO):
    """DDPPO wrapper around PPO"""
    pass
