# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import numpy as np

import torch


class RolloutStoragePol:
    """Class for storing rollout information about policy model for RL trainers.
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        config=None,
    ):
        """
        Create an object of a class for storing rollout information about policy model for chat2map active mapping trainer.
        :param num_steps: number of rollout steps before update
        :param num_envs: number of environments
        :param observation_space: observation space
        :param action_space: action space
        :param recurrent_hidden_state_size: poicy RNN hidden state size
        :param num_recurrent_layers: number of recurrent layers
        :param config: config
        """

        self.ppo_cfg = config.RL.PPO
        self.env_cfg = config.TASK_CONFIG.ENVIRONMENT
        self.sim_cfg = config.TASK_CONFIG.SIMULATOR

        self.observations = {}
        self.observationKeys_toSkip = ["rgb", "depth"]

        for sensor in observation_space.spaces:
            if sensor not in self.observationKeys_toSkip:
                self.observations[sensor] = torch.zeros(
                    num_steps + 1,
                    num_envs,
                    *observation_space.spaces[sensor].shape
                )

        self.is_random_agent = (self.ppo_cfg.agent_type == "random")
        if self.is_random_agent:
            self.num_agents = self.sim_cfg.ALL_AGENTS.NUM
            self.num_steps_per_agent = self.env_cfg.MAX_CONTEXT_LENGTH
            self.visual_budget = self.env_cfg.VISUAL_BUDGET

            self.num_max_set_context_views = self.visual_budget - self.num_agents
            self.total_num_idxs_sampled_context_views = (self.num_steps_per_agent - 1) * self.num_agents

            self.random_pol_oneHot_view_dist = torch.zeros(num_envs, self.total_num_idxs_sampled_context_views)
            self.current_idx_sampled_context_view = torch.zeros(num_envs)
            for env_idx in range(num_envs):
                idxs_to_set_context_views = np.random.choice(self.total_num_idxs_sampled_context_views,
                                                             self.num_max_set_context_views,
                                                             replace=False).tolist()
                self.random_pol_oneHot_view_dist[env_idx, idxs_to_set_context_views] = 1
        else:
            self.recurrent_hidden_states = torch.zeros(
                num_steps + 1,
                num_recurrent_layers,
                num_envs,
                recurrent_hidden_state_size,
            )

            self.rewards = torch.zeros(num_steps, num_envs, 1)
            self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
            self.returns = torch.zeros(num_steps + 1, num_envs, 1)

            self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = 1
            else:
                action_shape = action_space.shape[0]

            self.actions = torch.zeros(num_steps, num_envs, action_shape)
            self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
            if action_space.__class__.__name__ == "ActionSpace":
                self.actions = self.actions.long()
                self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        """set the device of all data containers in the rollout storage object"""
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        if not self.is_random_agent:
            self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.returns = self.returns.to(device)
            self.action_log_probs = self.action_log_probs.to(device)
            self.actions = self.actions.to(device)
            self.prev_actions = self.prev_actions.to(device)

        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        """insert stuff into the rollout storage"""
        for sensor in observations:
            if sensor not in self.observationKeys_toSkip:
                self.observations[sensor][self.step + 1].copy_(
                    observations[sensor]
                )

        if self.is_random_agent:
            for mask_idx in range(masks.size(0)):
                if masks[mask_idx].item() == 0.:
                    self.current_idx_sampled_context_view[mask_idx] = 0

                    idxs_to_set_context_views = np.random.choice(self.total_num_idxs_sampled_context_views,
                                                                 self.num_max_set_context_views,
                                                                 replace=False).tolist()

                    self.random_pol_oneHot_view_dist[mask_idx] = 0
                    self.random_pol_oneHot_view_dist[mask_idx, idxs_to_set_context_views] = 1
                else:
                    self.current_idx_sampled_context_view[mask_idx] += self.num_agents
                    self.current_idx_sampled_context_view[mask_idx] %= self.total_num_idxs_sampled_context_views
        else:
            self.recurrent_hidden_states[self.step + 1].copy_(
                recurrent_hidden_states
            )
            self.actions[self.step].copy_(actions)
            self.prev_actions[self.step + 1].copy_(actions)
            self.action_log_probs[self.step].copy_(action_log_probs)
            self.value_preds[self.step].copy_(value_preds)
            self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """rollout storage after policy update"""
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        if not self.is_random_agent:
            self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
            self.prev_actions[0].copy_(self.prev_actions[-1])

        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        """compute the return for an episode"""
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        """function to yield observations and targets in a sequential fashion during policy update"""
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind]
                )

                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).
        :param t: first dimension of tensor.
        :param n: second dimension of tensor.
        :param tensor: target tensor to be flattened.
        :return: flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorageMapper:
    """Class for storing rollout information about mapper
    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        eval_=False,
        num_agents=2,
    ):
        """
        Create an object of a class for storing rollout information about mapper.
        :param num_steps: number of rollout steps before update
        :param num_envs: number of environments
        :param observation_space: observation space
        :param eval_: flag saying if in eval mode or not
        :param num_agents: number of agents in a conversation episode
        """

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.eval = eval_
        self.num_agents = num_agents

        if not eval_:
            self.observations = {}
        self.observationKeys_toSkip = ["rgb", "depth", "current_context_idx", "all_context_audio_mask",
                                       "all_query_mask",  "current_context_rAz"]

        self.validObsKeys_2_storageKeys = {"current_context_rgb": "context_rgbs",
                                           "current_context_map": "context_maps",
                                           "current_context_pose": "context_views_pose",
                                           "previous_context_view_mask": "context_views_mask",
                                           "current_context_selfAudio": "context_selfAudio",
                                           "current_context_audio_mask": "context_selfAudio_mask",
                                           "current_context_otherAudio": "context_otherAudio",
                                           "current_context_otherAudio_pose": "context_otherAudio_pose",
                                           "current_query_mask": "query_views_mask",
                                           "current_query_globCanMapEgoCrop_gt": "query_globCanMapEgoCrop_gt",
                                           "current_query_globCanMapEgoCrop_gt_exploredPartMask":\
                                               "query_globCanMapEgoCrop_gt_exploredPartMask"
                                           }

        self.storageKeys_2_validObsKeys = dict()
        for k, v in self.validObsKeys_2_storageKeys.items():
            self.storageKeys_2_validObsKeys[v] = k

        if not eval_:
            for sensor in observation_space.spaces:
                if sensor not in self.observationKeys_toSkip:
                    assert sensor in self.validObsKeys_2_storageKeys
                    storage_sensor_name = self.validObsKeys_2_storageKeys[sensor]
                    self.observations[storage_sensor_name] = torch.zeros(
                        num_steps,
                        num_envs,
                        *observation_space.spaces[sensor].shape
                    )

    def to(self, device):
        """set the device of all data containers in the rollout storage object"""
        assert not self.eval
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

    def insert(
        self,
        observations,
        step,
    ):
        """insert stuff into the rollout storage"""
        assert not self.eval
        for sensor in observations:
            self.observations[sensor][step].copy_(
                observations[sensor]
            )

    def recurrent_generator(self, num_mini_batch):
        """function to yield observations and targets in a sequential fashion during mapper update"""
        assert not self.eval
        total_dataset_len = self.num_envs * self.num_steps
        assert total_dataset_len >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(total_dataset_len, num_mini_batch)
        )

        batch_size = int(np.round(total_dataset_len / num_mini_batch))
        perm = torch.randperm(total_dataset_len)
        for start_ind in range(0, total_dataset_len, batch_size):
            observations_batch = defaultdict(list)

            for offset in range(batch_size):
                if start_ind + offset == total_dataset_len:
                    break
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor].view((-1, *self.observations[sensor].size()[2:]))[ind, ...]
                    )

            """ These are all tensors of size (batch_size, ...) """
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 0
                )

            yield (
                observations_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).
        :param t: first dimension of tensor.
        :param n: second dimension of tensor.
        :param tensor: target tensor to be flattened.
        :return: flattened tensor of size (t*n, ...)
        """

        assert not self.eval
        return tensor.view(t * n, *tensor.size()[2:])
