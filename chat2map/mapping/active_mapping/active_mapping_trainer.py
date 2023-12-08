# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional, DefaultDict
import json
import random
import pickle
import gzip
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from torch import distributed as distrib

from habitat import Config, logger
from chat2map.common.base_trainer import BaseRLTrainer
from chat2map.common.baseline_registry import baseline_registry
from chat2map.common.env_utils import construct_envs
from chat2map.common.environments import get_env_class
from chat2map.common.rollout_storage import RolloutStoragePol, RolloutStorageMapper
from chat2map.common.tensorboard_utils import TensorboardWriter
from chat2map.mapping.active_mapping.ddppo_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from chat2map.common.utils import (
    batch_obs,
    linear_decay,
    load_points_data,
    get_stitched_top_down_maps,
)
from chat2map.common.loss_n_evalMetrics import compute_loss_n_evalMetrics
from chat2map.mapping.active_mapping.policy import ActiveMappingPolicy
from chat2map.mapping.active_mapping.active_mapping import PPO, DDPPO


SCENE_IDX_TO_NAME = {
    "mp3d":
        {0: 'sT4fr6TAbpF', 1: 'E9uDoFAP3SH', 2: 'VzqfbhrpDEA', 3: 'kEZ7cmS4wCh', 4: '29hnd4uzFmX', 5: 'ac26ZMwG7aT',
         6: 's8pcmisQ38h', 7: 'rPc6DW4iMge', 8: 'EDJbREhghzL', 9: 'mJXqzFtmKg4', 10: 'B6ByNegPMKs', 11: 'JeFG25nYj2p',
         12: '82sE5b5pLXE', 13: 'D7N2EKCX4Sj', 14: '7y3sRwLe3Va', 15: '5LpN3gDmAk7', 16: 'gTV8FGcVJC9', 17: 'ur6pFq6Qu1A',
         18: 'qoiz87JEwZ2', 19: 'PuKPg4mmafe', 20: 'VLzqgDo317F', 21: 'aayBHfsNo7d', 22: 'JmbYfDe2QKZ', 23: 'XcA2TqTSSAj',
         24: '8WUmhLawc2A', 25: 'sKLMLpTHeUy', 26: 'r47D5H71a5s', 27: 'Uxmj2M2itWa', 28: 'Pm6F8kyY3z2', 29: 'p5wJjkQkbXX',
         30: '759xd9YjKW5', 31: 'JF19kD82Mey', 32: 'V2XKFyX4ASd', 33: '1LXtFkjw3qL', 34: '17DRP5sb8fy', 35: '5q7pvUzZiYa',
         36: 'VVfe2KiqLaN', 37: 'Vvot9Ly1tCj', 38: 'ULsKaCPVFJR', 39: 'D7G3Y4RVNrH', 40: 'uNb9QFRL6hY', 41: 'ZMojNkEp431',
         42: '2n8kARJN3HM', 43: 'vyrNrziPKCB', 44: 'e9zR4mvMWw7', 45: 'r1Q1Z4BcV1o', 46: 'PX4nDJXEHrG', 47: 'YmJkqBEsHnH',
         48: 'b8cTxDM8gDG', 49: 'GdvgFV5R1Z5', 50: 'pRbA3pwrgk9', 51: 'jh4fc5c5qoQ', 52: '1pXnuDYAj8r', 53: 'S9hNv5qa7GM',
         54: 'VFuaQ6m2Qom', 55: 'cV4RVeZvu5T', 56: 'SN83YJsR3w2', 57: '2azQ1b91cZZ', 58: '5ZKStnWn8Zo', 59: '8194nk5LbLH',
         60: 'ARNzJeq3xxb', 61: 'EU6Fwq7SyZv', 62: 'QUCTc6BB5sX', 63: 'TbHJrupSAjP', 64: 'UwV83HsGsw3', 65: 'Vt2qJdWjCF2',
         66: 'WYY7iVyf5p8', 67: 'X7HyMhZNoso', 68: 'YFuZgdQ5vWj', 69: 'Z6MFQCViBuw', 70: 'fzynW3qQPVF', 71: 'gYvKGZ5eRqb',
         72: 'gxdoqLR6rwA', 73: 'jtcxE69GiFV', 74: 'oLBMNvg9in8', 75: 'pLe4wQe7qrG', 76: 'pa4otMbVnkk', 77: 'q9vSo1VnCiC',
         78: 'rqfALeAoiTq', 79: 'wc2JMjhGNzB', 80: 'x8F5xyUWy9e', 81: 'yqstnuAEVhm', 82: 'zsNo4HB9uLZ'},
}


SCENE_SPLITS = {
    "mp3d":
        {
            "train": ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX',
                      'ac26ZMwG7aT', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4',
                      '5LpN3gDmAk7', 'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe',
                      'VLzqgDo317F', 'aayBHfsNo7d', 'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A',
                      'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa', 'Pm6F8kyY3z2', 'p5wJjkQkbXX',
                      '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL', '17DRP5sb8fy',
                      '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
                      'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7',
                      'r1Q1Z4BcV1o', 'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5',
                      'pRbA3pwrgk9', 'jh4fc5c5qoQ', '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom',
                      'cV4RVeZvu5T', 'SN83YJsR3w2', ],
            "val": ['QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG', 'oLBMNvg9in8',
                    'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH', ],
            "test": ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'fzynW3qQPVF',
                     'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'gYvKGZ5eRqb', 'jtcxE69GiFV', 'gxdoqLR6rwA',
                     'WYY7iVyf5p8', 'YFuZgdQ5vWj', 'rqfALeAoiTq', 'x8F5xyUWy9e',],
        },
}


BINARY_MAP_QUALITY_METRICS = ["f1_score", "iou"]


@baseline_registry.register_trainer(name="chat2map_activeMappingTrainer")
class ActiveMappingTrainer(BaseRLTrainer):
    """Trainer class for chat2map active mapper
    PPO paper: https://arxiv.org/abs/1707.06347.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        
        assert "RGB_SENSOR" in self.config.SENSORS

        self.mapper_batch = None
        self.mapper_step_observation = None

    def _setup_activeMapping_agent(self, world_rank=0, eval=False) -> None:
        """
        Sets up chat2map active mapper model.
        :param world_rank: ddppo world rank
        :param eval: flag saying if in eval mode or not
        :return: None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        ppo_cfg = self.config.RL.PPO

        self.actor_critic = ActiveMappingPolicy(
            config=self.config,
            observation_space=self.envs.observation_spaces[0],
        )

        self.actor_critic.to(self.device)

        if not eval:
            self.load_pretrained_ckpt(is_randomAgent=(ppo_cfg.agent_type == "random"))

        if ppo_cfg.use_ddppo:
            self.agent = DDPPO(
                actor_critic=self.actor_critic,
                config=self.config,
            )
        else:
            self.agent = PPO(
                actor_critic=self.actor_critic,
                config=self.config,
            )

    def save_checkpoint(self,
                        file_name: str,
                        update: int,
                        count_steps: int,
                        lr_scheduler_pol,
                        best_performance=None) -> None:
        """
        Save checkpoint with specified name.
        :param file_name: file name for checkpoint
        :param update: number of model updates completed
        :param count_steps: number of update steps completed
        :param lr_scheduler_pol: learning rate scheduler for the chat2map active mapper policy
        :param best_performance: best validation performance of the chat2map active mapper yet
        :return: None
        """

        ppo_cfg = self.config.RL.PPO

        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }

        checkpoint["optimizer_mapper"] = self.agent.optimizer_mapper.state_dict()
        checkpoint["last_update"] = update
        checkpoint["last_count_steps"] = count_steps

        assert ppo_cfg.agent_type in ["random", "chat2map_activeMapper"]
        if ppo_cfg.agent_type == "chat2map_activeMapper":
            assert self.agent.optimizer_pol is not None
            checkpoint["optimizer_pol"] = self.agent.optimizer_pol.state_dict()

            assert lr_scheduler_pol is not None
            checkpoint["lr_scheduler_pol"] = lr_scheduler_pol.state_dict()

        if best_performance is not None:
            checkpoint["best_performance"] = best_performance

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """
        Load checkpoint of specified path as a dict.
        :param checkpoint_path: path of target checkpoint
        :param args: additional positional args
        :param kwargs: additional keyword args
        :return: dict containing checkpoint info
        """

        return torch.load(checkpoint_path, *args, **kwargs)

    def load_activeMapper_state_dict(self, checkpoint_path: str, lr_scheduler_pol, just_copy_config=False):
        """
        Load active mapper state dictionary
        :param checkpoint_path: path of target checkpoint
        :param lr_scheduler_pol: lr scheduler for chat2map active mapper policy
        :param just_copy_config: flag saying if config to be copied only during the function call or the full state dictionary
                                is to be read
        :return: tuple containing starting update and number of steps value for this training if just_copy_config=False,
                else None
        """
        ppo_cfg = self.config.RL.PPO

        ckpt_dict = torch.load(checkpoint_path, map_location="cpu")

        if just_copy_config:
            self.config = ckpt_dict["config"]
        else:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.agent.optimizer_mapper.load_state_dict(ckpt_dict["optimizer_mapper"])

            assert ppo_cfg.agent_type in ["random", "chat2map_activeMapper"]
            if ppo_cfg.agent_type == "chat2map_activeMapper":
                assert self.agent.optimizer_pol is not None
                self.agent.optimizer_pol.load_state_dict(ckpt_dict["optimizer_pol"])

                assert lr_scheduler_pol is not None
                lr_scheduler_pol.load_state_dict(ckpt_dict["lr_scheduler_pol"])

            return ckpt_dict["last_update"] + 1, ckpt_dict["last_count_steps"]

    def load_pretrained_ckpt(self, is_randomAgent=False):
        """
        load pretrained checkpoint
        :param is_randomAgent: flag saying if the agent being trained is a random agent or not
        :return: None
        """
        ppo_cfg = self.config.RL.PPO

        assert os.path.isfile(ppo_cfg.pretrained_ckpt_path)

        ckpt_dict = torch.load(ppo_cfg.pretrained_ckpt_path, map_location="cpu")

        for name in self.actor_critic.mapper.state_dict():
            if is_randomAgent:
                # pretrained w/ DataParallel
                self.actor_critic.mapper.state_dict()[name].copy_(ckpt_dict["state_dict"]["actor_critic.module." + name])
            else:
                self.actor_critic.mapper.state_dict()[name].copy_(ckpt_dict["state_dict"]["actor_critic.mapper." + name])

    def _collect_rollout_step(
            self,
            rollouts_pol,
            rollouts_mapper,
            current_episode_step,
            current_episode_reward,
            current_episode_map_metrics,
            current_episode_dist_probs,
            episode_counts,
            episode_steps,
            episode_rewards,
            episode_map_metrics_allSteps,
            episode_map_metrics_lastStep,
            episode_dist_probs,
    ):
        """
        collects rollouts for training mapper in supervised fashion and the policy with PPO
        """

        ppo_cfg = self.config.RL.PPO
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR

        num_agents = sim_cfg.ALL_AGENTS.NUM
        assert num_agents == 2
        num_actions = 4

        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        """sample action"""
        with torch.no_grad():
            step_observation = {
                k: v[rollouts_pol.step] for k, v in rollouts_pol.observations.items()
            }

            """get actions"""
            if ppo_cfg.agent_type == "random":
                actions = []
                for batch_idx, current_idx_sampled_context_view in enumerate(rollouts_pol.current_idx_sampled_context_view):
                    current_action_slice = rollouts_pol.random_pol_oneHot_view_dist[batch_idx][int(current_idx_sampled_context_view.item()):
                                                                    int(current_idx_sampled_context_view.item()) + num_agents].tolist()
                    if current_action_slice == [0, 0]:
                        actions.append([0])
                    elif current_action_slice == [1, 0]:
                        actions.append([1])
                    elif current_action_slice == [0, 1]:
                        actions.append([2])
                    elif current_action_slice == [1, 1]:
                        actions.append([3])
                    else:
                        raise ValueError
                actions = torch.tensor(actions, dtype=torch.uint8)

                values = torch.zeros((actions.size(0), 1)).to(actions.device)
                actions_log_probs = torch.log(torch.ones((actions.size(0), 1)) / num_actions).to(actions.device)
                distribution_probs = torch.ones((actions.size(0), num_actions)).to(actions.device) / num_actions
                recurrent_hidden_states = torch.zeros((1, actions.size(0), ppo_cfg.hidden_size)).to(actions.device)
            else:
                (
                    values,
                    actions,
                    actions_log_probs,
                    distribution_entropy,
                    recurrent_hidden_states,
                    distribution_probs,
                ) = self.actor_critic.act(
                    step_observation,
                    rollouts_pol.recurrent_hidden_states[rollouts_pol.step],
                    rollouts_pol.prev_actions[rollouts_pol.step],
                    rollouts_pol.masks[rollouts_pol.step],
                )

        """first action in action space is STOP (0), increment action returned by policy by 1 to prevent stopping the 
        episode."""
        actions += 1

        pth_time += time.time() - t_sample_action
        t_step_env = time.time()

        # print("actions: ", actions)
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        env_time += time.time() - t_step_env

        """inserting observations for current step into rollouts_mapper"""
        step_observation_batch =\
            self.insert_batch_rollouts_mapper(
                rollouts_mapper,
                step_observation,
                actions,
                rollouts_pol.step,
                dones,
            )

        batch = batch_obs(observations, self.device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float
        )

        mapper_step_observation, query_maps_gt, query_maps_exploredMasks = self.build_mapper_step_observation(
            {
                k: v[rollouts_pol.step] for k, v in rollouts_mapper.observations.items()
            }
        )

        with torch.no_grad():
            preds = self.actor_critic.mapper_forward(mapper_step_observation).detach()

        """setting vision mask corresponding to current action to 0 to get map quality w/o vision"""
        validObsKeys_2_rolloutsMapperStorageKeys = rollouts_mapper.validObsKeys_2_storageKeys
        for batch_idx in range(actions.size(0)):
            current_context_idx_this_action = int(step_observation["current_context_idx"][batch_idx][0].item())
            step_observation_batch[validObsKeys_2_rolloutsMapperStorageKeys["previous_context_view_mask"]][batch_idx, :, current_context_idx_this_action] = 0.

        mapper_step_observation_wo_vision, _, _ = self.build_mapper_step_observation(
            step_observation_batch
        )
        with torch.no_grad():
            preds_wo_vision = self.actor_critic.mapper_forward(mapper_step_observation_wo_vision).detach()

        """getting rewards"""
        repeatPose_penalties = []
        for batch_idx in range(self.envs.num_envs):
            action_thisBatchIdx = (actions[batch_idx][0].item())

            current_context_idx_thisBatchIdx = int(step_observation["current_context_idx"][batch_idx][0].item())
            current_context_rAz = step_observation["current_context_rAz"][batch_idx, :, current_context_idx_thisBatchIdx, :]

            repeatPose_penalties_thisBatch = []
            for agent_idx in range(current_context_rAz.size(0)):
                thisBatch_thisAgent_current_context_rAz = tuple(current_context_rAz[agent_idx].int().tolist())
                use_frame_flag = False
                if agent_idx == 0:
                    if action_thisBatchIdx in [2, 4]:
                        use_frame_flag = True
                elif agent_idx == 1:
                    if action_thisBatchIdx in [3, 4]:
                        use_frame_flag = True

                if use_frame_flag:
                    if thisBatch_thisAgent_current_context_rAz not in self.unique_pose_tracker[batch_idx]:
                        self.unique_pose_tracker[batch_idx][thisBatch_thisAgent_current_context_rAz] = 1
                        repeatPose_penalties_thisBatch.append(0.)
                    else:
                        # self.unique_pose_tracker[batch_idx][thisBatch_thisAgent_current_context_rAz] += 1
                        repeatPose_penalties_thisBatch.append(1.)
                else:
                    repeatPose_penalties_thisBatch.append(0.)

            repeatPose_penalties_thisBatch = np.mean(repeatPose_penalties_thisBatch)
            repeatPose_penalties.append(repeatPose_penalties_thisBatch)

            if dones[batch_idx]:
                self.unique_pose_tracker[batch_idx] = dict()

                current_context_rAz = batch["current_context_rAz"][batch_idx, :, 0, :]
                for agent_idx in range(current_context_rAz.size(0)):
                    thisBatch_thisAgent_current_context_rAz = tuple(current_context_rAz[agent_idx].int().tolist())
                    assert thisBatch_thisAgent_current_context_rAz not in self.unique_pose_tracker[batch_idx]
                    self.unique_pose_tracker[batch_idx][thisBatch_thisAgent_current_context_rAz] = 1

        rewards = self.override_rewards(
            preds=preds,
            query_views_mask=mapper_step_observation["query_views_mask"],
            query_maps_gt=query_maps_gt,
            query_maps_exploredMasks=query_maps_exploredMasks,
        )

        rewards_wo_vision = self.override_rewards(
            preds=preds_wo_vision,
            query_views_mask=mapper_step_observation["query_views_mask"],
            query_maps_gt=query_maps_gt,
            query_maps_exploredMasks=query_maps_exploredMasks,
        )

        rewards = ((np.array(rewards) - np.array(rewards_wo_vision)) / (np.array(rewards_wo_vision) + 1e-13)).tolist()
        rewards = (np.array(rewards) - ppo_cfg.repeatPose_penalty_weight * np.array(repeatPose_penalties)).tolist()
        """end of getting rewards"""

        t_update_stats = time.time()
        current_episode_map_metrics_thisStep = dict()
        for metric_name in current_episode_map_metrics:
            if metric_name in ["f1_score_1", "f1_score_0", "iou_1", "iou_0"]:
                if metric_name in ["f1_score_1", "f1_score_0"]:
                    metric_type = "f1_score"
                elif metric_name in ["iou_1", "iou_0"]:
                    metric_type = "iou"
                else:
                    raise ValueError

                pred_occMap = (preds[..., :1] > 0.5).float()
                exploredPart_mask = query_maps_exploredMasks[..., :1]

                if metric_name.split("_")[-1] == "1":
                    target_category = 1.0
                elif metric_name.split("_")[-1] == "0":
                    target_category = 0.0
                else:
                    raise ValueError
            elif metric_name.split("_")[-1] == "loss":
                metric_type = metric_name
                pred_occMap = preds[..., :1]
                exploredPart_mask = None
                target_category = None

            gt_occMap = query_maps_gt[..., :1]
            query_mask = mapper_step_observation["query_views_mask"]

            # print("metric type: ", metric_type)
            map_metric_all_batch_idxs = compute_loss_n_evalMetrics(
                loss_or_metric_types=[metric_type],
                loss_or_metric_weights=[1.0],
                gt_occMap=gt_occMap,
                pred_occMap=pred_occMap,
                mask=query_mask,
                exploredPart_mask=exploredPart_mask.detach() if (exploredPart_mask is not None) else exploredPart_mask,
                target_category=target_category,
                dont_collapse_across_batch=True,
            )

            current_episode_map_metrics_thisStep[metric_name] = map_metric_all_batch_idxs.unsqueeze(-1).detach().cpu()
            current_episode_map_metrics[metric_name] += current_episode_map_metrics_thisStep[metric_name]

        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        current_episode_reward += rewards.clone().detach()
        current_episode_step += 1
        current_episode_dist_probs += distribution_probs.detach().cpu()

        """
        current_episode_reward is accumulating rewards across multiple updates,
        as long as the current episode is not finished
        the current episode reward is added to the episode rewards only if the current episode is done
        the episode count will also increase by 1
        """
        for metric_name in current_episode_map_metrics:
            assert metric_name in episode_map_metrics_allSteps
            episode_map_metrics_allSteps[metric_name] += (1 - masks) * (current_episode_map_metrics[metric_name] / current_episode_step)
            episode_map_metrics_lastStep[metric_name] += (1 - masks) * current_episode_map_metrics_thisStep[metric_name]

        episode_rewards += (1 - masks) * current_episode_reward
        episode_steps += (1 - masks) * current_episode_step
        episode_counts += 1 - masks
        episode_dist_probs += (1 - masks) * (current_episode_dist_probs / current_episode_step)

        """zeroing out current values when done"""
        for metric_name in current_episode_map_metrics:
            current_episode_map_metrics[metric_name] *= masks
        current_episode_reward *= masks
        current_episode_step *= masks
        current_episode_dist_probs *= masks

        rollouts_pol.insert(
            batch,
            recurrent_hidden_states,
            actions - 1,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def override_rewards(
            self,
            preds=None,
            query_views_mask=None,
            query_maps_gt=None,
            query_maps_exploredMasks=None,
    ):
        """computing rewards"""
        ppo_cfg = self.config.RL.PPO
        assert ppo_cfg.reward_type == "f1_score"

        loss_or_metric_types = [ppo_cfg.reward_type]

        eval_metric_1 = compute_loss_n_evalMetrics(
            loss_or_metric_types=loss_or_metric_types,
            loss_or_metric_weights=[1.0],
            gt_occMap=query_maps_gt[..., :1],
            pred_occMap=(preds[..., :1] > 0.5).float(),
            mask=query_views_mask,
            exploredPart_mask=query_maps_exploredMasks[..., :1],
            target_category=1.0,
            dont_collapse_across_batch=True,
        )

        eval_metric_0 = compute_loss_n_evalMetrics(
            loss_or_metric_types=[ppo_cfg.reward_type],
            loss_or_metric_weights=[1.0],
            gt_occMap=query_maps_gt[..., :1],
            pred_occMap=(preds[..., :1] > 0.5).float(),
            mask=query_views_mask,
            exploredPart_mask=query_maps_exploredMasks[..., :1],
            target_category=0.,
            dont_collapse_across_batch=True,
        )

        rewards = (0.5 * (eval_metric_1 + eval_metric_0)).tolist()

        return rewards

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

    def insert_batch_rollouts_mapper(self,
                                     rollouts_mapper,
                                     step_observation,
                                     actions,
                                     rollout_step_idx,
                                     dones,
                                     eval_=False,
                                     eval_first_step=False
                                     ):
        """insert batch in the rollout storage for the mapper"""
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        env_cfg = self.config.TASK_CONFIG.ENVIRONMENT

        visual_budget = env_cfg.VISUAL_BUDGET

        num_agents = sim_cfg.ALL_AGENTS.NUM
        assert num_agents == 2

        validObsKeys_2_rolloutsMapperStorageKeys = rollouts_mapper.validObsKeys_2_storageKeys
        batch = dict()
        for k, v in validObsKeys_2_rolloutsMapperStorageKeys.items():
            if k.split("_")[-1] != "mask":
                batch[v] = step_observation[k]

        updated_context_views_mask = step_observation["previous_context_view_mask"].clone()

        context_selfAudio_mask = []
        query_views_mask = []
        for batch_idx in range(actions.size(0)):
            action = actions[batch_idx][0].item()
            current_context_idx_this_action = int(step_observation["current_context_idx"][batch_idx][0].item())

            if not eval_first_step:
                if action == 1:
                    updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 0.
                    updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 0.
                elif action == 2:
                    if torch.sum(updated_context_views_mask[batch_idx]).item() + 1 <= visual_budget:
                        updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 1.
                    else:
                        updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 0.
                    updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 0.
                elif action == 3:
                    updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 0.
                    if torch.sum(updated_context_views_mask[batch_idx]).item() + 1 <= visual_budget:
                        updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 1.
                    else:
                        updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 0.
                elif action == 4:
                    if torch.sum(updated_context_views_mask[batch_idx]).item() + 2 <= visual_budget:
                        updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 1.
                        updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 1.
                    elif torch.sum(updated_context_views_mask[batch_idx]).item() + 1 <= visual_budget:
                        updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 1.
                        updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 0.
                    else:
                        updated_context_views_mask[batch_idx, 0, current_context_idx_this_action] = 0.
                        updated_context_views_mask[batch_idx, 1, current_context_idx_this_action] = 0.
                else:
                    raise ValueError

            done = dones[batch_idx]
            if done:
                context_selfAudio_mask.append(
                    step_observation["all_context_audio_mask"][batch_idx]
                )
                query_views_mask.append(
                    step_observation["all_query_mask"][batch_idx]
                )
            else:
                context_selfAudio_mask.append(
                    step_observation["current_context_audio_mask"][batch_idx]
                )
                query_views_mask.append(
                    step_observation["current_query_mask"][batch_idx]
                )

        batch[validObsKeys_2_rolloutsMapperStorageKeys["previous_context_view_mask"]] = updated_context_views_mask
        batch[validObsKeys_2_rolloutsMapperStorageKeys["current_context_audio_mask"]] = torch.stack(context_selfAudio_mask, dim=0)
        batch[validObsKeys_2_rolloutsMapperStorageKeys["current_query_mask"]] = torch.stack(query_views_mask, dim=0)

        if not eval_:
            rollouts_mapper.insert(
                batch,
                rollout_step_idx,
            )

        return batch

    def _update_pol(self, rollouts_pol):
        """
        updates policy
        :param rollouts_pol: rollout storage for the policy
        :return: 1. time.time() - t_update_model: time needed for policy update
                 2. value_loss: PPO value loss in this update
                 3. action_loss: PPO actions loss in this update
                 4. dist_entropy: PPO entropy loss in this update
        """
        ppo_cfg = self.config.RL.PPO

        t_update_model = time.time()
        value_loss = 0.
        action_loss = 0.
        dist_entropy = 0.

        if ppo_cfg.agent_type == "chat2Map_activeMapper":
            self.actor_critic.eval()
            with torch.no_grad():
                last_observation = {
                    k: v[-1] for k, v in rollouts_pol.observations.items()
                }

                next_value = self.actor_critic.get_value(
                    last_observation,
                    rollouts_pol.recurrent_hidden_states[-1],
                    rollouts_pol.prev_actions[-1],
                    rollouts_pol.masks[-1],
                ).detach()

            rollouts_pol.compute_returns(
                next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
            )
            self.actor_critic.train()

            value_loss, action_loss, dist_entropy = self.agent.update_pol(rollouts_pol)

        rollouts_pol.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _update_mapper(self, rollouts_mapper):
        """
        updates policy
        :param rollouts_mapper: rollout storage for the mapper
        :return: 1. time.time() - t_update_model: time needed for policy update
                 2. mapper_loss: mapper training loss
        """
        ppo_cfg = self.config.RL.PPO

        t_update_model = time.time()

        if ppo_cfg.freeze_mapper:
            mapper_loss = 0.
        else:
            mapper_loss = self.agent.update_mapper(rollouts_mapper)

        return (
            time.time() - t_update_model,
            mapper_loss,
        )

    def train(self) -> None:
        """
        Main method for training and validating the Chat2Map active mapper model.
        :return: None
        """

        count_checkpoints = 0
        if self.config.RESUME_AFTER_PREEMPTION:
            old_ckpt_found = False
            if os.path.isdir(self.config.CHECKPOINT_FOLDER) and (len(os.listdir(self.config.CHECKPOINT_FOLDER)) != 0):
                lst_ckpt_filenames = os.listdir(self.config.CHECKPOINT_FOLDER)

                ckpt_file_maxIdx = float('-inf')
                for ckpt_filename in lst_ckpt_filenames:
                    if (ckpt_filename.split(".")[1] not in ["best_reward_avgAllSteps", "best_reward_lastStep"])\
                            and (int(ckpt_filename.split(".")[1]) > ckpt_file_maxIdx):
                        ckpt_file_maxIdx = int(ckpt_filename.split(".")[1])
                most_recent_ckpt_filename = f"ckpt.{ckpt_file_maxIdx}.pth"

                most_recent_ckpt_file_path = os.path.join(self.config.CHECKPOINT_FOLDER,
                                                          most_recent_ckpt_filename)
                count_checkpoints = int(most_recent_ckpt_filename.split(".")[1]) + 1
                old_ckpt_found = True

            old_tb_dir = os.path.join(self.config.MODEL_DIR, "tb")
            if os.path.isdir(old_tb_dir):
                for old_tb_idx in range(1, 10000):
                    if not os.path.isdir(os.path.join(self.config.MODEL_DIR, f"tb_{old_tb_idx}")):
                        new_tb_dir = os.path.join(self.config.MODEL_DIR, f"tb_{old_tb_idx}")
                        os.system(f"mv {old_tb_dir} {new_tb_dir}")
                        break

            if old_ckpt_found:
                assert os.path.isfile(most_recent_ckpt_file_path)

        ppo_cfg = self.config.RL.PPO
        task_cfg = self.config.TASK_CONFIG
        env_cfg = self.config.TASK_CONFIG.ENVIRONMENT
        sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        passive_mapping_cfg = self.config.PassiveMapping

        num_agents = sim_cfg.ALL_AGENTS.NUM
        assert num_agents == 2

        val_while_training = self.config.VAL_WHILE_TRAINING
        if val_while_training:
            assert not self.config.STITCH_TOP_DOWN_MAPS_ACTIVE_MAPPING

        if ppo_cfg.use_ddppo:
            self.local_rank, tcp_store =\
                init_distrib_slurm(
                    ppo_cfg.ddppo_distrib_backend,
                    master_port=ppo_cfg.master_port,
                    master_addr=ppo_cfg.master_addr,
                )
            add_signal_handlers()

            num_rollouts_done_store = distrib.PrefixStore(
                "rollout_tracker", tcp_store
            )
            num_rollouts_done_store.set("num_done", "0")

            self.world_rank = distrib.get_rank()
            self.world_size = distrib.get_world_size()

            self.config.defrost()
            self.config.TORCH_GPU_ID = self.local_rank
            self.config.SIMULATOR_GPU_ID = self.local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.SEED += (
                self.world_rank * self.config.NUM_PROCESSES
            )
            self.config.TASK_CONFIG.SIMULATOR.SEED = self.config.SEED
            self.config.freeze()

        if (not ppo_cfg.use_ddppo) or (ppo_cfg.use_ddppo and (self.world_rank == 0)):
            logger.info(f"config: {self.config}")

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        if val_while_training and (ppo_cfg.agent_type == "random"):
            max_eval_episodes = 1500
            num_steps_per_agent = env_cfg.MAX_CONTEXT_LENGTH
            total_num_visual_frames = env_cfg.VISUAL_BUDGET

            num_max_set_context_views = total_num_visual_frames - num_agents
            total_num_idxs_sampled_context_views = (num_steps_per_agent - 1) * num_agents

            random_pol_oneHot_view_dist = torch.zeros(max_eval_episodes, total_num_idxs_sampled_context_views)
            for ep_idx in tqdm(range(max_eval_episodes)):
                idxs_to_set_context_views = np.random.choice(total_num_idxs_sampled_context_views,
                                                             num_max_set_context_views,
                                                             replace=False).tolist()

                random_pol_oneHot_view_dist[ep_idx, idxs_to_set_context_views] = 1

        self.envs = construct_envs(
            self.config,
            get_env_class(self.config.ENV_NAME),
            workers_ignore_signals=True if ppo_cfg.use_ddppo else False,
            is_train=True,
        )

        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if ppo_cfg.use_ddppo:
            torch.cuda.set_device(self.device)

        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_activeMapping_agent(world_rank=self.world_rank if ppo_cfg.use_ddppo else 0)

        if ppo_cfg.use_ddppo:
            self.agent.init_distributed(find_unused_params=True)
        if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(param.numel() for param in self.agent.parameters() if param.requires_grad)
                )
            )

        rollouts_pol = RolloutStoragePol(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=ppo_cfg.rnn_num_layers,
            config=self.config,
        )
        rollouts_pol.to(self.device)

        rollouts_mapper = RolloutStorageMapper(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            num_agents=num_agents,
        )
        rollouts_mapper.to(self.device)

        observations = self.envs.reset()

        if ppo_cfg.use_ddppo:
            batch = batch_obs(observations, device=self.device)
        else:
            batch = batch_obs(observations)

        for sensor in rollouts_pol.observations:
            rollouts_pol.observations[sensor][0].copy_(batch[sensor])

        self.unique_pose_tracker = []
        for env_idx in range(self.envs.num_envs):
            self.unique_pose_tracker.append(dict())
        current_context_rAz = batch["current_context_rAz"][:, :, 0, :]
        for batch_idx in range(current_context_rAz.size(0)):
            for agent_idx in range(current_context_rAz[batch_idx].size(0)):
                thisBatch_thisAgent_current_context_rAz = tuple(current_context_rAz[batch_idx][agent_idx].int().tolist())
                assert thisBatch_thisAgent_current_context_rAz not in self.unique_pose_tracker[batch_idx]
                self.unique_pose_tracker[batch_idx][thisBatch_thisAgent_current_context_rAz] = 1

        """episode_x accumulates over the entire training"""
        num_actions = 4

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        current_episode_step = torch.zeros(self.envs.num_envs, 1)
        current_episode_dist_probs = torch.zeros(self.envs.num_envs, num_actions)

        episode_rewards = torch.zeros(self.envs.num_envs, 1)
        episode_counts = torch.zeros(self.envs.num_envs, 1)
        episode_steps = torch.zeros(self.envs.num_envs, 1)
        episode_dist_probs = torch.zeros(self.envs.num_envs, num_actions)

        window_episode_reward = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_counts = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_step = deque(maxlen=ppo_cfg.reward_window_size)
        window_episode_dist_probs = deque(maxlen=ppo_cfg.reward_window_size)

        current_episode_map_metrics = dict()
        episode_map_metrics_allSteps = dict()
        window_episode_map_metrics_allSteps = dict()
        episode_map_metrics_lastStep = dict()
        window_episode_map_metrics_lastStep = dict()
        for rec_metric in (passive_mapping_cfg.EvalMetrics.types + passive_mapping_cfg.TrainLosses.types):
            if rec_metric in BINARY_MAP_QUALITY_METRICS:
                current_episode_map_metrics[rec_metric + "_1"] = torch.zeros(self.envs.num_envs, 1)
                current_episode_map_metrics[rec_metric + "_0"] = torch.zeros(self.envs.num_envs, 1)

                episode_map_metrics_allSteps[rec_metric + "_1"] = torch.zeros(self.envs.num_envs, 1)
                episode_map_metrics_allSteps[rec_metric + "_0"] = torch.zeros(self.envs.num_envs, 1)
                window_episode_map_metrics_allSteps[rec_metric + "_1"] = deque(maxlen=ppo_cfg.reward_window_size)
                window_episode_map_metrics_allSteps[rec_metric + "_0"] = deque(maxlen=ppo_cfg.reward_window_size)

                episode_map_metrics_lastStep[rec_metric + "_1"] = torch.zeros(self.envs.num_envs, 1)
                episode_map_metrics_lastStep[rec_metric + "_0"] = torch.zeros(self.envs.num_envs, 1)
                window_episode_map_metrics_lastStep[rec_metric + "_1"] = deque(maxlen=ppo_cfg.reward_window_size)
                window_episode_map_metrics_lastStep[rec_metric + "_0"] = deque(maxlen=ppo_cfg.reward_window_size)
            else:
                current_episode_map_metrics[rec_metric] = torch.zeros(self.envs.num_envs, 1)

                episode_map_metrics_allSteps[rec_metric] = torch.zeros(self.envs.num_envs, 1)
                window_episode_map_metrics_allSteps[rec_metric] = deque(maxlen=ppo_cfg.reward_window_size)

                episode_map_metrics_lastStep[rec_metric] = torch.zeros(self.envs.num_envs, 1)
                window_episode_map_metrics_lastStep[rec_metric] = deque(maxlen=ppo_cfg.reward_window_size)

        if val_while_training:
            best_eval_episode_reward_allEps = float('-inf')
            best_eval_episode_reward_lastStep_allEps = float('-inf')

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0

        if ppo_cfg.agent_type == "chat2map_activeMapper":
            lr_scheduler_pol = LambdaLR(
                optimizer=self.agent.optimizer_pol,
                lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
            )

        starting_update = 0
        if self.config.RESUME_AFTER_PREEMPTION and old_ckpt_found:
            starting_update, count_steps =\
                self.load_activeMapper_state_dict(most_recent_ckpt_file_path,
                                                  lr_scheduler_pol=lr_scheduler_pol if\
                                                      (ppo_cfg.agent_type == "chat2map_activeMapper") else None,
                                                  just_copy_config=False,
                                                  )

        if ppo_cfg.use_ddppo:
            writer_obj = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            ) if self.world_rank == 0 else contextlib.suppress()
        else:
            writer_obj = TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )

        with writer_obj as writer:
            for update in range(starting_update, self.config.NUM_UPDATES):
                if ppo_cfg.agent_type == "chat2map_activeMapper":
                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler_pol.step()
                    if ppo_cfg.use_linear_clip_decay:
                        self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                            update, self.config.NUM_UPDATES
                        )

                if ppo_cfg.use_ddppo:
                    count_steps_delta = 0

                self.actor_critic.eval()

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps =\
                        self._collect_rollout_step(
                            rollouts_pol,
                            rollouts_mapper,
                            current_episode_step,
                            current_episode_reward,
                            current_episode_map_metrics,
                            current_episode_dist_probs,
                            episode_counts,
                            episode_steps,
                            episode_rewards,
                            episode_map_metrics_allSteps,
                            episode_map_metrics_lastStep,
                            episode_dist_probs,
                        )

                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    if ppo_cfg.use_ddppo:
                        count_steps_delta += delta_steps
                        if (
                            step
                            >= ppo_cfg.num_steps * ppo_cfg.short_rollout_threshold
                        ) and int(num_rollouts_done_store.get("num_done")) > (
                            ppo_cfg.sync_frac * self.world_size
                        ):
                            break
                    else:
                        count_steps += delta_steps

                if ppo_cfg.use_ddppo:
                    num_rollouts_done_store.add("num_done", 1)

                self.actor_critic.train()
                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_pol(
                    rollouts_pol
                )
                pth_time += delta_pth_time

                delta_pth_time, mapper_loss = self._update_mapper(
                    rollouts_mapper
                )
                pth_time += delta_pth_time
                self.actor_critic.eval()

                """computing stats"""
                if ppo_cfg.use_ddppo:
                    stat_idx = 0
                    stat_idx_num_actions = 0
                    stat_name_to_idx = {}
                    stat_name_to_idx_num_actions = {}
                    stack_lst_for_stats = []
                    stack_lst_for_stats_num_actions = []

                    stack_lst_for_stats.append(episode_rewards)
                    stat_name_to_idx["rewards"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats.append(episode_counts)
                    stat_name_to_idx["counts"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats.append(episode_steps)
                    stat_name_to_idx["steps"] = stat_idx
                    stat_idx += 1

                    stack_lst_for_stats_num_actions.append(episode_dist_probs)
                    stat_name_to_idx_num_actions["dist_probs"] = stat_idx_num_actions
                    stat_idx_num_actions += 1

                    for metric_name in episode_map_metrics_allSteps:
                        stack_lst_for_stats.append(episode_map_metrics_allSteps[metric_name])
                        stat_name_to_idx["all_" + metric_name] = stat_idx
                        stat_idx += 1

                    for metric_name in episode_map_metrics_lastStep:
                        stack_lst_for_stats.append(episode_map_metrics_lastStep[metric_name])
                        stat_name_to_idx["last_" + metric_name] = stat_idx
                        stat_idx += 1

                    stats = torch.stack(stack_lst_for_stats, 0).to(self.device)
                    distrib.all_reduce(stats)
                    stats_num_actions = torch.stack(stack_lst_for_stats_num_actions, 0).to(self.device)
                    distrib.all_reduce(stats_num_actions)

                    window_episode_reward.append(stats[stat_name_to_idx["rewards"]].clone())
                    window_episode_counts.append(stats[stat_name_to_idx["counts"]].clone())
                    window_episode_step.append(stats[stat_name_to_idx["steps"]].clone())
                    window_episode_dist_probs.append(stats_num_actions[stat_name_to_idx_num_actions["dist_probs"]].clone())
                    for metric_name in episode_map_metrics_allSteps:
                        window_episode_map_metrics_allSteps[metric_name].append(stats[stat_name_to_idx["all_" + metric_name]].clone())
                    for metric_name in episode_map_metrics_lastStep:
                        window_episode_map_metrics_lastStep[metric_name].append(stats[stat_name_to_idx["last_" + metric_name]].clone())

                    stats = torch.tensor(
                        [value_loss, action_loss, dist_entropy, mapper_loss, count_steps_delta], device=self.device,
                    )
                    distrib.all_reduce(stats)
                    count_steps += stats[4].item()

                    if self.world_rank == 0:
                        num_rollouts_done_store.set("num_done", "0")
                        value_loss = stats[0].item() / self.world_size
                        action_loss = stats[1].item() / self.world_size
                        dist_entropy = stats[2].item() / self.world_size
                        mapper_loss = stats[3].item() / self.world_size
                else:
                    window_episode_reward.append(episode_rewards.clone())
                    window_episode_counts.append(episode_counts.clone())
                    window_episode_step.append(episode_steps.clone())
                    window_episode_dist_probs.append(episode_dist_probs.clone())
                    for metric_name in episode_map_metrics_allSteps:
                        window_episode_map_metrics_allSteps[metric_name].append(episode_map_metrics_allSteps[metric_name].clone())
                    for metric_name in episode_map_metrics_lastStep:
                        window_episode_map_metrics_lastStep[metric_name].append(episode_map_metrics_lastStep[metric_name].clone())

                if (ppo_cfg.use_ddppo and self.world_rank == 0) or (not ppo_cfg.use_ddppo):
                    stats_keys = ["count", "reward", "step", 'dist_probs',]
                    stats_vals = [window_episode_counts, window_episode_reward, window_episode_step, window_episode_dist_probs,]
                    for metric_name in window_episode_map_metrics_allSteps:
                        stats_keys.append("all_" + metric_name)
                        stats_vals.append(window_episode_map_metrics_allSteps[metric_name])
                    for metric_name in window_episode_map_metrics_lastStep:
                        stats_keys.append("last_" + metric_name)
                        stats_vals.append(window_episode_map_metrics_lastStep[metric_name])

                    stats = zip(stats_keys, stats_vals)

                    deltas = {}
                    for k, v in stats:
                        if len(v) > 1:
                            deltas[k] = (v[-1] - v[0]).sum(dim=0)\
                                if (k == "dist_probs")\
                                else (v[-1] - v[0]).sum().item()
                        else:
                            deltas[k] = v[0].sum(dim=0) if (k == "dist_probs")\
                                else v[0].sum().item()

                    deltas["count"] = max(deltas["count"], 1.0)

                    """
                    this reward is averaged over all the episodes happened during window_size updates 
                    approximately number of steps is window_size * num_steps
                    """
                    logging.debug('Number of steps: {}'.format(deltas["step"] / deltas["count"]))
                    for k in deltas:
                        if k in ["count", "dist_probs"]:
                            continue

                        if k == "reward":
                            writer_str = "Environment/Reward"
                        elif k == "step":
                            writer_str = "Environment/Episode_length"
                        else:
                            writer_str = f"Mapper/{k}"

                        writer.add_scalar(
                            writer_str, deltas[k] / deltas["count"], count_steps
                        )

                    for i in range(num_actions):
                        if not isinstance(deltas['dist_probs'] / deltas["count"], float):
                            writer.add_scalar(
                                "Policy/Action_prob_{}".format(i), (deltas['dist_probs'] / deltas["count"])[i].item(),
                                count_steps
                            )
                        else:
                            writer.add_scalar(
                                "Policy/Action_prob_{}".format(i), deltas['dist_probs'] / deltas["count"], count_steps
                            )

                    writer.add_scalar(
                        'Policy/Value_Loss', value_loss, count_steps
                    )
                    writer.add_scalar(
                        'Policy/Action_Loss', action_loss, count_steps
                    )
                    writer.add_scalar(
                        'Policy/Entropy', dist_entropy, count_steps
                    )
                    if ppo_cfg.agent_type == "chat2map_activeMapper":
                        writer.add_scalar(
                            'Policy/Learning_Rate', lr_scheduler_pol.get_lr()[0], count_steps
                        )
                    writer.add_scalar(
                        'Mapper/Loss', mapper_loss, count_steps
                    )

                    """log stats"""
                    if (update > 0) and (update % self.config.LOG_INTERVAL == 0):

                        window_rewards = (
                            window_episode_reward[-1] - window_episode_reward[0]
                        ).sum()
                        window_counts = (
                            window_episode_counts[-1] - window_episode_counts[0]
                        ).sum()

                        if window_counts > 0:
                            logger.info(
                                "Average window size {} reward: {:3f}".format(
                                    len(window_episode_reward),
                                    (window_rewards / window_counts).item(),
                                )
                            )
                        else:
                            logger.info("No episodes finish in current window")

                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update, count_steps / (time.time() - t_start)
                            )
                        )
                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )

                    """save checkpoints"""
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        if val_while_training:
                            self.eval_envs = construct_envs(
                                self.config,
                                get_env_class(self.config.ENV_NAME),
                                workers_ignore_signals=False,
                                is_train=False,
                            )

                            """ temporary variables / objects for eval """
                            eval_not_done_masks = torch.ones(
                                self.eval_envs.num_envs, 1, device=self.device
                            )
                            eval_episode_idxs = torch.FloatTensor(list(range(self.eval_envs.num_envs))).to(device=self.device).unsqueeze(-1)
                            eval_episode_steps = torch.zeros(self.eval_envs.num_envs, 1, device=self.device)
                            eval_recurrent_hidden_states_pol = torch.zeros(
                                1 if ppo_cfg.agent_type == "random"\
                                    else self.actor_critic.policy.num_recurrent_layers,
                                self.eval_envs.num_envs,
                                ppo_cfg.hidden_size,
                                device=self.device,
                            )
                            eval_prev_actions = torch.zeros(
                                self.eval_envs.num_envs, 1, device=self.device, dtype=torch.long
                            )

                            tqdm_iterator = tqdm(total=self.config.EVAL.EPISODE_COUNT)

                            eval_current_episode_reward = torch.zeros(
                                self.eval_envs.num_envs, 1, device=self.device
                            )

                            eval_rewardPerStep_perSceneEpId = dict()
                            eval_stats_episodes = set()

                            eval_lastReward_perSceneEpId = dict()
                            eval_avgReward_perSceneEpId = dict()

                            eval_observations = self.eval_envs.reset()
                            eval_batch = batch_obs(eval_observations, device=self.device)

                            while (
                                    len(eval_stats_episodes) < self.config.EVAL.EPISODE_COUNT
                                    and (self.eval_envs.num_envs > 0)
                            ):
                                eval_current_episodes = self.eval_envs.current_episodes()

                                with torch.no_grad():
                                    eval_step_observation = eval_batch

                                    """get actions"""
                                    if ppo_cfg.agent_type == "random":
                                        eval_actions = []
                                        for eval_batch_idx, eval_episode_step in enumerate(eval_episode_steps):
                                            eval_ep_idx = int(eval_current_episodes[eval_batch_idx].episode_id) - 1
                                            eval_episode_step_ = int(eval_episode_step[0].item())

                                            current_action_slice =\
                                                random_pol_oneHot_view_dist[eval_ep_idx][eval_episode_step_ * num_agents:
                                                                                         (eval_episode_step_ + 1) * num_agents].tolist()
                                            if current_action_slice == [0, 0]:
                                                eval_actions.append([0])
                                            elif current_action_slice == [1, 0]:
                                                eval_actions.append([1])
                                            elif current_action_slice == [0, 1]:
                                                eval_actions.append([2])
                                            elif current_action_slice == [1, 1]:
                                                eval_actions.append([3])
                                            else:
                                                raise ValueError

                                        eval_actions = torch.tensor(eval_actions, dtype=torch.uint8).to(self.device)
                                    else:
                                        # raise NotImplementedError
                                        (
                                            _,
                                            eval_actions,
                                            _,
                                            _,
                                            eval_recurrent_hidden_states_pol,
                                            _,
                                        ) = self.actor_critic.act(
                                            eval_step_observation,
                                            eval_recurrent_hidden_states_pol,
                                            eval_prev_actions,
                                            eval_not_done_masks,
                                            deterministic=ppo_cfg.deterministic_eval,
                                        )

                                """first action in action space is STOP (0), increment action returned by policy by 1 to prevent stopping the 
                                episode. Fix later."""
                                eval_actions += 1

                                eval_outputs = self.eval_envs.step([a[0].item() for a in eval_actions])

                                eval_prev_actions = eval_actions - 1

                                eval_observations, eval_rewards, eval_dones, eval_infos = [
                                    list(x) for x in zip(*eval_outputs)
                                ]

                                """inserting observations for current step into rollouts_mapper"""
                                eval_mapper_batch =\
                                    self.insert_batch_rollouts_mapper(
                                        rollouts_mapper,
                                        eval_batch,
                                        eval_actions,
                                        None,
                                        eval_dones,
                                        eval_=True
                                    )

                                eval_not_done_masks = torch.tensor(
                                    [[0.0] if done else [1.0] for done in eval_dones],
                                    dtype=torch.float,
                                    device=self.device,
                                )

                                eval_mapper_step_observation, eval_query_maps_gt, eval_query_maps_exploredMasks =\
                                    self.build_mapper_step_observation(
                                        eval_mapper_batch,
                                    )

                                with torch.no_grad():
                                    eval_preds = self.actor_critic.mapper_forward(eval_mapper_step_observation).detach()

                                """ getting rewards """
                                eval_rewards = self.override_rewards(
                                    preds=eval_preds,
                                    query_views_mask=eval_mapper_step_observation["query_views_mask"],
                                    query_maps_gt=eval_query_maps_gt,
                                    query_maps_exploredMasks=eval_query_maps_exploredMasks,
                                )

                                eval_batch = batch_obs(eval_observations, self.device)

                                for i in range(self.eval_envs.num_envs):
                                    eval_scene_ep_id = (
                                        eval_current_episodes[i].scene_id.split("/")[-2],
                                        eval_current_episodes[i].episode_id
                                    )
                                    if eval_scene_ep_id not in eval_rewardPerStep_perSceneEpId:
                                        eval_rewardPerStep_perSceneEpId[eval_scene_ep_id] = [eval_rewards[i]]
                                    else:
                                        eval_rewardPerStep_perSceneEpId[eval_scene_ep_id].append(eval_rewards[i])

                                    eval_lastReward_perSceneEpId[eval_scene_ep_id] = eval_rewards[i]
                                    eval_avgReward_perSceneEpId[eval_scene_ep_id] = np.mean(eval_rewardPerStep_perSceneEpId[eval_scene_ep_id])

                                eval_episode_steps += 1

                                eval_next_episodes = self.eval_envs.current_episodes()

                                eval_envs_to_pause = []
                                for i in range(self.eval_envs.num_envs):
                                    """ pause envs which runs out of episodes """
                                    if (
                                        eval_next_episodes[i].scene_id.split("/")[-2],
                                        eval_next_episodes[i].episode_id,
                                    ) in eval_stats_episodes:
                                        eval_envs_to_pause.append(i)

                                    """ episode ended """
                                    if eval_not_done_masks[i].item() == 0:
                                        eval_stats_episodes.add(
                                            (
                                                eval_current_episodes[i].scene_id.split("/")[-2],
                                                eval_current_episodes[i].episode_id
                                            )
                                        )
                                        eval_current_episode_reward[i] = 0
                                        eval_episode_steps[i] = 0
                                        tqdm_iterator.update()

                                (
                                    self.eval_envs,
                                    eval_recurrent_hidden_states_pol,
                                    eval_not_done_masks,
                                    eval_current_episode_reward,
                                    eval_batch,
                                    eval_episode_steps,
                                    eval_episode_idxs,
                                ) = self._pause_envs(
                                    eval_envs_to_pause,
                                    self.eval_envs,
                                    eval_recurrent_hidden_states_pol,
                                    eval_not_done_masks,
                                    eval_current_episode_reward,
                                    eval_batch,
                                    eval_episode_steps,
                                    eval_episode_idxs,
                                )

                            """ closing the open environments after iterating over all episodes """
                            self.eval_envs.close()

                            avg_eval_lastReward_perSceneEpId = np.mean(list(eval_lastReward_perSceneEpId.values()))
                            writer.add_scalar(
                                f"Mapper/val_last_{ppo_cfg.reward_type}", avg_eval_lastReward_perSceneEpId, update
                            )
                            if avg_eval_lastReward_perSceneEpId > best_eval_episode_reward_lastStep_allEps:
                                best_eval_episode_reward_lastStep_allEps = avg_eval_lastReward_perSceneEpId
                                self.save_checkpoint(f"ckpt.best_reward_lastStep.pth",
                                                     update,
                                                     count_steps,
                                                     lr_scheduler_pol=lr_scheduler_pol if\
                                                         (ppo_cfg.agent_type == "chat2map_activeMapper")\
                                                         else None,
                                                     best_performance=best_eval_episode_reward_lastStep_allEps,
                                                     )

                            avg_eval_avgReward_perSceneEpId = np.mean(list(eval_avgReward_perSceneEpId.values()))
                            writer.add_scalar(
                                f"Mapper/val_allSteps_{ppo_cfg.reward_type}", avg_eval_avgReward_perSceneEpId, update
                            )
                            if avg_eval_avgReward_perSceneEpId > best_eval_episode_reward_allEps:
                                best_eval_episode_reward_allEps = avg_eval_avgReward_perSceneEpId
                                self.save_checkpoint(f"ckpt.best_reward_avgAllSteps.pth",
                                                     update,
                                                     count_steps,
                                                     lr_scheduler_pol=lr_scheduler_pol if\
                                                         (ppo_cfg.agent_type == "chat2map_activeMapper")\
                                                         else None,
                                                     best_performance=best_eval_episode_reward_allEps,
                                                     )

                        self.save_checkpoint(f"ckpt.{count_checkpoints}.pth",
                                             update,
                                             count_steps,
                                             lr_scheduler_pol=lr_scheduler_pol if\
                                                 (ppo_cfg.agent_type == "chat2map_activeMapper")\
                                                 else None,
                                             )
                        count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        """
        Evaluates a single checkpoint.
        :param checkpoint_path: path of checkpoint
        :param writer: tensorboard writer object for logging to tensorboard
        :param checkpoint_index: index of cur checkpoint for logging
        :return: a dictionary containing the results from the evaluation
        """

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        """ map location CPU is almost always better than mapping to a CUDA device. """
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        """ setting up config """
        config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        """ eval on those scenes only whose names are given in the eval config """
        if len(config.EPS_SCENES) != 0:
            full_dataset_path = config.TASK_CONFIG.DATASET.DATA_PATH.format(version=config.TASK_CONFIG.DATASET.VERSION,
                                                                            split=config.TASK_CONFIG.DATASET.SPLIT)
            with gzip.GzipFile(full_dataset_path, "rb") as fo:
                dataset = fo.read()
            dataset = dataset.decode("utf-8")
            dataset = json.loads(dataset)
            dataset_episodes = dataset["episodes"]

            eval_episode_count = 0
            for scene in config.EPS_SCENES:
                for episode in dataset_episodes:
                    if episode["scene_id"].split("/")[0] == scene:
                        eval_episode_count += 1

            if config.EVAL.EPISODE_COUNT > eval_episode_count:
                config.defrost()
                config.EVAL.EPISODE_COUNT = eval_episode_count
                config.freeze()

        logger.info(f"config: {config}")

        ppo_cfg = config.RL.PPO
        task_cfg = config.TASK_CONFIG
        env_cfg = config.TASK_CONFIG.ENVIRONMENT
        sim_cfg = config.TASK_CONFIG.SIMULATOR
        passive_mapping_cfg = config.PassiveMapping

        num_agents = sim_cfg.ALL_AGENTS.NUM
        assert num_agents == 2

        stitch_top_down_maps = config.STITCH_TOP_DOWN_MAPS_ACTIVE_MAPPING and (config.EVAL.SPLIT[:4] == "test")
        compare_against_last_top_down_maps_frame_counter = num_agents

        all_scenes_graphs_this_split = dict()
        if stitch_top_down_maps:
            split = "test"
            scenes_in_split = SCENE_SPLITS[sim_cfg.SCENE_DATASET][split]

            for scene in scenes_in_split:
                meta_dir = os.path.join(sim_cfg.AUDIO.META_DIR, scene)

                _, scene_graph = load_points_data(meta_dir,
                                                  sim_cfg.AUDIO.GRAPH_FILE,
                                                  scene_dataset=sim_cfg.SCENE_DATASET)

                all_scenes_graphs_this_split[scene] = scene_graph

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            is_train=False,
        )
        assert self.envs.num_envs == 1, "only 1 process test code implemented"

        self._setup_activeMapping_agent(eval=True)

        """ loading trained weights to policies, creating empty tensors for eval """
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.eval()
        self.agent.eval()
        self.agent.actor_critic.eval()

        """ temporary variables / objects for eval """
        not_done_masks = torch.ones(
            self.envs.num_envs, 1, device=self.device
        )
        episode_idxs = torch.FloatTensor(list(range(self.envs.num_envs))).to(device=self.device).unsqueeze(-1)
        episode_steps = torch.zeros(self.envs.num_envs, 1, device=self.device)
        test_recurrent_hidden_states_pol = torch.zeros(
            self.actor_critic.policy.num_recurrent_layers,
            self.envs.num_envs,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        test_prev_actions = torch.zeros(
            self.envs.num_envs, 1, device=self.device, dtype=torch.long
        )

        tqdm_iterator = tqdm(total=config.EVAL.EPISODE_COUNT)

        """ dict of dicts that stores stats of simulator-returned performance metrics per episode """
        stats_episodes = dict()

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        episode_rec_metrics = dict()
        for rec_metric in (passive_mapping_cfg.EvalMetrics.types + passive_mapping_cfg.TrainLosses.types):
            if rec_metric in BINARY_MAP_QUALITY_METRICS:
                if stitch_top_down_maps:
                    episode_rec_metrics["predsStitched_" + rec_metric + "_1"] = dict()
                    episode_rec_metrics["predsStitched_" + rec_metric + "_0"] = dict()
                    episode_rec_metrics["inputsStitched_" + rec_metric + "_1"] = dict()
                    episode_rec_metrics["inputsStitched_" + rec_metric + "_0"] = dict()
                else:
                    episode_rec_metrics[rec_metric + "_1"] = dict()
                    episode_rec_metrics[rec_metric + "_0"] = dict()
            else:
                if stitch_top_down_maps:
                    episode_rec_metrics["predsStitched_" + rec_metric] = dict()
                    episode_rec_metrics["inputsStitched_" + rec_metric] = dict()
                else:
                    episode_rec_metrics[rec_metric] = dict()

        """needed only for creating batch for mapper"""
        rollouts_mapper = RolloutStorageMapper(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            eval_=True,
            num_agents=num_agents,
        )

        """ resetting environments for 1st step of eval """
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        """ looping over episodes """
        while (
            len(stats_episodes) < config.EVAL.EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                step_observation = batch

                """get actions"""
                (
                    _,
                    actions,
                    _,
                    _,
                    test_recurrent_hidden_states_pol,
                    _,
                ) = self.actor_critic.act(
                    step_observation,
                    test_recurrent_hidden_states_pol,
                    test_prev_actions,
                    not_done_masks,
                    deterministic=ppo_cfg.deterministic_eval,
                )

            """first action in action space is STOP (0), increment action returned by policy by 1 to prevent stopping the 
            episode."""
            actions += 1

            if stitch_top_down_maps:
                assert actions.size(0) == 1
                if actions[0][0].item() in [2, 3]:
                    if compare_against_last_top_down_maps_frame_counter + 1 > env_cfg.VISUAL_BUDGET_COMPARE_AGAINST_LAST_TOP_DOWN_MAPS:
                        actions[...] = 1
                    else:
                        compare_against_last_top_down_maps_frame_counter += 1
                elif actions[0][0].item() == 4:
                    if compare_against_last_top_down_maps_frame_counter + 2 > env_cfg.VISUAL_BUDGET_COMPARE_AGAINST_LAST_TOP_DOWN_MAPS:
                        if compare_against_last_top_down_maps_frame_counter + 1 > env_cfg.VISUAL_BUDGET_COMPARE_AGAINST_LAST_TOP_DOWN_MAPS:
                            actions[...] = 1
                        else:
                            compare_against_last_top_down_maps_frame_counter += 1
                            actions[...] = 2
                    else:
                        compare_against_last_top_down_maps_frame_counter += 2

            outputs = self.envs.step([a[0].item() for a in actions])

            test_prev_actions = actions - 1

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            """inserting observations for current step into rollouts_mapper"""
            mapper_batch =\
                self.insert_batch_rollouts_mapper(
                    rollouts_mapper,
                    batch,
                    actions,
                    None,
                    dones,
                    eval_=True
                )

            if int(episode_steps[0].item()) == 0:
                mapper_batch_0 =\
                    self.insert_batch_rollouts_mapper(
                        rollouts_mapper,
                        batch,
                        actions,
                        None,
                        dones,
                        eval_=True,
                        eval_first_step=True,
                    )

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            mapper_step_observation, query_maps_gt, query_maps_exploredMasks = self.build_mapper_step_observation(
                mapper_batch
            )

            if int(episode_steps[0].item()) == 0:
                mapper_step_observation_0, _, _ = self.build_mapper_step_observation(
                    mapper_batch_0
                )

            if stitch_top_down_maps:
                assert "current_context_rAz" in batch
                mapper_step_observation["query_views_rAz"] = batch["current_context_rAz"].reshape(
                    batch["current_context_rAz"].size(0),
                    batch["current_context_rAz"].size(1) * batch["current_context_rAz"].size(2),
                    *batch["current_context_rAz"].size()[3:]
                ).int()

                if int(episode_steps[0].item()) == 0:
                    mapper_step_observation_0["query_views_rAz"] = mapper_step_observation["query_views_rAz"]

                assert "current_sitched_query_globCanMapEgoCrop_gt" in batch
                stitched_query_maps_gt = []
                for batch_idx, done in enumerate(dones):
                    stitched_query_maps_gt.append(batch["current_sitched_query_globCanMapEgoCrop_gt"][batch_idx][-1])
                    if done:
                        compare_against_last_top_down_maps_frame_counter = num_agents

                stitched_query_maps_gt = torch.stack(stitched_query_maps_gt, dim=0)

            with torch.no_grad():
                preds = self.actor_critic.mapper_forward(mapper_step_observation).detach()

                if int(episode_steps[0].item()) == 0:
                    preds_0 = self.actor_critic.mapper_forward(mapper_step_observation_0).detach()

            """ getting rewards """
            rewards = self.override_rewards(
                preds=preds,
                query_views_mask=mapper_step_observation["query_views_mask"],
                query_maps_gt=query_maps_gt,
                query_maps_exploredMasks=query_maps_exploredMasks,
            )

            if stitch_top_down_maps:
                assert self.envs.num_envs == 1

                scene_idx_thisEp = int(batch["episode_scene_idx"][0][0].item())
                scene_name_thisEp = SCENE_IDX_TO_NAME[sim_cfg.SCENE_DATASET][scene_idx_thisEp]
                assert scene_name_thisEp in all_scenes_graphs_this_split
                graph_thisEp = all_scenes_graphs_this_split[scene_name_thisEp]

                ref_rAz_thisEp = batch['episode_ref_rAz'][0].int().tolist()
                query_views_rAz_thisEp = mapper_step_observation["query_views_rAz"][0]
                query_views_mask_thisEp = mapper_step_observation["query_views_mask"][0]

                query_stitchedPredGlobCanOccMapsEgoCrops_ph =\
                    np.zeros((
                        task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                        task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                        1)).astype("float32")

                query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph =\
                    np.zeros((
                        task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                        task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                        1)).astype("float32")

                preds_thisEp = preds[0].cpu().numpy()

                if int(episode_steps[0].item()) == 0:
                    query_views_rAz_thisEp_0 = mapper_step_observation["query_views_rAz"][0]
                    query_views_mask_thisEp_0 = mapper_step_observation["query_views_mask"][0]

                    query_stitchedPredGlobCanOccMapsEgoCrops_ph_0 =\
                        np.zeros((
                            task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                            task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                            1)).astype("float32")

                    query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0 =\
                        np.zeros((
                            task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                            task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                            1)).astype("float32")

                    preds_thisEp_0 = preds_0[0].cpu().numpy()

                query_maps_gt = stitched_query_maps_gt

                for query_view_idx in range(query_views_mask_thisEp.size(0)):
                    query_view_mask_val = query_views_mask_thisEp[query_view_idx].item()
                    assert query_view_mask_val in [0, 1]
                    if query_view_mask_val == 1:
                        query_view_rAz = query_views_rAz_thisEp[query_view_idx].tolist()

                        query_stitchedPredGlobCanOccMapsEgoCrops_ph, query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph =\
                            get_stitched_top_down_maps(stitched_map=query_stitchedPredGlobCanOccMapsEgoCrops_ph,
                                                       stitch_component=preds_thisEp[query_view_idx],
                                                       ref_pose=ref_rAz_thisEp,
                                                       target_pose=query_view_rAz,
                                                       graph=graph_thisEp,
                                                       is_occupancy=True,
                                                       is_pred=True,
                                                       is_ego_360deg_crops=True,
                                                       stitched_map_updateCounter=query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph,
                                                       map_scale=sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE,
                                                       )

                query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph =\
                    (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph == 0.).astype("float32") + \
                    (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph != 0.).astype("float32") *\
                    query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph
                query_stitchedPredGlobCanOccMapsEgoCrops_ph = query_stitchedPredGlobCanOccMapsEgoCrops_ph\
                                                              / (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph + 1.0e-13)
                assert np.prod((query_stitchedPredGlobCanOccMapsEgoCrops_ph >= 0).astype("float32") *\
                                  (query_stitchedPredGlobCanOccMapsEgoCrops_ph <= 1.).astype("float32")).item() == 1,\
                    print(query_stitchedPredGlobCanOccMapsEgoCrops_ph.max(), query_stitchedPredGlobCanOccMapsEgoCrops_ph.min())
                query_stitchedPredGlobCanOccMapsEgoCrops_ph = (query_stitchedPredGlobCanOccMapsEgoCrops_ph > 0.5).astype("float32")

                preds = torch.from_numpy(query_stitchedPredGlobCanOccMapsEgoCrops_ph).to(self.device).unsqueeze(0)

                allAgents_context_stitchedEgoLocalOccMaps_ph =\
                    np.zeros((
                        task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                        task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                        task_cfg.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS)).astype("float32")

                context_views_mask_thisEp =\
                    mapper_batch["context_views_mask"][0].reshape(mapper_batch["context_views_mask"][0].size(0) *\
                                                               mapper_batch["context_views_mask"][0].size(1))
                context_maps_thisEp = mapper_batch["context_maps"][0].reshape(
                    mapper_batch["context_maps"].size(1) * mapper_batch["context_maps"].size(2),
                    *mapper_batch["context_maps"].size()[3:]
                ).cpu().numpy()
                for context_view_idx in range(context_views_mask_thisEp.size(0)):
                    context_view_mask_val = context_views_mask_thisEp[context_view_idx].item()
                    assert context_view_mask_val in [0, 1]
                    if context_view_mask_val == 1:
                        context_view_rAz = query_views_rAz_thisEp[context_view_idx].tolist()

                        allAgents_context_stitchedEgoLocalOccMaps_ph =\
                            get_stitched_top_down_maps(stitched_map=allAgents_context_stitchedEgoLocalOccMaps_ph,
                                                       stitch_component=context_maps_thisEp[context_view_idx],
                                                       ref_pose=ref_rAz_thisEp,
                                                       target_pose=context_view_rAz,
                                                       graph=graph_thisEp,
                                                       num_channels=task_cfg.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS,
                                                       map_scale=sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE,
                                                       )

                inputs_stitched = torch.from_numpy(allAgents_context_stitchedEgoLocalOccMaps_ph).to(self.device).unsqueeze(0)

                preds = preds * (inputs_stitched[..., 1:] != 1.0).float() + inputs_stitched[..., :1] * (inputs_stitched[..., 1:] == 1.0).float()

                if int(episode_steps[0].item()) == 0:
                    for query_view_idx in range(query_views_mask_thisEp_0.size(0)):
                        query_view_mask_val_0 = query_views_mask_thisEp_0[query_view_idx].item()
                        assert query_view_mask_val_0 in [0, 1]
                        if query_view_mask_val_0 == 1:
                            query_view_rAz_0 = query_views_rAz_thisEp_0[query_view_idx].tolist()

                            query_stitchedPredGlobCanOccMapsEgoCrops_ph_0, query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0 =\
                                get_stitched_top_down_maps(stitched_map=query_stitchedPredGlobCanOccMapsEgoCrops_ph_0,
                                                           stitch_component=preds_thisEp_0[query_view_idx],
                                                           ref_pose=ref_rAz_thisEp,
                                                           target_pose=query_view_rAz_0,
                                                           graph=graph_thisEp,
                                                           is_occupancy=True,
                                                           is_pred=True,
                                                           is_ego_360deg_crops=True,
                                                           stitched_map_updateCounter=query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0,
                                                           map_scale=sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE,
                                                           )

                    query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0 =\
                        (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0 == 0.).astype("float32") + \
                        (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0 != 0.).astype("float32") *\
                        query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0
                    query_stitchedPredGlobCanOccMapsEgoCrops_ph_0 = query_stitchedPredGlobCanOccMapsEgoCrops_ph_0\
                                                                  / (query_stitchedPredGlobCanOccMapsEgoCrops_validUpdateCount_ph_0 + 1.0e-13)
                    assert np.prod((query_stitchedPredGlobCanOccMapsEgoCrops_ph_0 >= 0).astype("float32") *\
                                      (query_stitchedPredGlobCanOccMapsEgoCrops_ph_0 <= 1.).astype("float32")).item() == 1,\
                        print(query_stitchedPredGlobCanOccMapsEgoCrops_ph_0.max(), query_stitchedPredGlobCanOccMapsEgoCrops_ph_0.min())
                    query_stitchedPredGlobCanOccMapsEgoCrops_ph_0 = (query_stitchedPredGlobCanOccMapsEgoCrops_ph_0 > 0.5).astype("float32")

                    preds_0 = torch.from_numpy(query_stitchedPredGlobCanOccMapsEgoCrops_ph_0).to(self.device).unsqueeze(0)

                    allAgents_context_stitchedEgoLocalOccMaps_ph_0 =\
                        np.zeros((
                            task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                            task_cfg.TASK.CONTEXT_STITCHED_EGO_LOCAL_MAP_SENSOR.SIZE,
                            task_cfg.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS)).astype("float32")

                    context_views_mask_thisEp_0 =\
                        mapper_batch_0["context_views_mask"][0].reshape(mapper_batch_0["context_views_mask"][0].size(0) *\
                                                                   mapper_batch_0["context_views_mask"][0].size(1))
                    context_maps_thisEp_0 = mapper_batch_0["context_maps"][0].reshape(
                        mapper_batch_0["context_maps"].size(1) * mapper_batch_0["context_maps"].size(2),
                        *mapper_batch_0["context_maps"].size()[3:]
                    ).cpu().numpy()
                    for context_view_idx in range(context_views_mask_thisEp_0.size(0)):
                        context_view_mask_val = context_views_mask_thisEp_0[context_view_idx].item()
                        assert context_view_mask_val in [0, 1]
                        if context_view_mask_val == 1:
                            context_view_rAz = query_views_rAz_thisEp_0[context_view_idx].tolist()

                            allAgents_context_stitchedEgoLocalOccMaps_ph_0 =\
                                get_stitched_top_down_maps(stitched_map=allAgents_context_stitchedEgoLocalOccMaps_ph_0,
                                                           stitch_component=context_maps_thisEp_0[context_view_idx],
                                                           ref_pose=ref_rAz_thisEp,
                                                           target_pose=context_view_rAz,
                                                           graph=graph_thisEp,
                                                           num_channels=task_cfg.TASK.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS,
                                                           map_scale=sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP.SCALE,
                                                           )

                    inputs_stitched_0 = torch.from_numpy(allAgents_context_stitchedEgoLocalOccMaps_ph_0).to(self.device).unsqueeze(0)

                    preds_0 = preds_0 * (inputs_stitched_0[..., 1:] != 1.0).float() + inputs_stitched_0[..., :1] * (inputs_stitched_0[..., 1:] == 1.0).float()

            batch = batch_obs(observations, self.device)

            episode_rec_metrics_thisStep = dict()
            if int(episode_steps[0].item()) == 0:
                episode_rec_metrics_thisStep_0 = dict()

            for metric_name in episode_rec_metrics:
                if stitch_top_down_maps:
                    exploredPart_mask = None
                    if metric_name in ["inputsStitched_f1_score_1",
                                       "inputsStitched_f1_score_0",
                                       "inputsStitched_iou_1",
                                       "inputsStitched_iou_0",
                                       "predsStitched_f1_score_1",
                                       "predsStitched_f1_score_0",
                                       "predsStitched_iou_1",
                                       "predsStitched_iou_0",
                                       ]:
                        if metric_name in ["inputsStitched_f1_score_1", "inputsStitched_f1_score_0",
                                           "predsStitched_f1_score_1", "predsStitched_f1_score_0"]:
                            metric_type = "f1_score"
                        elif metric_name in ["inputsStitched_iou_1", "inputsStitched_iou_0",
                                             "predsStitched_iou_1", "predsStitched_iou_0"]:
                            metric_type = "iou"
                        else:
                            raise ValueError

                        if metric_name in ["inputsStitched_f1_score_1", "inputsStitched_f1_score_0",
                                           "inputsStitched_iou_1", "inputsStitched_iou_0"]:
                            pred_occMap = (inputs_stitched[..., :1] > 0.5).float()
                        elif metric_name in ["predsStitched_f1_score_1", "predsStitched_f1_score_0",
                                           "predsStitched_iou_1", "predsStitched_iou_0"]:
                            pred_occMap = (preds[..., :1] > 0.5).float()

                        if int(episode_steps[0].item()) == 0:
                            if metric_name in ["inputsStitched_f1_score_1", "inputsStitched_f1_score_0",
                                               "inputsStitched_iou_1", "inputsStitched_iou_0"]:
                                pred_occMap_0 = (inputs_stitched_0[..., :1] > 0.5).float()
                            elif metric_name in ["predsStitched_f1_score_1", "predsStitched_f1_score_0",
                                               "predsStitched_iou_1", "predsStitched_iou_0"]:
                                pred_occMap_0 = (preds_0[..., :1] > 0.5).float()     

                        if metric_name.split("_")[-1] == "1":
                            target_category = 1.0
                        elif metric_name.split("_")[-1] == "0":
                            target_category = 0.0
                        else:
                            raise ValueError
                    else:
                        raise NotImplementedError
                else:
                    if metric_name in ["f1_score_1", "f1_score_0", "iou_1", "iou_0"]:
                        if metric_name in ["f1_score_1", "f1_score_0"]:
                            metric_type = "f1_score"
                        elif metric_name in ["iou_1", "iou_0"]:
                            metric_type = "iou"
                        else:
                            raise ValueError

                        pred_occMap = (preds[..., :1] > 0.5).float()
                        exploredPart_mask = query_maps_exploredMasks[..., :1]

                        if metric_name.split("_")[-1] == "1":
                            target_category = 1.0
                        elif metric_name.split("_")[-1] == "0":
                            target_category = 0.0
                        else:
                            raise ValueError
                    elif metric_name.split("_")[-1] == "loss":
                        metric_type = metric_name
                        pred_occMap = preds[..., :1]
                        exploredPart_mask = None
                        target_category = None

                gt_occMap = query_maps_gt[..., :1]

                if stitch_top_down_maps:
                    query_mask = torch.ones(mapper_step_observation["query_views_mask"].size(0)).to(self.device)

                    if int(episode_steps[0].item()) == 0:
                        query_mask_0 = torch.ones(mapper_step_observation_0["query_views_mask"].size(0)).to(self.device)
                else:
                    query_mask = mapper_step_observation["query_views_mask"]

                map_metric_all_batch_idxs = compute_loss_n_evalMetrics(
                    loss_or_metric_types=[metric_type],
                    loss_or_metric_weights=[1.0],
                    gt_occMap=gt_occMap,
                    pred_occMap=pred_occMap,
                    mask=query_mask,
                    exploredPart_mask=exploredPart_mask,
                    target_category=target_category,
                    is_stitched=stitch_top_down_maps,
                    eval_mode=True,
                )
                episode_rec_metrics_thisStep[metric_name] = map_metric_all_batch_idxs

                if int(episode_steps[0].item()) == 0:
                    map_metric_all_batch_idxs_0 = compute_loss_n_evalMetrics(
                        loss_or_metric_types=[metric_type],
                        loss_or_metric_weights=[1.0],
                        gt_occMap=gt_occMap,
                        pred_occMap=pred_occMap_0,
                        mask=query_mask_0,
                        exploredPart_mask=exploredPart_mask,
                        target_category=target_category,
                        is_stitched=stitch_top_down_maps,
                        eval_mode=True,
                    )
                    episode_rec_metrics_thisStep_0[metric_name] = map_metric_all_batch_idxs_0

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards

            episode_steps += 1

            next_episodes = self.envs.current_episodes()

            envs_to_pause = []
            for i in range(self.envs.num_envs):
                """ pause envs which runs out of episodes """
                if (
                    next_episodes[i].scene_id.split("/")[-2],
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                scene_ep_id = (
                    current_episodes[i].scene_id.split("/")[-2],
                    current_episodes[i].episode_id
                )
                for metric_name in episode_rec_metrics:
                    assert metric_name in episode_rec_metrics_thisStep

                    if scene_ep_id not in episode_rec_metrics[metric_name]:
                        episode_rec_metrics[metric_name][scene_ep_id] = [episode_rec_metrics_thisStep_0[metric_name][i],
                                                                         episode_rec_metrics_thisStep[metric_name][i]]
                    else:
                        episode_rec_metrics[metric_name][scene_ep_id].append(episode_rec_metrics_thisStep[metric_name][i])

                """ episode ended """
                if not_done_masks[i].item() == 0:
                    episode_stats = dict()

                    episode_stats["reward"] = current_episode_reward[i].item()

                    """ use scene_id + episode_id as unique id for storing stats """
                    stats_episodes[
                        (
                            current_episodes[i].scene_id.split("/")[-2],
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    current_episode_reward[i] = 0
                    episode_steps[i] = 0
                    tqdm_iterator.update()

            (
                self.envs,
                test_recurrent_hidden_states_pol,
                not_done_masks,
                current_episode_reward,
                batch,
                episode_steps,
                episode_idxs,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states_pol,
                not_done_masks,
                current_episode_reward,
                batch,
                episode_steps,
                episode_idxs,
            )

        """ closing the open environments after iterating over all episodes """
        self.envs.close()

        """ mean and std of simulator-returned metrics and rewards """
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = dict()
            aggregated_stats[stat_key]["mean"] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )
            aggregated_stats[stat_key]["std"] = np.std(
                [v[stat_key] for v in stats_episodes.values()]
            )

        if len(episode_rec_metrics) > 0:
            with open(os.path.join(config.MODEL_DIR, "eval_metrics.pkl"), "wb") as fo:
                pickle.dump(episode_rec_metrics, fo, protocol=pickle.HIGHEST_PROTOCOL)

            if stitch_top_down_maps:
                for metric_name in episode_rec_metrics:
                    lst_all_metric_val = []
                    lst_last_metric_val = []
                    for scene_ep_id in episode_rec_metrics[metric_name]:
                        lst_all_metric_val += episode_rec_metrics[metric_name][scene_ep_id]
                        lst_last_metric_val.append(episode_rec_metrics[metric_name][scene_ep_id][-1])

                    aggregated_stats["all_" + metric_name] = {"mean": np.mean(lst_all_metric_val),
                                                              "std": np.std(lst_all_metric_val)}
                    aggregated_stats["last_" + metric_name] = {"mean": np.mean(lst_last_metric_val),
                                                               "std": np.std(lst_last_metric_val)}

        """ dump stats file to disk """
        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT,
                                                            config.SEED)
                                  )
        new_stats_episodes = {','.join(key): value for key, value in stats_episodes.items()}
        with open(stats_file, 'w') as fo:
            json.dump(new_stats_episodes, fo)

        """ write eval metrics to train.log, terminal and tb """
        result = {}
        episode_metrics_mean = {}
        episode_metrics_std = {}

        for metric_uuid in aggregated_stats.keys():
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid]["mean"]
            episode_metrics_std[metric_uuid] = aggregated_stats[metric_uuid]["std"]
            result['episode_{}_mean'.format(metric_uuid)] = aggregated_stats[metric_uuid]["mean"]
            result['episode_{}_std'.format(metric_uuid)] = aggregated_stats[metric_uuid]["std"]

        for metric_uuid in episode_metrics_mean.keys():
            logger.info(
                f"Average episode {metric_uuid}: mean -- {episode_metrics_mean[metric_uuid]:.6f}, "
                f"std -- {episode_metrics_std[metric_uuid]:.6f}"
            )
            writer.add_scalar(
                f"{metric_uuid}/{config.EVAL.SPLIT}/mean",
                episode_metrics_mean[metric_uuid],
                checkpoint_index,
            )

            writer.add_scalar(
                f"{metric_uuid}/{config.EVAL.SPLIT}/std",
                episode_metrics_std[metric_uuid],
                checkpoint_index,
            )

        return result
