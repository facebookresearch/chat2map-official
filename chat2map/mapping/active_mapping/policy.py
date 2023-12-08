# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch
import torch.nn as nn

from chat2map.mapping.passive_mapping.policy import PassiveMappingPolicy

from chat2map.mapping.policy_models.pose_enc import PoseEnc
from chat2map.mapping.policy_models.audio_cnn import AudioEnc
from chat2map.mapping.policy_models.visual_cnn import VisualEnc
from chat2map.mapping.policy_models.fuse_net import FuseNet
from chat2map.mapping.policy_models.rnn_state_encoder import RNNStateEncoder

from chat2map.common.utils import CategoricalNet


class CriticHead(nn.Module):
    """
    Critic network
    """

    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

    def forward(self,
                observations,
                rnn_hidden_states,
                prev_actions,
                masks):
        pass


class PolicyNet(Net):
    """Network which passes the observation through CNNs and concatenates
        them into a single vector before passing that through RNN.
    """

    def __init__(self, config, observation_space,):
        super().__init__()

        self.config = config
        self.task_cfg = config.TASK_CONFIG.TASK
        self.env_cfg = config.TASK_CONFIG.ENVIRONMENT

        self.sim_cfg = config.TASK_CONFIG.SIMULATOR
        self.num_agents = self.sim_cfg.ALL_AGENTS.NUM
        assert self.num_agents == 2

        self.ppo_cfg = config.RL.PPO
        self.pose_enc_cfg = self.ppo_cfg.PoseEnc
        self.audio_enc_cfg = self.ppo_cfg.AudioEnc
        self.action_enc_cfg = self.ppo_cfg.ActionEnc
        self.fuse_net_cfg = self.ppo_cfg.FuseNet
        self.visual_enc_cfg = self.ppo_cfg.VisualEnc

        self.visual_encoder = VisualEnc(
            config,
        )

        self.audio_encoder = AudioEnc(
            config,
        )

        self.pose_encoder = PoseEnc(
            observation_space,
            config,
        )

        self.fuse_vision_n_pose_net = FuseNet(
            in_n_out_sizes=[
                self.visual_encoder.n_out_feats + self.pose_encoder.n_out_feats,
                self.fuse_net_cfg.output_size
            ],
        )

        self.fuse_audio_n_pose_net = FuseNet(
            in_n_out_sizes=[
                self.audio_encoder.n_out_feats + self.pose_encoder.n_out_feats,
                self.fuse_net_cfg.output_size
            ],
        )

        self.fuse_prev_n_curr_selfAudio_net = FuseNet(
            in_n_out_sizes=[
                2 * self.fuse_audio_n_pose_net.n_out_feats,
                self.fuse_net_cfg.output_size
            ],
        )

        self.fuse_prev_n_curr_otherAudio_net = FuseNet(
            in_n_out_sizes=[
                2 * self.fuse_audio_n_pose_net.n_out_feats,
                self.fuse_net_cfg.output_size
            ],
        )

        self.fuse_prev_n_curr_selfPose_net = FuseNet(
            in_n_out_sizes=[
                2 * self.pose_encoder.n_out_feats,
                self.fuse_net_cfg.output_size
            ],
        )

        self.action_encoder = FuseNet(
            in_n_out_sizes=[
                1,
                self.action_enc_cfg.output_size
            ],
        )

        self.hidden_size = self.ppo_cfg.hidden_size
        rnn_input_size = self.num_agents * (
                self.fuse_vision_n_pose_net.n_out_feats +\
                self.fuse_prev_n_curr_selfAudio_net.n_out_feats +\
                self.fuse_prev_n_curr_otherAudio_net.n_out_feats +\
                self.fuse_prev_n_curr_selfPose_net.n_out_feats
        )\
                         + self.action_encoder.n_out_feats
        rnn_num_layers = self.ppo_cfg.rnn_num_layers

        self.state_encoder = RNNStateEncoder(rnn_input_size, self.hidden_size, rnn_num_layers)

    @property
    def is_blind(self):
        return False

    @property
    def output_size(self):
        return self.hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self,
                observations,
                rnn_hidden_states,
                prev_actions,
                masks):
        """computes policy features"""
        bs = observations["prevNcurr_allAgnts_selfPose"].size(0)

        lst_rnn_feats = []
        for agent_idx in range(self.num_agents):
            prev_selfPose = observations["prevNcurr_allAgnts_selfPose"][:, 0, agent_idx]
            prev_selfPose_feats = self.pose_encoder({"pose": prev_selfPose})

            prev_view_mask = observations["prev_allAgnts_view_mask"][:, agent_idx]
            prev_map = observations["prev_allAgnts_map"][:, agent_idx]
            prev_rgb = observations["prev_allAgnts_rgb"][:, agent_idx]

            prev_view_feats = self.visual_encoder(
                {"map": prev_map,
                 "rgb": prev_rgb}
            )
            prev_view_feats = prev_view_feats * prev_view_mask.unsqueeze(-1)
            fused_prev_view_n_pose_feats = self.fuse_vision_n_pose_net(
                {"feat1": prev_view_feats,
                 "feat2": prev_selfPose_feats}
            )

            lst_rnn_feats.append(fused_prev_view_n_pose_feats)

            prev_selfAudio_mask = observations["prevNcurr_allAgnts_selfAudio_mask"][:, 0, agent_idx]
            prev_selfAudio = observations["prevNcurr_allAgnts_selfAudio"][:, 0, agent_idx]
            prev_selfAudio_feats = self.audio_encoder(
                {"audio": prev_selfAudio}
            )
            prev_selfAudio_feats = prev_selfAudio_feats * prev_selfAudio_mask.unsqueeze(-1)
            fused_prev_selfAudio_n_pose_feats = self.fuse_audio_n_pose_net(
                {'feat1': prev_selfAudio_feats,
                 'feat2': prev_selfPose_feats,}
            )

            curr_selfPose = observations["prevNcurr_allAgnts_selfPose"][:, 1, agent_idx]
            curr_selfAudio_mask = observations["prevNcurr_allAgnts_selfAudio_mask"][:, 1, agent_idx]
            curr_selfAudio = observations["prevNcurr_allAgnts_selfAudio"][:, 1, agent_idx]

            curr_selfPose_feats = self.pose_encoder({"pose": curr_selfPose})

            curr_selfAudio_feats = self.audio_encoder(
                {"audio": curr_selfAudio}
            )
            curr_selfAudio_feats = curr_selfAudio_feats * curr_selfAudio_mask.unsqueeze(-1)
            fused_curr_selfAudio_n_pose_feats = self.fuse_audio_n_pose_net(
                {
                    "feat1": curr_selfAudio_feats,
                    "feat2": curr_selfPose_feats,
                }
            )

            fused_prev_n_curr_selfAudio_feats = self.fuse_prev_n_curr_selfAudio_net(
                {"feat1": fused_prev_selfAudio_n_pose_feats,
                 "feat2": fused_curr_selfAudio_n_pose_feats}
            )
            lst_rnn_feats.append(fused_prev_n_curr_selfAudio_feats)

            prev_otherPose = observations["prevNcurr_allAgnts_otherPose"][:, 0, agent_idx]
            prev_otherAudio_mask = observations["prevNcurr_allAgnts_otherAudio_mask"][:, 0, agent_idx]
            prev_otherAudio = observations["prevNcurr_allAgnts_otherAudio"][:, 0, agent_idx]

            prev_otherPose_feats = self.pose_encoder({"pose": prev_otherPose})
            prev_otherAudio_feats = self.audio_encoder(
                {"audio": prev_otherAudio}
            )
            prev_otherAudio_feats = prev_otherAudio_feats * prev_otherAudio_mask.unsqueeze(-1)
            fused_prev_otherAudio_n_pose_feats = self.fuse_audio_n_pose_net(
                {'feat1': prev_otherAudio_feats,
                 'feat2': prev_otherPose_feats,}
            )

            curr_otherPose = observations["prevNcurr_allAgnts_otherPose"][:, 1, agent_idx]
            curr_otherAudio_mask = observations["prevNcurr_allAgnts_otherAudio_mask"][:, 1, agent_idx]
            curr_otherAudio = observations["prevNcurr_allAgnts_otherAudio"][:, 1, agent_idx]

            curr_otherPose_feats = self.pose_encoder({"pose": curr_otherPose})
            curr_otherAudio_feats = self.audio_encoder(
                {"audio": curr_otherAudio}
            )
            curr_otherAudio_feats = curr_otherAudio_feats * curr_otherAudio_mask.unsqueeze(-1)
            fused_curr_otherAudio_n_pose_feats = self.fuse_audio_n_pose_net(
                {'feat1': curr_otherAudio_feats,
                 'feat2': curr_otherPose_feats,}
            )

            fused_prev_n_curr_otherAudio_feats = self.fuse_prev_n_curr_otherAudio_net(
                {"feat1": fused_prev_otherAudio_n_pose_feats,
                 "feat2": fused_curr_otherAudio_n_pose_feats}
            )
            lst_rnn_feats.append(fused_prev_n_curr_otherAudio_feats)

            fused_prev_n_curr_selfPose_feats = self.fuse_prev_n_curr_selfPose_net(
                {"feat1": prev_selfPose_feats,
                 "feat2": curr_selfPose_feats,}
            )
            lst_rnn_feats.append(fused_prev_n_curr_selfPose_feats)

        prev_action_inps = (prev_actions * masks).float()

        prev_action_feats = self.action_encoder(
            {"feat1": prev_action_inps,
             "feat2": torch.tensor([]).to(prev_actions.device)}
        )
        lst_rnn_feats.append(prev_action_feats)

        try:
            rnn_feats = torch.cat(lst_rnn_feats, dim=1)
        except AssertionError as error:
            for data in lst_rnn_feats:
                print(data.size())

        try:
            rnn_feats_new, rnn_hidden_states_new = self.state_encoder(rnn_feats, rnn_hidden_states, masks)
        except AssertionError as error:
            print(rnn_feats.size(), rnn_hidden_states.size(), masks.size(), rnn_feats_new.size(),
                  rnn_hidden_states_new.size())

        assert not torch.isnan(rnn_feats_new).any().item()

        return rnn_feats_new, rnn_hidden_states_new


class Policy(nn.Module):
    """
    Parent class of chat2map active mapper policy
    """

    def __init__(
            self,
            mapper,
            policy,
            dim_actions
    ):
        """Creates an instance of the parent class of chat2map active mapper policy"""
        super().__init__()

        self.visual_enc_cfg = self.ppo_cfg.VisualEnc

        self.num_agents = self.sim_cfg.ALL_AGENTS.NUM

        self.mapper = mapper
        self.policy = policy
        self.dim_actions = dim_actions

        if self.ppo_cfg.agent_type == "chat2map_activeMapper":
            self.action_dist = CategoricalNet(
                self.policy.output_size, self.dim_actions
            )
            self.critic = CriticHead(self.policy.output_size)

    def mapper_forward(self, obs_batch):
        """computes estimate of the target maps given the observations"""
        preds = self.mapper(obs_batch)
        return preds

    def forward(self):
        raise NotImplementedError

    def build_policy_observation(self,
                                 observation,
                                 ):
        """builds policy observation"""
        policy_observation = dict()

        bs = observation["current_context_idx"].size(0)

        lst_current_context_idx = observation["current_context_idx"][..., 0].int().tolist()
        lst_previous_context_idx = (observation["current_context_idx"][..., 0] - 1).int().tolist()
        bs_rng_lst = list(range(bs))

        """initially - B x num_agents x num_steps x ..."""
        """pose inputs"""
        selfPose = observation["current_context_pose"].permute(0, 2, 1, 3)
        current_selfPose = selfPose[bs_rng_lst, lst_current_context_idx]
        previous_selfPose = selfPose[bs_rng_lst, lst_previous_context_idx]
        prevNcurr_allAgnts_selfPose = torch.stack([previous_selfPose, current_selfPose], dim=1)
        policy_observation["prevNcurr_allAgnts_selfPose"] = prevNcurr_allAgnts_selfPose

        """view inputs"""
        view_mask = observation["previous_context_view_mask"].permute(0, 2, 1)
        prev_allAgnts_view_mask = view_mask[bs_rng_lst, lst_previous_context_idx]
        policy_observation["prev_allAgnts_view_mask"] = prev_allAgnts_view_mask

        map = observation["current_context_map"].permute(0, 2, 1, 3, 4, 5)
        prev_allAgnts_map = map[bs_rng_lst, lst_previous_context_idx]
        policy_observation["prev_allAgnts_map"] = prev_allAgnts_map

        rgb = observation["current_context_rgb"].permute(0, 2, 1, 3, 4, 5)
        prev_allAgnts_rgb = rgb[bs_rng_lst, lst_previous_context_idx]
        policy_observation["prev_allAgnts_rgb"] = prev_allAgnts_rgb

        """self audio inputs"""
        selfAudio_mask = observation["current_context_audio_mask"].permute(0, 2, 1)
        current_allAgnts_selfAudio_mask = selfAudio_mask[bs_rng_lst, lst_current_context_idx]
        previous_allAgnts_selfAudio_mask = selfAudio_mask[bs_rng_lst, lst_previous_context_idx]
        prevNcurr_allAgnts_selfAudio_mask = torch.stack([previous_allAgnts_selfAudio_mask,
                                                        current_allAgnts_selfAudio_mask],
                                                       dim=1)
        policy_observation["prevNcurr_allAgnts_selfAudio_mask"] = prevNcurr_allAgnts_selfAudio_mask

        selfAudio = observation["current_context_selfAudio"].permute(0, 2, 1, 3, 4)
        current_selfAudio = selfAudio[bs_rng_lst, lst_current_context_idx]
        previous_selfAudio = selfAudio[bs_rng_lst, lst_previous_context_idx]
        prevNcurr_allAgnts_selfAudio = torch.stack([previous_selfAudio, current_selfAudio], dim=1)
        policy_observation["prevNcurr_allAgnts_selfAudio"] = prevNcurr_allAgnts_selfAudio

        """audio from other ego inputs"""
        otherPose = observation["current_context_otherAudio_pose"].permute(0, 2, 1, 3)
        current_otherPose = otherPose[bs_rng_lst, lst_current_context_idx]
        previous_otherPose = otherPose[bs_rng_lst, lst_previous_context_idx]
        prevNcurr_allAgnts_otherPose = torch.stack([previous_otherPose, current_otherPose], dim=1)
        policy_observation["prevNcurr_allAgnts_otherPose"] = prevNcurr_allAgnts_otherPose

        otherAudio_mask = observation["current_context_audio_mask"][:, [1, 0], :].permute(0, 2, 1)
        current_allAgnts_otherAudio_mask = otherAudio_mask[bs_rng_lst, lst_current_context_idx]
        previous_allAgnts_otherAudio_mask = otherAudio_mask[bs_rng_lst, lst_previous_context_idx]
        prevNcurr_allAgnts_otherAudio_mask = torch.stack([previous_allAgnts_otherAudio_mask,
                                                       current_allAgnts_otherAudio_mask],
                                                      dim=1)
        policy_observation["prevNcurr_allAgnts_otherAudio_mask"] = prevNcurr_allAgnts_otherAudio_mask

        otherAudio = observation["current_context_otherAudio"].permute(0, 2, 1, 3, 4)
        current_otherAudio = otherAudio[bs_rng_lst, lst_current_context_idx]
        previous_otherAudio = otherAudio[bs_rng_lst, lst_previous_context_idx]
        prevNcurr_allAgnts_otherAudio = torch.stack([previous_otherAudio, current_otherAudio], dim=1)
        policy_observation["prevNcurr_allAgnts_otherAudio"] = prevNcurr_allAgnts_otherAudio

        return policy_observation

    def act(
        self,
        observation,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        """predicts actions"""
        policy_observation = self.build_policy_observation(
            observation,
        )

        feats, rnn_hidden_states = self.policy(
            policy_observation,
            rnn_hidden_states,
            prev_actions,
            masks,
        )

        dist = self.action_dist(feats)
        value = self.critic(feats)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        dist_probs = dist.get_probs()

        return value, action, action_log_probs, dist_entropy, rnn_hidden_states, dist_probs

    def get_value(
            self,
            observation,
            rnn_hidden_states,
            prev_actions,
            masks,
    ):
        """gets state value"""
        policy_observation = self.build_policy_observation(
            observation,
        )

        feats, _ = self.policy(
            policy_observation,
            rnn_hidden_states,
            prev_actions,
            masks,
        )

        return self.critic(feats)

    def evaluate_actions(
            self,
            observation,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
    ):
        """evaluates predicted actions"""
        policy_observation = self.build_policy_observation(
            observation,
        )

        feats, rnn_hidden_states = self.policy(
            policy_observation,
            rnn_hidden_states,
            prev_actions,
            masks,
        )

        dist = self.action_dist(feats)
        value = self.critic(feats)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hidden_states


class ActiveMappingPolicy(Policy):
    """
    Chat2map active mapper policy
    """

    def __init__(
            self,
            config,
            observation_space,
    ):
        """Creates an instance of the chat2map active mapper policy"""
        self.config = config
        self.passive_mapping_cfg = self.config.PassiveMapping
        self.ppo_cfg = self.config.RL.PPO
        self.task_cfg = self.config.TASK_CONFIG.TASK
        self.sim_cfg = self.config.TASK_CONFIG.SIMULATOR
        self.env_cfg = self.config.TASK_CONFIG.ENVIRONMENT

        self.num_agents = self.sim_cfg.ALL_AGENTS.NUM
        assert self.num_agents == 2
        dim_actions = 4

        mapper = self.build_mapper()

        policy = nn.Sequential()
        if self.config.RL.PPO.agent_type == "chat2map_activeMapper":
            policy = PolicyNet(
                config,
                observation_space
            )

        super().__init__(
            mapper,
            policy,
            dim_actions
        )

    def build_mapper(self):
        """builds mapper network"""
        mapper_policy_className = PassiveMappingPolicy

        mapper = mapper_policy_className(
            cfg=self.config,
        )

        return mapper
