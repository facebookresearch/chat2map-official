# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


class PoseEnc(nn.Module):
    """
    Takes in pose attributes and produces and produces their embeddings
    """

    def __init__(self, observation_space, config):
        super().__init__()

        self.config = config

        ppo_cfg = config.RL.PPO
        self.pose_enc_cfg = ppo_cfg.PoseEnc

        assert "current_context_pose" in observation_space.spaces
        self._n_positional_obs = observation_space.spaces["current_context_pose"].shape[-1]

        if self._n_positional_obs == 3:
            self._n_positional_obs += 2
        else:
            pass

        self.pose_enc = nn.Sequential(
            nn.Linear(self._n_positional_obs, self.pose_enc_cfg.output_size, bias=False),
        )

    @property
    def n_out_feats(self):
        return self.pose_enc_cfg.output_size

    def forward(self, observation):
        """Given pose, computes their embeddings"""
        pose = observation["pose"]

        return self.pose_enc(pose)
