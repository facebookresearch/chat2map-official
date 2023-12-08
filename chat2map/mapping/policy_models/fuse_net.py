# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


class FuseNet(nn.Module):
    """Net to fuse encoded features in the chat2map policy"""

    def __init__(self, in_n_out_sizes=[]):
        super().__init__()

        self.output_size = in_n_out_sizes[1]

        self.fuse_net = nn.Sequential(
            nn.Linear(in_n_out_sizes[0], in_n_out_sizes[1]),
        )

    @property
    def is_blind(self):
        return False

    @property
    def n_out_feats(self):
        return self.output_size

    def forward(self, observation):
        """Given two 1D features, fuses them into a single feature using a linear layer"""

        net_input = []

        assert "feat1" in observation
        feat1 = observation["feat1"]
        net_input.append(feat1)

        assert "feat2" in observation
        feat2 = observation["feat2"]
        net_input.append(feat2)

        net_input = torch.cat(net_input, dim=1)

        fused_feats = self.fuse_net(net_input)

        return fused_feats
