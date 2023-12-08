# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


class FusionNet(nn.Module):
    """Network to fuse modality features, positional embeddings and modality type tag embeddings"""

    def __init__(self,):
        super().__init__()

    def forward(self, observations):
        """fuses given different features"""
        for observation_idx, observation in enumerate(observations):
            if observation_idx == 0:
                out = observation
            else:
                out = out + observation

        return out
