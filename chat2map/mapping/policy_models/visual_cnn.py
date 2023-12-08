# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def conv_block(input_nc, output_nc, kernel_size=(4, 4), norm_layer=nn.BatchNorm2d, padding=(1, 1), stride=(2, 2), bias=False,
               groups=16, use_groupnorm=False,):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,)
    downrelu = nn.LeakyReLU(0.2, True)
    if norm_layer is not None:
        if use_groupnorm:
            downnorm = norm_layer(groups, output_nc)
        else:
            downnorm = norm_layer(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv, downrelu])


class VisualEnc(nn.Module):
    """Takes in observations and produces an embedding of the topdown map ( + rgb) components"""

    def __init__(self, config,):
        super().__init__()

        self.config = config

        self.task_cfg = config.TASK_CONFIG.TASK

        self.ppo_cfg = config.RL.PPO
        self.visual_enc_cfg = self.ppo_cfg.VisualEnc

        assert sorted(self.config.SENSORS) == ["DEPTH_SENSOR", "RGB_SENSOR"]

        self._n_input_channels = self.task_cfg.CONTEXT_EGO_LOCAL_MAP_SENSOR.NUM_CHANNELS

        map_enc = [
            conv_block(self._n_input_channels, 64),
            conv_block(64, 64),
            conv_block(64, 128, padding=(2, 2)),
            conv_block(128, 256, (3, 3), padding=(1, 1), stride=(1, 1)),
            conv_block(256, self.visual_enc_cfg.output_size, (3, 3), padding=(1, 1), stride=(1, 1)),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))),
        ]

        self.map_enc = nn.Sequential(*map_enc)

        for module in self.map_enc:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("leaky_relu", 0.2)
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

        rgb_enc = [
            conv_block(3, 64),
            conv_block(64, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))),
        ]
        self.rgb_enc = nn.Sequential(*rgb_enc)

        self.fuse_net = nn.Linear(2 * self.visual_enc_cfg.output_size, self.visual_enc_cfg.output_size, bias=False,)

        for module in self.rgb_enc:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("leaky_relu", 0.2)     # nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    @property
    def is_blind(self):
        return False

    @property
    def n_out_feats(self):
        return self.visual_enc_cfg.output_size

    def forward(self, observation):
        """Given the sampled RGB frames and 90 degree FoV local egocentric occupancy maps, computes their features"""
        assert "map" in observation
        map = observation["map"]
        map = map.permute(0, 3, 1, 2)

        map_feats = self.map_enc(map)
        map_feats = map_feats.squeeze(-1).squeeze(-1)

        assert "rgb" in observation
        rgb = observation["rgb"]
        """ permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH] """
        rgb = rgb.permute(0, 3, 1, 2)
        """ normalize RGB """
        rgb = rgb.float() / 255.0

        rgb_feats = self.rgb_enc(rgb)
        rgb_feats = rgb_feats.squeeze(-1).squeeze(-1)

        feats = torch.cat([map_feats, rgb_feats], dim=-1)

        feats = self.fuse_net(feats)

        return feats
