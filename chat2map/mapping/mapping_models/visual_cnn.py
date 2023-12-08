# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def conv_block(input_nc, output_nc, kernel_size=(4, 4), norm_layer=nn.BatchNorm2d, padding=(1, 1), stride=(2, 2),
               bias=False):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,)
    downrelu = nn.LeakyReLU(0.2, True)
    if norm_layer is not None:
        downnorm = norm_layer(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv, downrelu])


def convT_block(input_nc, output_nc, kernel_size=(4, 4), outermost=False, norm_layer=nn.BatchNorm2d, stride=(2, 2),
                padding=(1, 1), output_padding=(0, 0), bias=False, use_sigmoid=False,):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                                output_padding=output_padding, bias=bias)
    uprelu = nn.ReLU(True)
    if not outermost:
        if norm_layer is not None:
            upnorm = norm_layer(output_nc)
            return nn.Sequential(*[upconv, upnorm, uprelu])
        else:
            return nn.Sequential(*[upconv, uprelu])
    else:
        if use_sigmoid:
            return nn.Sequential(*[upconv, nn.Sigmoid()])
        else:
            return nn.Sequential(*[upconv,])


class VisualEnc(nn.Module):
    """Visual encoder"""

    def __init__(self, cfg=None):
        """Takes in RGB images and 90 degree FoV local egocentric map inputs and encodes them"""
        super().__init__()

        passive_mapping_cfg = cfg.PassiveMapping
        sim_cfg = cfg.TASK_CONFIG.SIMULATOR

        assert "RGB_SENSOR" in cfg.SENSORS

        self._n_inputMap_channels = sim_cfg.EGO_LOCAL_OCC_MAP.NUM_CHANNELS

        self._num_out_channels = passive_mapping_cfg.VisualEnc.num_out_channels
        assert passive_mapping_cfg.MemoryNet.Transformer.input_size == 2 * self._num_out_channels

        cnn_layers = [
            conv_block(self._n_inputMap_channels, 64, norm_layer=nn.BatchNorm2d),
            conv_block(64, 64, norm_layer= nn.BatchNorm2d),
            conv_block(64, 128, padding=(2, 2), norm_layer=nn.BatchNorm2d),
            conv_block(128, 256, (3, 3), padding=(1, 1), stride=(1, 1), norm_layer=nn.BatchNorm2d),
            conv_block(256, self._num_out_channels, (3, 3), padding=(1, 1), stride=(1, 1), norm_layer=nn.BatchNorm2d)
        ]
        self.cnn = nn.Sequential(*cnn_layers)

        for module in self.cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("leaky_relu", 0.2)
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

        rgb_cnn_layers = [
            conv_block(3, 64, norm_layer=nn.BatchNorm2d),
            conv_block(64, 64, norm_layer=nn.BatchNorm2d),
            conv_block(64, 128, norm_layer=nn.BatchNorm2d),
            conv_block(128, 256, norm_layer=nn.BatchNorm2d),
            conv_block(256, self._num_out_channels, norm_layer=nn.BatchNorm2d),
        ]
        self.rgb_cnn = nn.Sequential(*rgb_cnn_layers)

        for module in self.rgb_cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("leaky_relu", 0.2)
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    @property
    def is_blind(self):
        return False

    @property
    def n_out_feats(self):
        return 16 * 512

    def _preprocess_rgb(self, rgb_observations):
        return rgb_observations

    def forward(self, observations,):
        """Given RGB imags and 90 degree FoV egocentric local occupancy maps, produces visual features"""
        assert "occ_map" in observations
        occMap_observations = observations["occ_map"]
        occMap_observations = occMap_observations.permute(0, 3, 1, 2)

        occMap_out = self.cnn(occMap_observations)

        assert "rgb" in observations
        rgb_observations = observations["rgb"]
        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
        rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        rgb_observations = rgb_observations.float() / 255.0  # normalize RGB
        rgb_observations = self._preprocess_rgb(rgb_observations)

        rgb_out = self.rgb_cnn(rgb_observations)

        out = torch.cat([occMap_out, rgb_out], dim=1)

        return out


class OccMapDec(nn.Module):
    """Occupancy map decoder"""

    def __init__(self, passive_mapping_cfg, sim_cfg,):
        """Takes in feature outputs of the transformer decoder and predicts estimates of 360 degree FoV local
           egocentric occupancy map targets"""
        super().__init__()

        self._passive_mapping_cfg = passive_mapping_cfg
        self._glob_can_occ_map_ego_crop_cfg = sim_cfg.GT_GLOBAL_CANONICAL_OCC_MAP_EGO_CROP

        assert self._glob_can_occ_map_ego_crop_cfg.SIZE in [64, 80, 96, 128]

        assert passive_mapping_cfg.MemoryNet.type == "transformer"

        assert passive_mapping_cfg.MemoryNet.Transformer.decoder_out_size == 1024
        self._n_inputMapFeat_channels = 1024
        self._inputFeat_h = 4
        self._inputFeat_w = 4
        self._input_feat_size = self._n_inputMapFeat_channels * self._inputFeat_h * self._inputFeat_w

        if self._glob_can_occ_map_ego_crop_cfg.SIZE == 64:
            self.dec_cnn = nn.Sequential(
                convT_block(1024, 64 * 8, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 8, 64 * 4, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 4, 64 * 2, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 2, 64 * 1, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 1, self._glob_can_occ_map_ego_crop_cfg.NUM_CHANNELS, (3, 3), stride=(1, 1),
                            padding=(1, 1), outermost=True, use_sigmoid=True,),
            )
        elif self._glob_can_occ_map_ego_crop_cfg.SIZE == 80:
            self.dec_cnn = nn.Sequential(
                conv_block(1024, 64 * 8, kernel_size=(2, 2), padding=(1, 1), stride=(1, 1), norm_layer=nn.BatchNorm2d),
                convT_block(64 * 8, 64 * 8, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 8, 64 * 4, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 4, 64 * 2, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 2, 64 * 1, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 1, self._glob_can_occ_map_ego_crop_cfg.NUM_CHANNELS, (3, 3), stride=(1, 1),
                            padding=(1, 1), outermost=True, use_sigmoid=True,),
            )
        elif self._glob_can_occ_map_ego_crop_cfg.SIZE == 96:
            self.dec_cnn = nn.Sequential(
                conv_block(1024, 64 * 8, kernel_size=(1, 1), padding=(1, 1), stride=(1, 1), norm_layer=nn.BatchNorm2d),
                convT_block(64 * 8, 64 * 8, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 8, 64 * 4, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 4, 64 * 2, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 2, 64 * 1, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 1, self._glob_can_occ_map_ego_crop_cfg.NUM_CHANNELS, (3, 3), stride=(1, 1),
                            padding=(1, 1), outermost=True, use_sigmoid=True,),
            )
        elif self._glob_can_occ_map_ego_crop_cfg.SIZE == 128:
            self.dec_cnn = nn.Sequential(
                convT_block(1024, 64 * 8, norm_layer=nn.BatchNorm2d),  
                convT_block(64 * 8, 64 * 4, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 4, 64 * 2, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 2, 64 * 1, norm_layer=nn.BatchNorm2d),
                convT_block(64 * 1, self._glob_can_occ_map_ego_crop_cfg.NUM_CHANNELS,
                            outermost=True, use_sigmoid=True,),
            )
        else:
            raise NotImplementedError

        self.layer_init()

    def layer_init(self):
        for module in self.dec_cnn:
            for layer in module:
                if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                    nn.init.kaiming_normal_(
                        layer.weight, nn.init.calculate_gain("relu")
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if layer.affine:
                        layer.weight.data.fill_(1)
                        layer.bias.data.zero_()

    def forward(self, observations,):
        """Given feature outputs of the transformer memory decoder, computes estimates of the 360 degree FoV local
           egocentric target occupancy maps"""
        assert "memory_outFeats" in observations
        memory_outFeats = observations["memory_outFeats"]
        assert len(memory_outFeats.size()) == 2
        assert memory_outFeats.size(1) == self._input_feat_size
        memory_outFeats =\
            memory_outFeats.reshape((memory_outFeats.size(0),
                                     self._inputFeat_h,
                                     self._inputFeat_w,
                                     -1))
        memory_outFeats = memory_outFeats.permute((0, 3, 1, 2))

        out = self.dec_cnn(memory_outFeats)

        assert len(out.size()) == 4
        # permute tensor to dimension [BATCH x HEIGHT x WIDTH x CHANNEL]
        out = out.permute(0, 2, 3, 1)

        return out
