# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


# min freq for sinusoidal positional encodings, source: https://arxiv.org/pdf/1706.03762.pdf
MIN_FREQ = 1e-4


class PositionalNet(nn.Module):
    """
    Takes in positional attributes and produces and produces their embeddings
    """

    def __init__(self, passive_mapping_cfg,):
        """
        Creates an instance of the class to take in positional attributes and produces and produces their embeddings
        :param passive_mapping_cfg: passive mapping config
        """
        super().__init__()
        self._passive_mapping_cfg = passive_mapping_cfg
        self._positional_net_cfg = passive_mapping_cfg.PositionalNet

        self._n_positional_obs = 5

        # source: 1. https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
        #         2. https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        self._freqs = MIN_FREQ ** (2 * (torch.arange(self._positional_net_cfg.num_freqs_for_sinusoidal,
                                                     dtype=torch.float32) // 2) /
                                   self._positional_net_cfg.num_freqs_for_sinusoidal)

        assert passive_mapping_cfg.MemoryNet.Transformer.input_size == self._positional_net_cfg.patch_hwCh[2]
        self._n_out_feats = self._positional_net_cfg.patch_hwCh[2]

        self._positional_linear = nn.Sequential(
            nn.Linear(self._positional_net_cfg.num_freqs_for_sinusoidal * self._n_positional_obs,
                      self._n_out_feats,
                      bias=False),
        )

    @property
    def n_out_feats(self):
        return self._n_out_feats

    def forward(self, observations):
        """given the positional observations, computes the positional embeddings"""

        positional_obs = observations["positional_obs"]
        assert len(positional_obs.size()) == 2
        assert positional_obs.size(-1) == self._n_positional_obs

        freqs = self._freqs.unsqueeze(0).repeat((positional_obs.size(0), 1)).to(positional_obs.device)

        positional_net_out = []
        for positional_obs_idx in range(self._n_positional_obs):
            positional_obs_thisIdx = positional_obs[:, positional_obs_idx].unsqueeze(-1)
            positional_obs_thisIdx = positional_obs_thisIdx * freqs
            positional_obs_thisIdxClone = positional_obs_thisIdx.clone()
            positional_obs_thisIdxClone[..., ::2] = torch.cos(positional_obs_thisIdx[..., ::2])
            positional_obs_thisIdxClone[..., 1::2] = torch.sin(positional_obs_thisIdx[..., 1::2])

            positional_net_out.append(positional_obs_thisIdxClone)

        positional_net_out = torch.cat(positional_net_out, dim=-1)

        assert len(positional_net_out.size()) == 2
        assert positional_net_out.size(0) == positional_obs.size(0)
        assert positional_net_out.size(1) == (self._freqs.size(0) * self._n_positional_obs)

        positional_net_out = self._positional_linear(positional_net_out)
        positional_net_out = positional_net_out.unsqueeze(-1).unsqueeze(-1)
        positional_net_out = positional_net_out.repeat(
            (1,
             1,
             self._positional_net_cfg.patch_hwCh[0],
             self._positional_net_cfg.patch_hwCh[1])
        )

        return positional_net_out


class PatchPositionalNet(nn.Module):
    """Takes in the positions of the feats corresponding to contiguous patches in an image or an audio spectrogram
    in the rasterized order and produces their embeddings"""

    def __init__(self, passive_mapping_cfg,):
        """
        Creates an instance of the class that takes in the positions of the feats corresponding to contiguous patches
        in an image or an audio spectrogram in the rasterized order and produces their embeddings
        :param passive_mapping_cfg: passive mapping config
        """

        super().__init__()
        self._passive_mapping_cfg = passive_mapping_cfg
        self._positional_net_cfg = passive_mapping_cfg.PositionalNet

        self._n_positional_obs = 1
        self._n_out_feats = self._positional_net_cfg.patch_hwCh[2]

        # source: 1. https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
        #         2. https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        self._freqs = MIN_FREQ ** (2 * (torch.arange(self._positional_net_cfg.num_freqs_for_sinusoidal,
                                                     dtype=torch.float32) // 2) /
                                   self._positional_net_cfg.num_freqs_for_sinusoidal)

        self._patch_positional_conv = nn.Sequential(
            nn.Conv2d(self._positional_net_cfg.num_freqs_for_sinusoidal *self._n_positional_obs,
                      self._n_out_feats,
                      kernel_size=1,
                      bias=False),
        )

        positional_net_out = []
        for i in range(self._positional_net_cfg.patch_hwCh[0]):
            positional_net_out_thisRow = []
            for j in range(self._positional_net_cfg.patch_hwCh[1]):
                raster_idx = i * self._positional_net_cfg.patch_hwCh[1] + j

                positional_obs_thisIdx = raster_idx * self._freqs
                positional_obs_thisIdxClone = positional_obs_thisIdx.clone()

                positional_obs_thisIdxClone[..., ::2] = torch.cos(positional_obs_thisIdxClone[..., ::2])
                positional_obs_thisIdxClone[..., 1::2] = torch.sin(positional_obs_thisIdxClone[..., 1::2])

                positional_net_out_thisRow.append(positional_obs_thisIdxClone)

            positional_net_out.append(torch.stack(positional_net_out_thisRow, dim=0))

        positional_net_out = torch.stack(positional_net_out, dim=0).permute((2, 0, 1))
        self._positional_net_out = positional_net_out

        assert self._n_out_feats == passive_mapping_cfg.MemoryNet.Transformer.input_size

    @property
    def n_out_feats(self):
        return self._n_out_feats

    def forward(self, observations):
        positional_obs = observations["positional_obs"]
        positional_net_out = self._positional_net_out.unsqueeze(0).repeat((positional_obs.size(0), 1, 1, 1))\
                            .to(positional_obs.device)

        positional_net_out = self._patch_positional_conv(positional_net_out)

        return positional_net_out
