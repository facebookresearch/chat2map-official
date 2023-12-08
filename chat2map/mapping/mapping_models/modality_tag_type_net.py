# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn


class ModalityTagTypeNet(nn.Module):
    """Takes the modality type tag for a certain modality and produces its embeddings"""

    def __init__(self, n_modality_tag_types, passive_mapping_cfg,):
        """
        Creates an instance of the class that takes the modality type tag for a certain modality and produces its
        embeddings
        :param n_modality_tag_types: number of modality tag types
        :param passive_mapping_cfg: passive mapping config
        """

        super().__init__()
        self._positional_net_cfg = passive_mapping_cfg.PositionalNet

        self._out_h = self._positional_net_cfg.patch_hwCh[0]
        self._out_w = self._positional_net_cfg.patch_hwCh[1]
        self._n_out_ch = self._positional_net_cfg.patch_hwCh[2]

        assert self._n_out_ch == passive_mapping_cfg.modality_tag_type_encoding_size, print(self._n_out_ch,
                                                                                            passive_mapping_cfg.modality_tag_type_encoding_size)
        self.modality_tag_type_lookup_dict = nn.Embedding(n_modality_tag_types,
                                                          passive_mapping_cfg.modality_tag_type_encoding_size,)

    def forward(self, x):
        """Given the modality type tag, computes the modality embeddings"""
        out = self.modality_tag_type_lookup_dict(x)
        out = out.unsqueeze(-1).unsqueeze(-1)
        out = out.repeat((1, 1, self._out_h, self._out_w))
        return out
