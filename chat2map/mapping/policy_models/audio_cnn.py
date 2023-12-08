# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils import weight_norm


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


class AudioEnc(nn.Module):
    """Encodes the spatial audio input to a policy"""

    def __init__(self, config,):
        super().__init__()

        self.config = config

        sim_cfg = config.TASK_CONFIG.SIMULATOR
        self.audio_cfg = sim_cfg.AUDIO

        self.ppo_cfg = config.RL.PPO
        self.audio_enc_cfg = self.ppo_cfg.AudioEnc

        self._n_input_channels = 9

        self.stft_converter = torchaudio.transforms.Spectrogram(
            n_fft=self.audio_cfg.N_FFT,
            win_length=self.audio_cfg.WIN_LENGTH,
            hop_length=self.audio_cfg.HOP_LENGTH,
            power=2,
        )

        self.audio_enc = nn.Sequential(
            conv_block(self._n_input_channels, 64),
            conv_block(64, 64, (8, 8), stride=(4, 4), padding=(2, 2)),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, self.audio_enc_cfg.output_size),
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1))),
        )

        for module in self.audio_enc:
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
                        
    @property
    def n_out_feats(self):
        return self.audio_enc_cfg.output_size

    def forward(self, observation):
        """Given the spatial audio input, transforms it into a spectrogram and encodes it into audio features"""
        assert "audio" in observation
        audio = observation["audio"]
        audio = audio.permute(0, 2, 1)

        bs = audio.size(0)
        n_channels = audio.size(1)
        audio = self.stft_converter(audio.reshape(audio.size(0) * audio.size(1), -1)).pow(0.5)
        audio = audio.reshape(bs, n_channels, *audio.size()[1:])

        audio_feats = self.audio_enc(audio)

        audio_feats = audio_feats.squeeze(-1).squeeze(-1)

        return audio_feats
