# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils import weight_norm


def conv_block(input_nc, output_nc, kernel_size=(4, 4), norm_layer=nn.BatchNorm2d, padding=(1, 1), stride=(2, 2), bias=False):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,)
    downrelu = nn.LeakyReLU(0.2, True)
    if norm_layer is not None:
        downnorm = norm_layer(output_nc)
        return nn.Sequential(*[downconv, downnorm, downrelu])
    else:
        return nn.Sequential(*[downconv, downrelu])


class AudioEnc(nn.Module):
    """Audio encoder"""

    def __init__(self, cfg,):
        """Transforms the spatial audio into spectrograms and computes their features"""
        super().__init__()

        self._passive_mapping_cfg = cfg.PassiveMapping
        self._task_cfg = cfg.TASK_CONFIG
        self._env_cfg = self._task_cfg.ENVIRONMENT

        self._sim_cfg = self._task_cfg.SIMULATOR
        self._audio_cfg = self._sim_cfg.AUDIO

        audioEnc_cfg = self._passive_mapping_cfg.AudioEnc

        self._n_input_channels = audioEnc_cfg.num_input_channels

        self.stft_model = torchaudio.transforms.Spectrogram(
            n_fft=self._audio_cfg.N_FFT,
            win_length=self._audio_cfg.WIN_LENGTH,
            hop_length=self._audio_cfg.HOP_LENGTH,
            power=2,
        )

        self.model = nn.Sequential(
            conv_block(self._n_input_channels, 64, norm_layer=nn.BatchNorm2d),
            conv_block(64, 64, (8, 8), stride=(4, 4), padding=(2, 2), norm_layer=nn.BatchNorm2d),
            conv_block(64, 128, norm_layer=nn.BatchNorm2d),
            conv_block(128, 256, norm_layer=nn.BatchNorm2d),
            conv_block(256, self._passive_mapping_cfg.MemoryNet.Transformer.input_size, norm_layer=nn.BatchNorm2d),
        )

        for module in self.model:
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
    def n_out_feats(self):
        return 1024

    def forward(self, observations):
        """Given the audio waveforms, transforms them into spectrograms and computes their features"""
        assert "audio" in observations
        audio_wavs = observations["audio"]
        audio_wavs = audio_wavs.permute(0, 2, 1)

        B = audio_wavs.size(0)
        n_channels = audio_wavs.size(1)

        audio_mag_spects = self.stft_model(audio_wavs.reshape(audio_wavs.size(0) * audio_wavs.size(1), -1)).pow(0.5)
        audio_mag_spects = audio_mag_spects.reshape(B, n_channels, *audio_mag_spects.size()[1:])

        out = self.model(audio_mag_spects)
        assert out.size(2) == self._passive_mapping_cfg.PositionalNet.patch_hwCh[0]
        assert out.size(3) == self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]

        return out
