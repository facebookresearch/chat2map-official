# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from chat2map.mapping.mapping_models.visual_cnn import VisualEnc, OccMapDec
from chat2map.mapping.mapping_models.audio_cnn import AudioEnc
from chat2map.mapping.mapping_models.modality_tag_type_net import ModalityTagTypeNet
from chat2map.mapping.mapping_models.positional_net import PositionalNet, PatchPositionalNet
from chat2map.mapping.mapping_models.fusion_net import FusionNet
from chat2map.mapping.mapping_models.memory_net import TransformerMemory


class Policy(nn.Module):
    """
    Parent class of model for passive mapping
    """
    def __init__(self,
                 context_views_enc,
                 context_audio_enc,
                 pose_net,
                 patchPose_net,
                 modality_tag_type_lookup_dict,
                 fusion_net,
                 memory_net,
                 query_occMap_dec,
                 cfg
                 ):
        """Given the audio streams and sampled frames during a conversation, the model predicts estimates of target
            occupancy maps"""
        super().__init__()
        self.context_views_enc = context_views_enc
        self.context_audio_enc = context_audio_enc
        self.pose_net = pose_net
        self.patchPose_net = patchPose_net
        self.modality_tag_type_lookup_dict = modality_tag_type_lookup_dict
        self.fusion_net = fusion_net
        self.memory_net = memory_net
        self.query_occMap_dec = query_occMap_dec

        self._cfg = cfg

        self._task_cfg = cfg.TASK_CONFIG
        self._env_cfg = self._task_cfg.ENVIRONMENT

        self._sim_cfg = self._task_cfg.SIMULATOR
        self._audio_cfg = self._sim_cfg.AUDIO

        self._passive_mapping_cfg = cfg.PassiveMapping

        self.max_context_length = self._env_cfg.MAX_CONTEXT_LENGTH
        self.max_query_length = self._env_cfg.MAX_QUERY_LENGTH

    def forward(self, observations):
        """Given the audio streams and sampled frames during a conversation, predicts estimates of target
            occupancy maps"""
        # --------------------------------------------- context encoding ------------------------------------------------
        context_feats = []
        for feat_idx in range(3):
            context_feats.append([])

        context_key_padding_mask = []

        """views encoder"""
        assert "context_maps" in observations
        context_maps = observations["context_maps"]
        
        assert "context_views_pose" in observations
        context_views_pose = observations["context_views_pose"]
        
        assert "context_views_mask" in observations
        context_views_mask = observations["context_views_mask"]
        
        assert len(context_views_mask.size()) == 3
        B = context_maps.size(0)
        num_agents = context_maps.size(1)

        context_maps = context_maps.reshape((-1, *context_maps.size()[3:]))
        context_views_dct = {"occ_map": context_maps}
        if "RGB_SENSOR" in self._cfg.SENSORS:
            assert "context_rgbs" in observations
            context_rgbs = observations["context_rgbs"]
            context_rgbs = context_rgbs.reshape((-1, *context_rgbs.size()[3:]))
            context_views_dct["rgb"] = context_rgbs

        context_views_feats = self.context_views_enc(context_views_dct)
        context_feats[0].append(context_views_feats)

        # B x num_agents x max_context_length x ... -> (B * num_agents * max_context_length) x ...; B: batch size,
        # max_context_length: transformer source sequence length S (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        context_views_pose = context_views_pose.reshape((-1, *context_views_pose.size()[3:]))

        context_views_poseFeats = self.pose_net({"positional_obs": context_views_pose})
        context_feats[0].append(context_views_poseFeats)

        context_views_posePatchFeats = self.patchPose_net({"positional_obs": context_views_pose})
        context_feats[0].append(context_views_posePatchFeats)

        context_views_modalityType = torch.LongTensor([0]).to(context_views_poseFeats.device)
        context_views_modalityTypeFeats = self.modality_tag_type_lookup_dict(context_views_modalityType)
        context_views_modalityTypeFeats =\
            context_views_modalityTypeFeats.repeat((context_views_posePatchFeats.size(0), 1, 1, 1))
        context_feats[0].append(context_views_modalityTypeFeats)

        # B x num_agents x max_context_length -> B x (num_agents * max_context_length); B: batch size,
        context_views_mask = context_views_mask.reshape((context_views_mask.size(0), -1))
        context_views_mask = context_views_mask.unsqueeze(-1).unsqueeze(-1)
        context_views_mask = context_views_mask.repeat((1,
                                                        1,
                                                        self._passive_mapping_cfg.PositionalNet.patch_hwCh[0],
                                                        self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]))
        context_views_mask = context_views_mask.reshape((context_views_mask.size(0),
                                                         context_views_mask.size(1) *\
                                                         context_views_mask.size(2) *\
                                                         context_views_mask.size(3)))
        context_key_padding_mask.append(context_views_mask)

        """self audio encoder"""
        assert "context_selfAudio" in observations
        context_selfAudio = observations["context_selfAudio"]

        assert "context_selfAudio_pose" in observations
        context_selfAudio_pose = observations["context_selfAudio_pose"]

        assert "context_selfAudio_mask" in observations
        context_selfAudio_mask = observations["context_selfAudio_mask"]
        assert len(context_selfAudio_mask.size()) == 3

        assert "context_otherAudio" in observations
        context_otherAudio = observations["context_otherAudio"]

        context_selfAudio = context_selfAudio.reshape((-1, *context_selfAudio.size()[3:]))
        context_otherAudio = context_otherAudio.reshape((-1, *context_otherAudio.size()[3:]))
        context_audio = torch.cat([context_selfAudio, context_otherAudio], dim=0)

        context_audio_feats = self.context_audio_enc({"audio": context_audio})
        context_selfAudio_feats = context_audio_feats[:context_selfAudio.size(0)]
        context_feats[1].append(context_selfAudio_feats)

        # B x num_agents x max_context_length x ... -> (B * num_agents * max_context_length) x ...; B: batch size,
        # max_context_length: transformer source sequence length S (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        context_selfAudio_pose = context_selfAudio_pose.reshape((-1, *context_selfAudio_pose.size()[3:]))

        context_selfAudio_poseFeats = self.pose_net({"positional_obs": context_selfAudio_pose})
        context_feats[1].append(context_selfAudio_poseFeats)

        context_selfAudio_posePatchFeats = self.patchPose_net({"positional_obs": context_selfAudio_pose})
        context_feats[1].append(context_selfAudio_posePatchFeats)

        context_selfAudio_modalityType = torch.LongTensor([1]).to(context_selfAudio_poseFeats.device)
        context_selfAudio_modalityTypeFeats = self.modality_tag_type_lookup_dict(context_selfAudio_modalityType)
        context_selfAudio_modalityTypeFeats =\
            context_selfAudio_modalityTypeFeats.repeat((context_selfAudio_modalityTypeFeats.size(0), 1, 1, 1))
        context_feats[1].append(context_selfAudio_modalityTypeFeats)

        # B x num_agents x max_context_length -> B x (num_agents * max_context_length); B: batch size,
        context_selfAudio_mask = context_selfAudio_mask.reshape((context_selfAudio_mask.size(0), -1))
        context_selfAudio_mask = context_selfAudio_mask.unsqueeze(-1).unsqueeze(-1)
        context_selfAudio_mask = context_selfAudio_mask.repeat((1,
                                                                1,
                                                                self._passive_mapping_cfg.PositionalNet.patch_hwCh[0],
                                                                self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]))
        context_selfAudio_mask = context_selfAudio_mask.reshape((context_selfAudio_mask.size(0),
                                                                 context_selfAudio_mask.size(1) *\
                                                                 context_selfAudio_mask.size(2) *\
                                                                 context_selfAudio_mask.size(3)))
        context_key_padding_mask.append(context_selfAudio_mask)

        """audio from other ego encoder"""
        context_otherAudio_feats = context_audio_feats[context_otherAudio.size(0):]
        
        assert "context_otherAudio_pose" in observations
        context_otherAudio_pose = observations["context_otherAudio_pose"]

        assert "context_otherAudio_mask" in observations
        context_otherAudio_mask = observations["context_otherAudio_mask"]
        assert len(context_otherAudio_mask.size()) == 3
        
        context_feats[2].append(context_otherAudio_feats)

        # B x num_agents x max_context_length x ... -> (B * num_agents * max_context_length) x ...; B: batch size,
        # max_context_length: transformer source sequence length S (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        context_otherAudio_pose = context_otherAudio_pose.reshape((-1, *context_otherAudio_pose.size()[3:]))

        context_otherAudio_poseFeats = self.pose_net({"positional_obs": context_otherAudio_pose})
        context_feats[2].append(context_otherAudio_poseFeats)

        context_otherAudio_posePatchFeats = self.patchPose_net({"positional_obs": context_otherAudio_pose})
        context_feats[2].append(context_otherAudio_posePatchFeats)
        
        context_otherAudio_modalityType =\
            torch.LongTensor([2]).to(context_otherAudio_poseFeats.device)
        context_otherAudio_modalityTypeFeats = self.modality_tag_type_lookup_dict(context_otherAudio_modalityType)
        context_otherAudio_modalityTypeFeats =\
            context_otherAudio_modalityTypeFeats.repeat((context_otherAudio_modalityTypeFeats.size(0), 1, 1, 1))
        context_feats[2].append(context_otherAudio_modalityTypeFeats)

        # B x num_agents x max_context_length -> B x (num_agents * max_context_length); B: batch size,
        context_otherAudio_mask = context_otherAudio_mask.reshape((context_otherAudio_mask.size(0), -1))
        context_otherAudio_mask = context_otherAudio_mask.unsqueeze(-1).unsqueeze(-1)
        context_otherAudio_mask = context_otherAudio_mask.repeat((1,
                                                                  1,
                                                                  self._passive_mapping_cfg.PositionalNet.patch_hwCh[0],
                                                                  self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]))
        context_otherAudio_mask = context_otherAudio_mask.reshape((context_otherAudio_mask.size(0),
                                                                   context_otherAudio_mask.size(1) *\
                                                                   context_otherAudio_mask.size(2) *\
                                                                   context_otherAudio_mask.size(3)))
        context_key_padding_mask.append(context_otherAudio_mask)

        """fusion net"""
        context_fusedFeats = []
        for idx_contextFeats in range(len(context_feats)):
            temp_context_fusedFeats = self.fusion_net(context_feats[idx_contextFeats])
            temp_context_fusedFeats = temp_context_fusedFeats.permute((0, 2, 3, 1))
            temp_context_fusedFeats = temp_context_fusedFeats.reshape((B,
                                                                       num_agents * self.max_context_length,
                                                                       temp_context_fusedFeats.size(1),
                                                                       temp_context_fusedFeats.size(2),
                                                                       temp_context_fusedFeats.size(3)))
            assert temp_context_fusedFeats.size(2) == self._passive_mapping_cfg.PositionalNet.patch_hwCh[0]
            assert temp_context_fusedFeats.size(3) == self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]
            temp_context_fusedFeats = temp_context_fusedFeats.reshape((B,
                                                                       num_agents * self.max_context_length *\
                                                                       temp_context_fusedFeats.size(2) *\
                                                                       temp_context_fusedFeats.size(3),
                                                                       -1))
            temp_context_fusedFeats = temp_context_fusedFeats.permute(1, 0, 2)

            context_fusedFeats.append(temp_context_fusedFeats)

        context_fusedFeats = torch.cat(context_fusedFeats, dim=0)

        """context and memory key padding masks"""
        context_key_padding_mask = torch.cat(context_key_padding_mask, dim=-1)

        memory_key_padding_mask = context_key_padding_mask.clone()

        # --------------------------------------------- query encoding --------------------------------------------------
        query_feats = []

        """pose encoder"""
        assert "query_views_pose" in observations
        query_views_pose = observations["query_views_pose"]
        # B x max_query_length x ... -> (B * max_query_length) x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        query_views_pose = query_views_pose.reshape((-1, *query_views_pose.size()[2:]))

        query_views_poseFeats = self.pose_net({"positional_obs": query_views_pose})
        query_feats.append(query_views_poseFeats)

        query_views_posePatchFeats = self.patchPose_net({"positional_obs": query_views_pose})
        query_feats.append(query_views_posePatchFeats)

        """fusion net"""
        query_fusedFeats = self.fusion_net(query_feats)
        query_fusedFeats = query_fusedFeats.permute((0, 2, 3, 1))
        query_fusedFeats = query_fusedFeats.reshape((B,
                                                     self.max_query_length,
                                                     query_fusedFeats.size(1),
                                                     query_fusedFeats.size(2),
                                                     query_fusedFeats.size(3)))
        assert query_fusedFeats.size(2) == self._passive_mapping_cfg.PositionalNet.patch_hwCh[0]
        assert query_fusedFeats.size(3) == self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]
        query_fusedFeats = query_fusedFeats.reshape((B,
                                                     self.max_query_length *\
                                                     query_fusedFeats.size(2) *\
                                                     query_fusedFeats.size(3),
                                                     -1))

        # B x max_query_length x ... -> max_query_length x B x -1; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        query_fusedFeats = query_fusedFeats.permute(1, 0, 2)

        """query key padding mask"""
        assert "query_views_mask" in observations
        query_key_padding_mask = observations["query_views_mask"]
        assert len(query_key_padding_mask.size()) == 2
        query_key_padding_mask = query_key_padding_mask.unsqueeze(-1).unsqueeze(-1)
        query_key_padding_mask = query_key_padding_mask.repeat((1,
                                                                1,
                                                                self._passive_mapping_cfg.PositionalNet.patch_hwCh[0],
                                                                self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]))
        query_key_padding_mask = query_key_padding_mask.reshape((query_key_padding_mask.size(0),
                                                                 query_key_padding_mask.size(1) *\
                                                                 query_key_padding_mask.size(2) *\
                                                                 query_key_padding_mask.size(3)))

        """memory encoding: context aggregation"""
        memory_outFeats =\
            self.memory_net(
                {
                    "src_feats": context_fusedFeats,
                    "tgt_feats": query_fusedFeats,
                    "src_key_padding_mask": context_key_padding_mask,
                    "tgt_key_padding_mask": query_key_padding_mask,
                    "memory_key_padding_mask": memory_key_padding_mask,
                }
            )

        # max_query_length x B x ... -> B x max_query_length x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        memory_outFeats = memory_outFeats.permute(1, 0, 2)
        memory_outFeats = memory_outFeats.reshape((B,
                                                   self.max_query_length,
                                                   self._passive_mapping_cfg.PositionalNet.patch_hwCh[0],
                                                   self._passive_mapping_cfg.PositionalNet.patch_hwCh[1],
                                                   memory_outFeats.size(2)))
        memory_outFeats = memory_outFeats.reshape((B * self.max_query_length,
                                                   self._passive_mapping_cfg.PositionalNet.patch_hwCh[0] *\
                                                   self._passive_mapping_cfg.PositionalNet.patch_hwCh[1] *\
                                                   memory_outFeats.size(4)))

        """query occMap decoder"""
        query_occMap_pred = self.query_occMap_dec({"memory_outFeats": memory_outFeats})

        # (B * max_query_length) x ... -> B x max_query_length x ...; B: batch size,
        # max_query_length: transformer target sequence length T (https://pytorch.org/docs/1.4.0/nn.html#torch.nn.Transformer)
        query_occMap_pred = query_occMap_pred.reshape((B,
                                                       self.max_query_length,
                                                       *query_occMap_pred.size()[1:]))

        return query_occMap_pred


class PassiveMappingPolicy(Policy):
    """
    Model for passive mapping
    """
    def __init__(
        self,
        cfg,
    ):
        passive_mapping_cfg = cfg.PassiveMapping
        task_cfg = cfg.TASK_CONFIG
        sim_cfg = task_cfg.SIMULATOR

        # --------------------------------------------- context encoders -----------------------------------------------
        """pose net"""
        pose_net = PositionalNet(
            passive_mapping_cfg=passive_mapping_cfg,
        )

        patchPose_net = PatchPositionalNet(
            passive_mapping_cfg=passive_mapping_cfg,
        )

        """modality tag type lookup table"""
        modality_tag_type_lookup_dict = ModalityTagTypeNet(
            n_modality_tag_types=3,
            passive_mapping_cfg=passive_mapping_cfg,
        )

        """views encoder"""
        context_views_enc = VisualEnc(
            cfg=cfg,
        )

        """audio encoder"""
        context_audio_enc = AudioEnc(
            cfg=cfg,
        )

        """fusion net"""
        fusion_net = FusionNet()

        # --------------------------------------------- memory net -----------------------------------------------------
        memory_net = TransformerMemory(
            cfg=cfg,
        )

        # --------------------------------------- target occ-map decoder -----------------------------------------------
        query_occMap_dec = OccMapDec(
            passive_mapping_cfg=passive_mapping_cfg,
            sim_cfg=sim_cfg,
        )

        super().__init__(
            context_views_enc,
            context_audio_enc,
            pose_net,
            patchPose_net,
            modality_tag_type_lookup_dict,
            fusion_net,
            memory_net,
            query_occMap_dec,
            cfg,
        )
