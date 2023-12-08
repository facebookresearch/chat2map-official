# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class TransformerDecoderWoSelfAttnLayer(nn.Module):
    r"""TransformerDecoderWoSelfAttnLayer is made up of multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderWoSelfAttnLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerWoSelfAttnInDecoder(nn.Module):
    r"""
    source: https://pytorch.org/docs/1.4.0/_modules/torch/nn/modules/transformer.html#Transformer.forward
    A transformer model without self-attention inside transformer decoder. User is able to modify the attributes as needed.
    The architecture is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation='relu', d_model_out=512,):
        super(TransformerWoSelfAttnInDecoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderWoSelfAttnLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.linear = nn.Sequential()
        if d_model != d_model_out:
            self.linear = nn.Sequential(
                nn.Linear(d_model, d_model_out),
            )

        self._reset_parameters()

        self.d_model = d_model
        self.d_model_out = d_model_out
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)

        out = self.linear(out)

        return out

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class TransformerMemory(nn.Module):
    """Transformer memory"""
    def __init__(self, cfg):
        """Creates an instance of the transformer memory"""
        super().__init__()

        self._cfg = cfg

        self._passive_mapping_cfg = cfg.PassiveMapping
        self._transformer_cfg = self._passive_mapping_cfg.MemoryNet.Transformer
        self._task_cfg = cfg.TASK_CONFIG
        self._env_cfg = self._task_cfg.ENVIRONMENT
        self._sim_cfg = self._task_cfg.SIMULATOR

        self.transformer = TransformerWoSelfAttnInDecoder(
            d_model=self._transformer_cfg.input_size,
            nhead=self._transformer_cfg.nhead,
            num_encoder_layers=self._transformer_cfg.num_encoder_layers,
            num_decoder_layers=self._transformer_cfg.num_decoder_layers,
            dim_feedforward=self._transformer_cfg.hidden_size,
            dropout=self._transformer_cfg.dropout,
            activation=self._transformer_cfg.activation,
            d_model_out=self._transformer_cfg.decoder_out_size,
        )

        context_length_multiplier = 3
        context_length_multiplier *= self._sim_cfg.ALL_AGENTS.NUM
        context_length_multiplier *= (self._passive_mapping_cfg.PositionalNet.patch_hwCh[0] *\
                                      self._passive_mapping_cfg.PositionalNet.patch_hwCh[1])

        query_length_multiplier = self._passive_mapping_cfg.PositionalNet.patch_hwCh[0] *\
                                  self._passive_mapping_cfg.PositionalNet.patch_hwCh[1]

        self._src_mask = self._convert_attn_masks_to_transformer_format(
            torch.ones((self._env_cfg.MAX_CONTEXT_LENGTH * context_length_multiplier,
                        self._env_cfg.MAX_CONTEXT_LENGTH * context_length_multiplier,))
        )
        self._mem_mask = self._convert_attn_masks_to_transformer_format(
            torch.ones((self._env_cfg.MAX_QUERY_LENGTH * query_length_multiplier,
                        self._env_cfg.MAX_CONTEXT_LENGTH * context_length_multiplier,))
        )

        self._tgt_mask = self._convert_attn_masks_to_transformer_format(
            torch.eye(self._env_cfg.MAX_QUERY_LENGTH * query_length_multiplier)
        )

    def _convert_key_padding_masks_to_transformer_format(self, key_padding_masks):
        r"""The key_padding_masks is a FloatTensor with
            -   0 for invalid locations, and
            -   1 for valid locations.
        The required format is a BoolTensor with
            -   True for invalid locations, and
            -   False for valid locations

        source:
            - https://pytorch.org/docs/1.4.0/_modules/torch/nn/modules/transformer.html#TransformerDecoder
            - https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        """
        return (1 - key_padding_masks) > 0

    def _convert_attn_masks_to_transformer_format(self, attn_masks):
        r"""The attn_masks is a FloatTensor with
            -   0 for invalid locations, and
            -   1 for valid locations.
        The required format is a FloatTensor with
            -   float('-inf') for invalid locations, and
            -   0. for valid locations

        source:
            - https://pytorch.org/docs/1.4.0/_modules/torch/nn/modules/transformer.html#TransformerDecoder
            - https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3
        """
        return attn_masks.float().masked_fill(attn_masks == 0, float('-inf')).masked_fill(attn_masks == 1, float(0.0))

    def forward(self, observations):
        """computes transformer memory features given observations"""
        assert "src_feats" in observations
        src_feats = observations["src_feats"]

        assert "tgt_feats" in observations
        tgt_feats = observations["tgt_feats"]

        """how masks works -- source: https://github.com/pytorch/pytorch/blob/7f73f1d591afba823daa4a99a939217fb54d7688/torch/nn/functional.py#L3360"""
        assert "src_key_padding_mask" in observations
        src_key_padding_mask = self._convert_key_padding_masks_to_transformer_format(observations["src_key_padding_mask"])

        assert "tgt_key_padding_mask" in observations
        tgt_key_padding_mask = self._convert_key_padding_masks_to_transformer_format(observations["tgt_key_padding_mask"])

        assert "memory_key_padding_mask" in observations
        memory_key_padding_mask = self._convert_key_padding_masks_to_transformer_format(observations["memory_key_padding_mask"])

        self._src_mask = self._src_mask.to(src_feats.device)
        self._mem_mask = self._mem_mask.to(memory_key_padding_mask.device)
        self._tgt_mask = self._tgt_mask.to(tgt_feats.device)

        out = self.transformer(
            src_feats,
            tgt_feats,
            src_mask=self._src_mask,
            tgt_mask=self._tgt_mask,
            memory_mask=self._mem_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return out
