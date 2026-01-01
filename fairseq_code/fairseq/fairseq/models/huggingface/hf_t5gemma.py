# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import warnings
import torch
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
try:
    from transformers import T5GemmaConfig, T5GemmaForConditionalGeneration
except ImportError:
    #raise ImportError(
    #    "\n\nPlease install huggingface/transformers with:"
    #    "\n\n  pip install transformers"
    #)
    warnings.warn(
        "\n\nWarning: Failed to import T5Gemma from transformers. "
        "Some features may not be available.",
        UserWarning
    )
    # 设置为None，这样后续代码可以检查是否可用
    T5GemmaConfig = None
    T5GemmaForConditionalGeneration = None

logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 8192


@register_model("hf_t5gemma")
class HuggingFaceT5GemmaEncoderDecoder(BaseFairseqModel):
    def __init__(self, args, decoder):
        super().__init__()
        self.args = args
        self.model = decoder
        self.pad_idx = decoder.pad

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        #code
        parser.add_argument('--tie-word-embeddings', action='store_true',
                            help='whether to tie word embedding with final weight matrix')
        parser.add_argument('--intermediate-dim', type=int, metavar='N',
                            help='intermediate dimension')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)

        config = T5GemmaConfig(
            vocab_size=len(task.target_dictionary),
            # Encoder配置
            encoder={
                "hidden_size": args.embed_dim,
                "intermediate_size": args.intermediate_dim,
                "num_hidden_layers": args.num_layers,
                "num_attention_heads": args.num_attention_heads,
                "num_key_value_heads": args.num_key_value_heads,
                "max_position_embeddings": args.max_target_positions,
                "sliding_window": args.sliding_window,
                "rope_theta": args.rope_theta,
                "attention_dropout": args.dropout,
                "dropout_rate": args.dropout,
                "hidden_activation": "gelu_pytorch_tanh",
                "head_dim": args.embed_dim // args.num_attention_heads,
                "rms_norm_eps": 1e-6,
                "attn_logit_softcapping": 50.0,
                "query_pre_attn_scalar": args.embed_dim // args.num_attention_heads,
                "layer_types": ["sliding_attention", "full_attention"] * (args.num_layers // 2),
            },
            # Decoder配置 
            decoder={
                "hidden_size": args.embed_dim,
                "intermediate_size": args.intermediate_dim,
                "num_hidden_layers": args.num_layers,
                "num_attention_heads": args.num_attention_heads,
                "num_key_value_heads": args.num_key_value_heads,
                "max_position_embeddings": args.max_target_positions,
                "sliding_window": args.sliding_window,
                "rope_theta": args.rope_theta,
                "attention_dropout": args.dropout,
                "dropout_rate": args.dropout,
                "hidden_activation": "gelu_pytorch_tanh",
                "head_dim": args.embed_dim // args.num_attention_heads,
                "cross_attention_hidden_size": args.embed_dim,
                "rms_norm_eps": 1e-6,
                "attn_logit_softcapping": 50.0,
                "query_pre_attn_scalar": args.embed_dim // args.num_attention_heads,
                "layer_types": ["sliding_attention", "full_attention"] * (args.num_layers // 2),
                "is_decoder": True,
            },
            pad_token_id=task.target_dictionary.pad(),
            eos_token_id=[task.target_dictionary.eos()],
            dropout_rate=args.dropout,
            is_encoder_decoder=True,
            use_cache=True,
            tie_word_embeddings=args.tie_word_embeddings,
            #attn_implementation="eager"
        )
        print(config)
        model = T5GemmaForConditionalGeneration(config)

        # set zero embedding for padding symbol
        model.pad = task.target_dictionary.pad()
        model.model.encoder.embed_tokens.weight.data[model.pad].zero_()
        model.model.decoder.embed_tokens.weight.data[model.pad].zero_()
        return cls(args, model)

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, **kwargs
    ):
        attention_mask = src_tokens.ne(self.pad_idx).int()
        decoder_attention_mask = prev_output_tokens.ne(self.pad_idx).int()
        outputs = self.model(
            input_ids=src_tokens,
            attention_mask=attention_mask,
            decoder_input_ids=prev_output_tokens,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs.logits

    def extract_last_hidden_states(
        self, src_tokens, src_lengths, prev_output_tokens, **kwargs
    ):
        attention_mask = src_tokens.ne(self.pad_idx).int()
        decoder_attention_mask = prev_output_tokens.ne(self.pad_idx).int()
        outputs = self.model(
            input_ids=src_tokens,
            attention_mask=attention_mask,
            decoder_input_ids=prev_output_tokens,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs.decoder_hidden_states[-1] #batch_size x target_length x hidden_dimension

    def max_positions(self):
        return (self.model.config.encoder.max_position_embeddings, self.model.config.decoder.max_position_embeddings)


@register_model_architecture("hf_t5gemma", "hf_t5gemma")
def base_architecture(args):
    if getattr(args, "max_target_positions", None) is None:
        args.max_target_positions = getattr(
            args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
        )
    # T5Gemma-XL配置
    args.embed_dim = getattr(args, "embed_dim", 2048)
    args.num_attention_heads = getattr(args, "num_attention_heads", 32)
    args.num_key_value_heads = getattr(args, "num_key_value_heads", 32)
    args.num_layers = getattr(args, "num_layers", 24)
    args.intermediate_dim = getattr(args, "intermediate_dim", 5120)
    args.dropout = getattr(args, "dropout", 0.0)
    args.sliding_window = getattr(args, "sliding_window", 4096)
    args.rope_theta = getattr(args, "rope_theta", 10000.0)
    
    #code
    args.tie_word_embeddings = getattr(args, "tie_word_embeddings", False)

@register_model_architecture("hf_t5gemma", "hf_t5gemma_xl")
def hf_t5gemma_xl(args):
    base_architecture(args)

@register_model_architecture("hf_t5gemma", "hf_t5gemma_l")
def hf_t5gemma_l(args):
    """T5Gemma-L配置 (1024维度)"""
    if getattr(args, "max_target_positions", None) is None:
        args.max_target_positions = getattr(
            args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
        )
    
    # T5Gemma-L配置
    args.embed_dim = getattr(args, "embed_dim", 1024)  # 改为1024
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)  # 改为16
    args.num_key_value_heads = getattr(args, "num_key_value_heads", 16)  # 改为16
    args.num_layers = getattr(args, "num_layers", 24)  # 保持24层
    args.intermediate_dim = getattr(args, "intermediate_dim", 2816)  # 改为2816
    args.dropout = getattr(args, "dropout", 0.0)
    args.sliding_window = getattr(args, "sliding_window", 4096)
    args.rope_theta = getattr(args, "rope_theta", 10000.0)
    
    # code
    args.tie_word_embeddings = getattr(args, "tie_word_embeddings", False)


