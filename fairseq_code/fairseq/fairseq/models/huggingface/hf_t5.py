# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from typing import Dict, List, Optional

import torch
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
try:
    from transformers import T5Config, T5ForConditionalGeneration
except ImportError:
    raise ImportError(
        "\n\nPlease install huggingface/transformers with:"
        "\n\n  pip install transformers"
    )

logger = logging.getLogger(__name__)


DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("hf_t5")
class HuggingFaceT5EncoderDecoder(BaseFairseqModel):
    def __init__(self, args, decoder):
        super().__init__()
        self.args = args
        self.model = decoder
        self.pad_idx = decoder.pad_idx

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

        config = T5Config(
            vocab_size=len(task.target_dictionary),
            max_position_embeddings=args.max_target_positions,
            d_model=args.embed_dim,
            d_ff=args.intermediate_dim,
            num_layers=args.num_layers,
            num_decoder_layers=args.num_layers,
            num_heads=args.num_attention_heads,
            dropout_rate=args.dropout,
            layer_norm_epsilon=1e-6,
            decoder_start_token_id=task.target_dictionary.eos(),
            pad_token_id=task.target_dictionary.pad(),
            eos_token_id=task.target_dictionary.eos(),
            is_encoder_decoder=True,
            d_kv=64,
            relative_attention_num_buckets=32,
            tie_word_embeddings=args.tie_word_embeddings,
            output_hidden_states=True,
        )
        print(config)
        model = T5ForConditionalGeneration(config)

        # set zero embedding for padding symbol
        model.pad_idx = task.target_dictionary.pad()
        model.shared.weight.data[model.pad_idx].zero_()
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
        return (self.model.config.max_position_embeddings, self.model.config.max_position_embeddings)


@register_model_architecture("hf_t5", "hf_t5")
def base_architecture(args):
    if getattr(args, "max_target_positions", None) is None:
        args.max_target_positions = getattr(
            args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
        )
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.num_attention_heads = getattr(args, "num_attention_heads", 12)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    
    #code
    args.tie_word_embeddings = getattr(args, "tie_word_embeddings", False)
    args.intermediate_dim = getattr(args, "intermediate_dim", 3072)

@register_model_architecture("hf_t5", "hf_t5_large")
def hf_t5_large(args):
    args.embed_dim = getattr(args, "embed_dim", 1024)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_layers = getattr(args, "num_layers", 24)
    args.intermediate_dim = getattr(args, "intermediate_dim", 4096)
    base_architecture(args)



