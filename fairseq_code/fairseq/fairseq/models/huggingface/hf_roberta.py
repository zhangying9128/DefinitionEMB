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
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)


logger = logging.getLogger(__name__)



@register_model("hf_roberta")
class HuggingFaceRobertaLanguageModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--intermediate-size', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        default_architecture(args)
        return cls(HuggingFaceRobertaEncoder(args, task))


class HuggingFaceRobertaEncoder(FairseqDecoder):
    def __init__(self, args, task):
        try:
            from transformers import RobertaConfig, RobertaForMaskedLM
        except ImportError:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
            )

        super().__init__(task.target_dictionary)

        config = RobertaConfig(
            vocab_size=len(task.target_dictionary),
            type_vocab_size=args.type_vocab_size,
            max_position_embeddings=args.max_target_positions,
            intermediate_size=args.intermediate_size,
            hidden_size=args.embed_dim,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.attention_dropout,
            layer_norm_eps=1e-05,
        )
        self.model = RobertaForMaskedLM(config)
        # set zero embedding for padding symbol
        self.pad_idx = task.target_dictionary.pad()
        self.model.roberta.embeddings.word_embeddings.weight.data[self.pad_idx].zero_()


    def compute_mlm(self, src_tokens, src_lengths, positions, pred_size, **kwargs):
        sz = src_tokens.size(1)
        emb = self.encode_emb(self.decoder.sentence_encoder, src_tokens, positions, self.EDU_weight_token, self.discriminative_type)
        x = reverse_tensor(emb)
        positions_bias = self.encode_relative_emb(self.decoder.sentence_encoder, positions)
        for layer in self.decoder.sentence_encoder.layers:
            x, _ = layer(x, positions_bias=positions_bias)
        x = self.maybe_final_norm(self.decoder.sentence_encoder, x)
        x = reverse_tensor(x)
        x = self.output_layer(x[:, sz-pred_size:])
        return x

    def forward(
        self, input_ids, **kwargs
    ):
        attention_mask = input_ids.ne(self.pad_idx).int()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits

    def max_positions(self):
        return (512, 512)

@register_model_architecture("hf_roberta", "hf_roberta_base")
def default_architecture(args):
    args.type_vocab_size=1
    args.max_target_positions=514 #= getattr(args, "tokens_per_sample", 514)
    args.embed_dim = getattr(args, "embed_dim", 768)
    args.num_attention_heads = getattr(args, "num_attention_heads", 12)
    args.num_layers = getattr(args, "num_layers", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.intermediate_size = getattr(args, "intermediate_size", 3072)


'''
@register_model_architecture("hf_gpt2", "hf_gpt2_medium")
def hf_gpt2_medium(args):
    args.embed_dim = getattr(args, "embed_dim", 1024)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_layers = getattr(args, "num_layers", 24)
    default_architecture(args)


@register_model_architecture("hf_gpt2", "hf_gpt2_large")
def hf_gpt2_large(args):
    args.embed_dim = getattr(args, "embed_dim", 1280)
    args.num_attention_heads = getattr(args, "num_attention_heads", 20)
    args.num_layers = getattr(args, "num_layers", 36)
    default_architecture(args)


@register_model_architecture("hf_gpt2", "hf_gpt2_xl")
def hf_gpt2_xl(args):
    args.embed_dim = getattr(args, "embed_dim", 1600)
    args.num_attention_heads = getattr(args, "num_attention_heads", 25)
    args.num_layers = getattr(args, "num_layers", 48)
    default_architecture(args)
'''