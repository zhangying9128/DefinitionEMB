# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr


logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024

from transformers import T5Config, T5Model

@register_model("t5_emb")
class T5EMBModel(BaseFairseqModel):
    def __init__(self, args, decoder):
        super().__init__()
        self.args = args
        self.model = decoder

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

        #code
        self.L = None
        self.pad = decoder.pad
        self.defn_strategy = args.defn_strategy
        if args.defn_strategy == 'linear':
            self.L = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
            self.L.weight.data.normal_(mean=0.0, std=(args.embed_dim) ** -0.5)



    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        #----------code------------
        parser.add_argument(
            "--defn-strategy",
            type=str,
            choices=['linear'],
            help="load distilled embedding",
        )
        parser.add_argument(
            "--update-emb",
            action="store_true",
            help="whether to freeze the embedding weight",
        )                
        #----------code------------
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability for all fully connected layers '
                                 'in the embeddings, encoder, and pooler')
        parser.add_argument(
            "--max-target-positions", type=int, help="number of positional embeddings to learn"
        )
        #code 
        parser.add_argument('--tie-word-embeddings', action='store_true',
                            help='whether to tie word embedding with final weight matrix')
        parser.add_argument('--intermediate-dim', type=int, metavar='N',
                            help='intermediate dimension')

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

        model = T5Model(config)
        # set zero embedding for padding symbol
        model.pad = task.target_dictionary.pad()
        model.shared.weight.data[model.pad].zero_()
        return cls(args, model)

    def forward(
        self, 
        src_tokens, 
        src_lengths,
        prev_output_tokens,
        features_only= False,
        classification_head_name= None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        attention_mask = src_tokens.ne(self.pad).int()
        decoder_attention_mask = prev_output_tokens.ne(self.pad).int()
        outputs = self.model(
            input_ids=src_tokens,
            attention_mask=attention_mask,
            decoder_input_ids=prev_output_tokens,
            decoder_attention_mask=decoder_attention_mask,
        )

        # last hidden states
        x = outputs.last_hidden_state

        #code
        if self.defn_strategy == "linear":
            x = self.L(x)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        return x, 0

    def max_positions(self):
        return self.args.max_target_positions

    def embed_tokens(self, src_tokens):
        return self.model.encoder.embed_tokens(src_tokens)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        exit()
        self.classification_heads[name] = PegasusClassificationHead(
            input_dim=1024,
            inner_dim=inner_dim or 1024,
            num_classes=num_classes,
            activation_fn="tanh",
            pooler_dropout=0,
            do_spectral_norm=False,
        )
    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

@register_model_architecture("t5_emb", "t5_emb")
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
    args.tie_word_embeddings = getattr(args, "tie_word_embeddings", True)
    args.intermediate_dim = getattr(args, "intermediate_dim", 3072)

    #-----------code-----------
    args.defn_strategy = getattr(
        args, "defn_strategy", "linear"
    )
    args.update_emb = getattr(
        args, "update_emb", False
    )
    #-----------code-----------


@register_model_architecture("t5_emb", "t5_emb_base")
def t5_emb_base_architecture(args):
    base_architecture(args)


@register_model_architecture("t5_emb", "t5_emb_large")
def t5_emb_large(args):
    args.embed_dim = getattr(args, "embed_dim", 1024)
    args.num_attention_heads = getattr(args, "num_attention_heads", 16)
    args.num_layers = getattr(args, "num_layers", 24)
    args.intermediate_dim = getattr(args, "intermediate_dim", 4096)
    t5_emb_base_architecture(args)

