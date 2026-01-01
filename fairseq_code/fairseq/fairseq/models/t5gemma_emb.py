"""
T5Gemma: T5-style encoder-decoder with Gemma architecture
"""

import logging
import warnings

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
from torch.amp import autocast  # 新版PyTorch


logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 8192
# 导入T5Gemma相关类
try:
    from transformers import T5GemmaConfig, T5GemmaModel
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

@register_model("t5gemma_emb")
class T5GemmaEMBModel(BaseFairseqModel):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

        self.classification_heads = nn.ModuleDict()

        #zhagying
        self.L = None
        self.pad = model.pad
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
        # ----------code------------
        parser.add_argument('--embed-dim', type=int, metavar='N',
                            help='embedding dimension')
        parser.add_argument('--num-attention-heads', type=int, metavar='N',
                            help='num attention heads')
        parser.add_argument('--num-layers', type=int, metavar='N',
                            help='num layers')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument(
            "--max-target-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument('--intermediate-dim', type=int, metavar='N',
                            help='intermediate dimension')
        parser.add_argument('--num-key-value-heads', type=int, metavar='N',
                            help='num key value heads for grouped query attention')
        parser.add_argument('--sliding-window', type=int, metavar='N',
                            help='sliding window size')
        parser.add_argument('--rope-theta', type=float, default=10000.0,
                            help='RoPE theta parameter')

        #code 
        parser.add_argument('--tie-word-embeddings', action='store_true',
                            help='whether to tie decoder word embedding with final weight matrix')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        t5gemma_base_architecture(args)

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
            #attn_implementation="eager"
        )
        print(config)

        model = T5GemmaModel(config)
        # 设置模型始终输出hidden states
        model.config.output_hidden_states = True

        # set zero embedding for padding symbol
        model.pad = task.target_dictionary.pad()
        model.encoder.embed_tokens.weight.data[model.pad].zero_()
        model.decoder.embed_tokens.weight.data[model.pad].zero_()
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

        # 获取最后一层的隐藏状态
        x = outputs.decoder_hidden_states[-1] if hasattr(outputs, 'decoder_hidden_states') else outputs.last_hidden_state

        #code
        if self.defn_strategy == "linear":
            x = x.float()
            x = self.L(x)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        return x, 0

    def max_positions(self):
        return self.args.max_target_positions

    def embed_tokens(self, src_tokens):
        if hasattr(self.model, 'encoder'):
            return self.model.encoder.embed_tokens(src_tokens)
        return self.model.shared(src_tokens)

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


@register_model_architecture("t5gemma_emb", "t5gemma_emb_base")
def t5gemma_base_architecture(args):
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
    
    # code
    args.defn_strategy = getattr(args, "defn_strategy", "linear")
    args.update_emb = getattr(args, "update_emb", False)


@register_model_architecture("t5gemma_emb", "t5gemma_emb_xl")
def t5gemma_xl_architecture(args):
    t5gemma_base_architecture(args)

@register_model_architecture("t5gemma_emb", "t5gemma_emb_l")
def t5gemma_l_architecture(args):
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
    args.defn_strategy = getattr(args, "defn_strategy", "linear")
    args.update_emb = getattr(args, "update_emb", False)