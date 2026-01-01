# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
)

class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.share_encoder_decoder_input_embeddings:
            #code
            if src_dict != tgt_dict:
                raise ValueError("--share-encoder-decoder-input-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-encoder-decoder-input-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-encoder-decoder-input-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = False
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

#code 20230825
import math
# Quick utility function to sample from the Gumbel-distribution: -Log(-Log(Uniform)), eps to avoid numerical errors
def sample_gumbel(logits):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    return gumbels

#code 20230807
import torch.nn.functional as F
class Code(nn.Module):
    # code code: need to modify generate-code to handle out-of-vocabulary
    # code code: need to generate embedding_matrix for weight-tying, it may be very slow each time
    def __init__(self, num_embeddings, embedding_dim, padding_idx, distill_K, distill_M):
        super().__init__()
        self.M = distill_M
        self.K = distill_K
        self.A = nn.Parameter(torch.Tensor(distill_M, embedding_dim, distill_K))
        self.vocabs = nn.Parameter(torch.zeros(num_embeddings, distill_M, dtype=torch.int64), requires_grad=False)

        # can save more memory if embedding_dim > distill_M * distill_K
        hidden_size = int(distill_M * distill_K / 2)

        self.h_w = nn.Linear(embedding_dim, hidden_size, bias=True)
        self.alpha_w = nn.Linear(hidden_size, distill_M * distill_K, bias=True)
        self.reset_parameters(self.A, embedding_dim)

        #code 20230910 add A
        self.add_para_A = False
        self.add_para_vocabsd = False

        #self.add_A(8)

    def add_A(self, distill_KK):
        self.KK = distill_KK
        self.alpha_w_AK = nn.Linear(int(self.M * self.K / 2), self.M * distill_KK, bias=True).to(device="cuda")
        self.AK = nn.Parameter(torch.Tensor(self.M, self.A.size(1), distill_KK)).to(device="cuda")
        self.reset_parameters(self.AK, self.A.size(1))
        self.add_para_A = True

    def add_vocabs_d(self):
        self.vocabs_d = nn.Parameter(self.vocabs.new_zeros((len(self.vocabs), self.M, self.K + self.KK), dtype=torch.float).scatter_(-1, self.vocabs.unsqueeze(2), 1.0).view(-1, len(self.vocabs)), requires_grad=False)
        self.add_para_vocabsd = True

    # Initialize parameters of Code Embedding
    def reset_parameters(self, matrix, embedding_dim, factor=1.0):
        stdv = 1. / math.sqrt(embedding_dim)
        matrix.data.uniform_(-stdv, stdv)

    def generate_d(self, input, hidden_states, tau=1, training=True, eps=1e-12):
        # d is the corresponding one-hot vector of a code
        hidden_states = torch.tanh(self.h_w(hidden_states))
        alpha = self.alpha_w(hidden_states)

        if self.add_para_A:
            alpha_AK = self.alpha_w_AK(hidden_states)
            alpha = torch.cat((alpha, alpha_AK), dim=-1)
        alpha = F.softplus(alpha)

        # This rearranges alpha to be more intuitively BATCH_SIZE X M X K
        if not self.add_para_A:
            alpha = alpha.view(-1, self.M, self.K)
        else:
            alpha = alpha.view(-1, self.M, self.K+self.KK)

        # Take the log of all elements
        log_alpha = torch.log(alpha + eps)


        # We apply Gumbel-softmax trick to get code vectors d
        d = F.softmax((log_alpha + sample_gumbel(log_alpha)) / tau, dim=-1)

        if not training:
            # d BATCH_SIZE x M x K
            _, ind = d.max(dim=-1)

            # map ind to input vocab
            self.vocabs[input] = ind

            # Allows us when not training to convert soft vector to a hard, binarized one-hot encoding vector
            d = torch.zeros_like(d).scatter_(-1, ind.unsqueeze(2), 1.0)

        # d is now BATCH_SIZE x M x K x 1
        d = d.unsqueeze(-1)
        return d

    def extract_code(self, input:Tensor):
        return self.vocabs[input]

    # Operation necessary for proper batch matrix multiplication
    # OUTPUT SHAPE: BATCH_SIZE X M X EMBEDDING_DIM
    def embedding_d(self, d):
        if self.add_para_A:
            return torch.matmul(torch.cat((self.A, self.AK),dim=2), d).squeeze(-1)
        else:
            return torch.matmul(self.A, d).squeeze(-1)

    def embedding_code(self, codes):
        if self.add_para_A:
            emb = index_select(torch.cat((self.A, self.AK),dim=2), codes)
        else:
            emb = index_select(self.A, codes)

        emb = emb.sum(0)
        return emb

    def embedding(self, input:Tensor):
        codes = self.extract_code(input)
        return self.embedding_code(codes)

    def embedding_matrix(self):
        return self.embedding_code(self.vocabs)

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])    

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '('+ name +'): ' \
                    +'Parameter(' + str(tuple(p[1].shape))+ ', requires_grad='+ str(p[1].requires_grad) +')\n' 
        return string_repr

    def output_projection(self, hidden_states, tau=1, eps=1e-12):
        # d is the corresponding one-hot vector of a code
        bsz, seq_len, _ = hidden_states.size()
        hidden_states = torch.tanh(self.h_w(hidden_states))
        alpha = self.alpha_w(hidden_states)
        if self.add_para_A:
            alpha_AK = self.alpha_w_AK(hidden_states)
            alpha = torch.cat((alpha, alpha_AK), dim=-1)
        alpha = F.softplus(alpha)

        # This rearranges alpha to be more intuitively BATCH_SIZE X M X K
        if not self.add_para_A:
            alpha = alpha.view(-1, self.M, self.K)
        else:
            alpha = alpha.view(-1, self.M, self.K+self.KK)

        # Take the log of all elements
        log_alpha = torch.log(alpha + eps)

        # We apply Gumbel-softmax trick to get code vectors d
        d = F.softmax((log_alpha + sample_gumbel(log_alpha)) / tau, dim=-1)
        d = d.view(len(d), -1)

        logits = torch.mm(d, self.vocabs_d).view(bsz, seq_len, -1)
        return logits

def index_select(A, vocabs):
    #input: 
    #self.A  m x emb x k
    #self.vocabs V x m 
    A = A.transpose(1, 2)
    vocabs = vocabs.t()

    #output:
    #m x V x emb
    return A.gather(1, vocabs.unsqueeze(2).expand(vocabs.size(0), vocabs.size(1), A.size(2)))


class CodeBook(nn.Module):
    # code 20230808: better method to handle padding_idx? maybe set self.A[0] = 0?
    def __init__(self, num_embeddings, embedding_dim, padding_idx, distill_K, distill_M):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.head = Code(num_embeddings, embedding_dim, padding_idx, distill_K, distill_M)
        self.emb_updated, self.weight = False, None

    def forward(self, input:Tensor):
        input_size = input.size()
        input = input.view(-1)
        if self.head.add_para_vocabsd:
            emb = self.head.embedding(input)
            if self.padding_idx is not None:
                padding_mask = input == self.padding_idx
                with torch.no_grad():
                    emb[padding_mask].fill_(0)
        else:
            if not self.emb_updated:
                emb = self.embedding_matrix()
                self.weight = emb
                self.emb_updated = True

            emb = self.weight[input]

        emb = emb.view(*input_size, emb.size(-1))
        return emb

    def distill_embedding(self, input, hidden_states):
        input_size = input.size()
        input = input.view(-1)
        hidden_states = hidden_states.reshape(input.size(0), hidden_states.size(-1))
        output = self.head.embedding_d(self.head.generate_d(input, hidden_states))
        output = torch.sum(output, dim=1)
        output = output.view(*input_size, output.size(-1))
        return output

    def output_code(self, input, hidden_states):
        d = self.head.generate_d(input, hidden_states, training=False)
        return d

    def embedding_matrix(self):
        emb = self.head.embedding_matrix()

        # fill padding_idx with zero
        if self.padding_idx is not None:
            with torch.no_grad():
                emb[self.padding_idx].fill_(0)

        # emb is now VOCAB_SIZE x embedding_dim
        return emb


