# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_model import BaseFairseqModel

from fairseq.models.bart.hub_interface import BARTHubInterface

import math

logger = logging.getLogger(__name__)


@register_model("linear")
class LinearModel(BaseFairseqModel):

    def __init__(self, args):
        super().__init__()

        self.w1 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim // 2, bias=False)
        self.w2 = nn.Linear(args.encoder_embed_dim // 2, args.encoder_embed_dim, bias=False)
        self.w1.weight.data.normal_(mean=0.0, std=(args.encoder_embed_dim) ** -0.5)
        self.w2.weight.data.normal_(mean=0.0, std=(args.encoder_embed_dim) ** -0.5)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        logger.info(args)
        return cls(args)

    def forward(
        self,
        embeddings,
    ):
        x = F.relu(self.w1(embeddings))
        x = self.w2(x)

        return x

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])


@register_model_architecture("linear", "linear")
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)


