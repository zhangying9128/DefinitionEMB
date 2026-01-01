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


@register_model("discriminator")
class Discriminator(BaseFairseqModel):

    def __init__(self, args):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(args.decoder_embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)
        logger.info(args)
        return cls(args)

    def forward(self, z):
        validity = self.main(z)
        return validity

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


@register_model_architecture("discriminator", "discriminator")
def base_architecture(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)


