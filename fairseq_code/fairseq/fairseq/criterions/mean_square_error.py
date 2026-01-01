# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

@dataclass
class MeanSquareErrorCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("mean_square_error", dataclass=MeanSquareErrorCriterionConfig)
class MeanSquareErrorCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.sentence_avg = args.sentence_avg
        self.eos_idx = task.target_dictionary.eos()
        self.target_encoder_embeddings = getattr(args, 'target_encoder_embeddings', False)  # 新增
        self.scale_factor = 100.0  # Scale factor for loss values

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        # fmt: on
        parser.add_argument('--target-encoder-embeddings', action="store_true",
                            help='use encoder embeddings instead of decoder embeddings for T5/T5Gemma models')
        
    @classmethod
    def build_criterion(cls, cfg: MeanSquareErrorCriterionConfig, task):
        return cls(cfg, task)

    def get_target_embeddings(self, target_model, sample):
        """
        Get target embeddings based on model architecture
        """
        with torch.no_grad():
            # Check model type and get appropriate embeddings
            model_type = type(target_model).__name__.lower()    

            # For T5Gemma or T5 models
            if 't5' in model_type or 'gemma' in model_type:
                if self.target_encoder_embeddings:
                    if hasattr(target_model, 'encoder') and hasattr(target_model.encoder, 'embed_tokens'):
                        target_embedding = target_model.encoder.embed_tokens(sample['target'])
                    elif hasattr(target_model, 'shared'):
                        # Some T5 models use shared embeddings
                        target_embedding = target_model.shared(sample['target'])
                    else:
                        raise AttributeError("Cannot find encoder embeddings in T5/T5Gemma model")
                else:
                    if hasattr(target_model, 'decoder') and hasattr(target_model.decoder, 'embed_tokens'):
                        target_embedding = target_model.decoder.embed_tokens(sample['target'])
                    elif hasattr(target_model, 'shared'):
                        # Some T5 models use shared embeddings
                        target_embedding = target_model.shared(sample['target'])
                    else:
                        raise AttributeError("Cannot find decoder embeddings in T5/T5Gemma model")
            
            # For BART models
            elif 'bart' in model_type:
                target_embedding = target_model.decoder.embed_tokens(sample['target'])
            
            # For RoBERTa models
            elif 'roberta' in model_type:
                target_embedding = target_model.encoder.sentence_encoder.embed_tokens(sample['target'])
            
            # Default fallback - try common patterns
            else:
                if hasattr(target_model, 'decoder') and hasattr(target_model.decoder, 'embed_tokens'):
                    target_embedding = target_model.decoder.embed_tokens(sample['target'])
                elif hasattr(target_model, 'encoder') and hasattr(target_model.encoder, 'embed_tokens'):
                    target_embedding = target_model.encoder.embed_tokens(sample['target'])
                elif hasattr(target_model, 'shared'):
                    target_embedding = target_model.shared(sample['target'])
                else:
                    raise AttributeError(f"Cannot find embeddings for model type: {model_type}")
        
        return target_embedding

    def forward(self, models, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #model: our compressed model
        #target_model: the pre-trained model that needs to be compressed
        torch.set_printoptions(profile="default")

        model, target_model, _ = models

        with torch.no_grad():
            target_embedding = self.get_target_embeddings(target_model, sample)

            reproduced_embedding, _ = model(**sample["net_input"], features_only=True)                    

            loss = self.compute_loss(reproduced_embedding, target_embedding, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # average loss across embedding dims
        loss = loss / reproduced_embedding.size(2) * self.scale_factor
        logging_output = {
            "loss": loss.data,
            #"mse_loss": mse_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, reproduced_embedding, target_embedding, sample, reduce=True, eps=1e-16):
        reproduced_embedding = reproduced_embedding.reshape(-1, reproduced_embedding.size(-1))
        target_embedding = target_embedding.view(-1, reproduced_embedding.size(-1))
        target = sample['target'].view(-1)
        pos_target_mask = sample['target_mask'].view(-1)
        pos_reproduced_embedding = reproduced_embedding[pos_target_mask]
        pos_target_embedding = target_embedding[pos_target_mask]

        loss = F.mse_loss(
            pos_reproduced_embedding,
            pos_target_embedding,
            reduction="sum" if reduce else "none",
        ) 
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
