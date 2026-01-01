# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

#code
import torch

@dataclass
class CrossEntropyDatamapCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy_datamap", dataclass=CrossEntropyDatamapCriterionConfig)
class CrossEntropyDatamapCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        #code
        self.count = torch.zeros(len(task.target_dictionary), dtype=torch.int, requires_grad=False)
        self.correctness = torch.zeros(len(task.target_dictionary), dtype=torch.int, requires_grad=False)
        if True:
            self.sum_x = torch.zeros(len(task.target_dictionary), requires_grad=False)
            self.sum_xsq = torch.zeros(len(task.target_dictionary), requires_grad=False)
        self.ones = torch.ones(len(task.target_dictionary), dtype=torch.int, requires_grad=False)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        #code
        if isinstance(net_output, tuple):
            net_output = net_output[0]

        if self.training:
            self.data_map(model, net_output, sample)

        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def data_map(self, model, net_output, sample):
        targets = model.get_targets(sample, net_output)
        targets = targets.view(-1).to('cpu')
        probs = F.softmax(net_output, dim=-1).to('cpu').detach()
        probs = probs.view(-1, probs.size(-1))
        predicts = torch.argmax(probs, dim=-1)
        gather_probs = probs.gather(dim=-1, index=targets.unsqueeze(1)).view(-1)
        gather_probs_sq = torch.square(gather_probs)
        self.count.scatter_add_(0, targets, self.ones[targets])
        self.correctness.scatter_add_(0, targets, (targets == predicts).int())
        self.sum_x.scatter_add_(0, targets, gather_probs)
        self.sum_xsq.scatter_add_(0, targets, gather_probs_sq)


    def compute_dict_loss(self, model, net_output, sample, reduce=True):
        definitions_emb, tokens_emb = net_output
        target = definitions_emb.new_ones((definitions_emb.size(0)))
        loss = F.cosine_embedding_loss(
            definitions_emb, 
            tokens_emb, 
            target,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = F.log_softmax(net_output, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

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
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
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
