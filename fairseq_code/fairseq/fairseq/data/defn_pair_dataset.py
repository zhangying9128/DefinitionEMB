# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, ConcatDataset, data_utils

import random
logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None, pad_idx=pad_idx):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )


    id = torch.LongTensor([s["id"] for s in samples])
    #code
    dataset_id = 0 if samples[0]["dataset_id"] == 0 else 1

    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src_tokens = src_tokens.index_select(0, sort_order)

    target = merge(
        "target",
        left_pad=left_pad_target,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    target = target.index_select(0, sort_order)

    tgt_lengths = torch.LongTensor(
        [s["target"].ne(pad_idx).long().sum() for s in samples]
    ).index_select(0, sort_order)

    prev_output_tokens = None
    if samples[0].get("prev_output_tokens", None) is not None:
        prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
    elif input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            "target",
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )

    #code
    target_mask = merge(
        "target_mask",
        left_pad=left_pad_target,
        pad_idx=False,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    target_mask = target_mask.index_select(0, sort_order)
    ntokens = target_mask.sum().item()


    batch = {
        "id": id,
        "dataset_id": dataset_id, #code
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }

    #code
    batch["tgt_lengths"] = tgt_lengths
    batch["target_mask"] = target_mask.to(torch.bool)


    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )


    return batch


class DefnPairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt,
        tgt_sizes,
        tgt_dict,
        tgt_masks, #code
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        filter_ratio=0, #code
        mask_context=False, #code
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.tgt_masks = tgt_masks #code
        self.filter_ratio = filter_ratio #code
        self.mask_context = mask_context #code
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]


        #code
        tgt_mask = self.tgt_masks[index]
        prev_output_tokens = None

        #mask context for roberta
        if self.mask_context:
            mask_idx = self.src_dict.index("50260")
            defn_range = torch.ne(tgt_item[:-1], self.tgt_dict.pad()).nonzero(as_tuple=True)[0]
            defn_source_tokens = src_item[defn_range].clone()
            defn_target_tokens = tgt_item[defn_range].clone()

            if True: #randomly mask
                sz = len(defn_range)
                positions = torch.randperm(sz)

                pred_prob, keep_prob, rand_prob = 1, 0.25, 0.25 # 0.25, 0.1, 0.1
                pred_size = round(sz * pred_prob)

                skip_range = range(sz - pred_size)
                defn_source_tokens[positions[skip_range]] = defn_target_tokens[positions[skip_range]].clone()
                defn_target_tokens[positions[skip_range]] = self.tgt_dict.pad()

                mask_range = range(sz - pred_size, sz)


                def mask_perm(masked_source_tokens, masked_target_tokens):

                    mask_prob = 1.0 - rand_prob - keep_prob
                    mask_indices = torch.bernoulli(torch.full(masked_source_tokens.shape, mask_prob)).bool()
                    random_indices = torch.bernoulli(torch.full(masked_source_tokens.shape, rand_prob / (1.0 - mask_prob))).bool() & ~mask_indices

                    masked_source_tokens[mask_indices] = mask_idx
                    masked_source_tokens[random_indices] = torch.from_numpy(
                        np.random.choice(np.random.randint(4, mask_idx - 1), random_indices.sum().tolist())
                    ).to(masked_source_tokens.device)

                    return masked_source_tokens

                def merge(masks, tokens):
                    out = torch.stack([masks, tokens]).t().flatten()
                    return out
            
                masked_source_tokens = mask_perm(defn_target_tokens[positions[mask_range]].clone(), defn_target_tokens[positions[mask_range]].clone())

                defn_source_tokens[positions[mask_range]] = masked_source_tokens
                src_item[defn_range] = defn_source_tokens
                tgt_item[defn_range] = defn_target_tokens




            max_idx = int((1 - self.filter_ratio) * len(self.tgt_dict))
            tgt_mask = torch.logical_or(torch.logical_and(torch.gt(tgt_item, self.tgt_dict.eos()), torch.lt(tgt_item, max_idx)), torch.ge(tgt_item, mask_idx - 1))


        elif not self.input_feeding:
            # for gpt2
            s = torch.cat((src_item[:-1], tgt_item[-1].unsqueeze(0), tgt_item[:-2]), dim=0)
            t = torch.cat((src_item[:-1], tgt_item[:-1]), dim=0)
            src_item, tgt_item = s, t

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
                tgt_mask = torch.cat([self.tgt_masks[index], torch.LongTensor(False)])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])
                tgt_mask = torch.cat([torch.LongTensor(False)], self.tgt_masks[index])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        #code
        dataset_id = 0
        if isinstance(self.src, ConcatDataset):
            dataset_id = self.src.dataset_id(index)
        assert len(tgt_item) == len(tgt_mask)

        example = {
            "id": index,
            "dataset_id": dataset_id,#code
            "source": src_item,
            "target": tgt_item,
            "target_mask": tgt_mask, #code
            "prev_output_tokens": prev_output_tokens, #code
        }
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        torch.set_printoptions(profile="full")
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

    #code
    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):

        #if not self.batch_sample:
        if not isinstance(self.src, ConcatDataset):
            return super().batch_by_size(
                indices, max_tokens, max_sentences, required_batch_size_multiple
            )

        #dataset_indices = {key: [] for key in self.datasets}
        dataset_indices = {key: [] for key in range(len(self.src.datasets))}
        for i in indices:
            #_, key = self._map_index(i)
            key, _ = self.src._get_dataset_and_sample_index(i)
            dataset_indices[key].append(i)

        batches = []
        for key in dataset_indices:
            cur_batches = super().batch_by_size(
                np.array(dataset_indices[key], dtype=np.int64),
                max_tokens,
                max_sentences,
                required_batch_size_multiple,
            )
            logger.info(f"Created {len(cur_batches)} batches for dataset {key}")
            #batches += cur_batches
            batches = cur_batches + batches

        # If this dataset is used in a distributed training setup,
        # then shuffle such that the order is seeded by the distributed rank
        # as well
        #if self.distributed_rank is not None:
        #    with data_utils.numpy_seed(self.seed, self.epoch, self.distributed_rank):
        #        np.random.shuffle(batches)
        return batches
