# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import random

from . import BaseWrapperDataset

def random_not_equal_to(exclude, low, high):
    while True:
        num = random.randint(low, high)
        if num != exclude:
            return num

class AppendRandomTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._sizes = np.array(dataset.sizes) + 4

    def __getitem__(self, idx, split="train"):
        item = self.dataset[idx]
        right_add = item.new([5457])
        if split == "train":
            #add_left = np.random.randint(0, 2)
            add_left = False
            if add_left:
                left_random = random_not_equal_to(5457, 4, 50260)
                left_add = item.new([left_random, 5457, left_random, 3])
                item = torch.cat([item[0].unsqueeze(-1), left_add, item[1:-1], right_add, item[-1].unsqueeze(-1)], dim=0)
            else:
                item = torch.cat([item[:-1], right_add, item[-1].unsqueeze(-1)], dim=0)
        else:
            item = torch.cat([item[:-1], right_add, item[-1].unsqueeze(-1)], dim=0)
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n
