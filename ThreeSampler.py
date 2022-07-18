# Code obtained from:
# https://github.com/CuriousAI/mean-teacher/blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/mean_teacher/data.py

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler



class ThreeStreamBatchSampler(Sampler):
    """Iterate three sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, third_indices, batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.third_indices = third_indices
        self.inter_batch_size = batch_size // 3

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        third_iter = iterate_eternally(self.third_indices)
        return (
            primary_batch + secondary_batch + third_batch
            for (primary_batch, secondary_batch, third_batch)
            in  zip(grouper(primary_iter, self.inter_batch_size),
                    grouper(secondary_iter, self.inter_batch_size),
                    grouper(third_iter, self.inter_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.inter_batch_size

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.inter_batch_size = batch_size // 2

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.inter_batch_size),
                    grouper(secondary_iter, self.inter_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.inter_batch_size


def iterate_once(iterable):
   
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
