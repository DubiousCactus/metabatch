#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Meta-dataset dataloader.
"""


import random
from multiprocessing import Manager
from typing import (
    Callable,
    Iterator,
    List,
    Optional,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

__all__ = ["TaskLoader"]


class SeededBatchSampler(Sampler):
    """
        Wraps another sampler to yield a mini-batch of indices, but seeds the dataset indexing to
        have consistant context + target set sizes across the batch.
        It is the default BatchSampler
        (https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler)
        modified to seed the context and target set sizes for all the elements of a batch
        (randomized at each batch).
    Args:
        dataset (Dataset): Dataset.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, dataset, batch_size, drop_last, shuffle: bool = False):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        if not isinstance(shuffle, bool):
            raise ValueError(
                "shuffle should be a boolean value, but got "
                "shuffle={}".format(shuffle)
            )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._sampling_inst = Manager().dict()
        dataset.register_sampling_inst(
            self._sampling_inst
        )  # Will modify every copy assigned to all workers, so they all get the same sampling_inst reference
        self.dataset = dataset
        self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    def new_batch(self):
        n_context = random.randint(self.dataset.min_pts, self.dataset.max_ctx_pts)
        n_targets = (
            (
                self.dataset.total_tgt_pts - n_context
                if self.dataset.eval
                else random.randint(n_context, self.dataset.max_tgt_pts)
            )
            if not self.dataset.predict_full_target
            else self.dataset.total_tgt_pts
        )
        return n_context, n_targets

    def __iter__(self) -> Iterator[List[int]]:
        """
        The sampler gets a copy of the dataset and just returns indices for the dataloader. I
        modified the default BatchSampler to seed the context and target sizes of each new batch
        for all workers, using a multiprocessing dictionary.
        """
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        ctx_pts, tgt_pts = self.new_batch()
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    for idx in batch:
                        self._sampling_inst[idx] = (ctx_pts, tgt_pts)
                    yield batch
                    ctx_pts, tgt_pts = self.new_batch()
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                self._sampling_inst[idx] = (ctx_pts, tgt_pts)
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
                    ctx_pts, tgt_pts = self.new_batch()
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class TaskLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        super().__init__(
            dataset,
            batch_sampler=SeededBatchSampler(dataset, batch_size, drop_last, shuffle),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
